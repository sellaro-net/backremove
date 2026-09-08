use std::{
    collections::{HashMap, HashSet},
    io::{self, Cursor, Write},
    num::NonZeroU64,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Instant,
};

use fast_image_resize::{
    FilterType as ResizeFilter, PixelType, ResizeAlg, ResizeOptions, Resizer,
    images::{Image, ImageRef},
};
use image::{
    ColorType, ImageDecoder, ImageEncoder, ImageFormat, Limits,
    codecs::png::{CompressionType, FilterType, PngEncoder},
};
use resvg::{tiny_skia, usvg};

use crate::{
    config::ImageLimits,
    error::{AppError, Result},
    types::{
        DecodedImage, EncodedImage, ImageInfo, InputFormat, InputTensor, Mask, MaskGeometry, Model,
        ModelSpec, Precision, PreparedImage,
    },
};

// A dimension bound is necessary in addition to a pixel bound: codec scanline,
// resize-coefficient and SVG integer-coordinate storage also depends on each axis.
const MAX_SIDE: u32 = 32_768;
const MIB: usize = 1024 * 1024;
const HEADER_BUDGET: u64 = 16 * MIB as u64;
const SVG_NAMESPACE: &str = "http://www.w3.org/2000/svg";

pub struct ImagePipeline {
    limits: ImageLimits,
    fonts: Arc<usvg::fontdb::Database>,
}

impl ImagePipeline {
    pub fn new(limits: ImageLimits, font_paths: &[PathBuf]) -> Result<Self> {
        let mut fonts = usvg::fontdb::Database::new();
        for path in font_paths {
            let metadata = std::fs::metadata(path).map_err(|error| {
                tracing::error!(%error, "SVG-Schriftdatei ist nicht lesbar");
                AppError::Internal
            })?;
            if !metadata.is_file() || metadata.len() > 32 * MIB as u64 {
                return Err(AppError::Internal);
            }
            let data = std::fs::read(path).map_err(|error| {
                tracing::error!(%error, "SVG-Schriftdatei konnte nicht geladen werden");
                AppError::Internal
            })?;
            fonts.load_font_data(data);
        }
        if fonts.faces().next().is_none() {
            tracing::error!("Das verifizierte SVG-Schriftpaket enthält keine lesbare Schrift");
            return Err(AppError::Internal);
        }
        fonts.set_sans_serif_family("Noto Sans");
        fonts.set_serif_family("Noto Serif");
        fonts.set_monospace_family("Noto Sans Mono");
        fonts.set_cursive_family("Noto Sans");
        fonts.set_fantasy_family("Noto Sans");
        Ok(Self {
            limits,
            fonts: Arc::new(fonts),
        })
    }

    pub fn inspect(&self, bytes: &[u8], content_type: &str) -> Result<ImageInfo> {
        if bytes.len() > self.limits.max_file_bytes {
            return Err(AppError::TooLarge);
        }
        let declared = content_type.split(';').next().unwrap_or("").trim();
        let format = match declared {
            "image/svg+xml" => InputFormat::Svg,
            "image/jpeg" | "image/png" | "image/webp" | "image/gif" | "image/avif" => {
                // image's magic table recognizes only major-brand `avif`, not
                // `avis` or `mif1` with an AVIF compatible brand. mp4parse performs
                // the authoritative brand/container validation below.
                let detected = if bytes.get(4..8) == Some(b"ftyp") {
                    ImageFormat::Avif
                } else {
                    image::guess_format(bytes).map_err(invalid_image)?
                };
                match detected {
                    ImageFormat::Jpeg => InputFormat::Jpeg,
                    ImageFormat::Png => InputFormat::Png,
                    ImageFormat::WebP => InputFormat::Webp,
                    ImageFormat::Gif => InputFormat::Gif,
                    ImageFormat::Avif => InputFormat::Avif,
                    _ => return Err(AppError::UnsupportedMedia),
                }
            }
            _ => return Err(AppError::UnsupportedMedia),
        };
        let (width, height, orientation, extra, multiplier) = match format {
            InputFormat::Svg => {
                let document = self.svg_document(bytes)?;
                let budget = self.check_svg(&document)?;
                let options = self.svg_options(false, Arc::new(Mutex::new(None)));
                let tree = usvg::Tree::from_xmltree(&document, &options).map_err(invalid_image)?;
                let size = tree.size().to_int_size();
                self.check_dimensions(size.width(), size.height())?;
                let scratch = self.svg_scratch(
                    tree.root(),
                    0,
                    &mut 0usize,
                    tiny_skia::Transform::identity(),
                )?;
                // Every simultaneously live isolated layer may reach the canvas
                // bound. Filter primitive outputs remain live for later references.
                let layers = budget
                    .depth
                    .checked_mul(4)
                    .and_then(|n| n.checked_add(budget.filter_primitives.checked_mul(8)?))
                    .ok_or(AppError::TooLarge)?;
                let extra = budget
                    .embedded_memory
                    .checked_add(bytes.len().checked_mul(32).ok_or(AppError::TooLarge)?)
                    .and_then(|n| n.checked_add(scratch))
                    .ok_or(AppError::TooLarge)?;
                (
                    size.width(),
                    size.height(),
                    1,
                    extra,
                    16usize.checked_add(layers).ok_or(AppError::TooLarge)?,
                )
            }
            InputFormat::Avif => {
                let avif = self.avif_context(bytes)?;
                let (mut width, mut height) = avif_dimensions(&avif)?;
                if matches!(
                    avif.image_rotation().map_err(invalid_image)?,
                    mp4parse::ImageRotation::D90 | mp4parse::ImageRotation::D270
                ) {
                    std::mem::swap(&mut width, &mut height);
                }
                (
                    width,
                    height,
                    1,
                    bytes.len().checked_mul(4).ok_or(AppError::TooLarge)?,
                    64,
                )
            }
            InputFormat::Gif => {
                let mut decoder = gif_reader(bytes, HEADER_BUDGET)?;
                let (mut width, mut height) =
                    (u32::from(decoder.width()), u32::from(decoder.height()));
                self.check_dimensions(width, height)?;
                let frame = decoder
                    .next_frame_info()
                    .map_err(invalid_image)?
                    .ok_or(AppError::InvalidImage)?;
                self.check_dimensions(u32::from(frame.width), u32::from(frame.height))?;
                width = width.max(u32::from(frame.left) + u32::from(frame.width));
                height = height.max(u32::from(frame.top) + u32::from(frame.height));
                (width, height, 1, 0, 16)
            }
            _ => {
                let (width, height, orientation) = self.raster_header(bytes, format)?;
                (width, height, orientation, 0, 24)
            }
        };
        let pixels = self.check_dimensions(width, height)?;
        let estimated_memory_bytes = pixels
            .checked_mul(multiplier)
            .and_then(|n| n.checked_add(64 * MIB))
            .and_then(|n| n.checked_add(self.limits.max_output_bytes))
            .and_then(|n| n.checked_add(extra))
            .ok_or(AppError::TooLarge)?;
        Ok(ImageInfo {
            width,
            height,
            format,
            orientation,
            estimated_memory_bytes,
        })
    }

    pub fn decode(&self, bytes: &[u8], info: ImageInfo, model: Model) -> Result<DecodedImage> {
        if bytes.len() > self.limits.max_file_bytes {
            return Err(AppError::TooLarge);
        }
        self.check_dimensions(info.width, info.height)?;
        let mut image = match info.format {
            InputFormat::Svg => self.decode_svg(bytes, info.estimated_memory_bytes)?,
            InputFormat::Gif => self.decode_gif(bytes)?.into_rgb()?,
            InputFormat::Avif => self.decode_avif(bytes)?.into_rgb()?,
            _ => self.decode_raster(bytes, info.format)?.into_rgb()?,
        };
        if image.width != info.width || image.height != info.height {
            return Err(AppError::InvalidImage);
        }
        if matches!(model, Model::Fast) {
            image = orient_rgb(image, info.orientation)?;
        }
        Ok(image)
    }

    pub fn prepare(&self, image: DecodedImage, spec: &ModelSpec) -> Result<PreparedImage> {
        self.check_rgb(&image)?;
        let side = spec.input_size;
        // Artifact validation enforces these, but this public entry point must not
        // permit an unbounded tensor allocation even with a malformed caller.
        if side == 0 || side > 1024 {
            return Err(AppError::Internal);
        }
        let (width, height) = match spec.model {
            Model::Fast => {
                let scale = f64::from(side) / f64::from(image.width.max(image.height));
                (
                    (f64::from(image.width) * scale).round_ties_even().max(1.0) as u32,
                    (f64::from(image.height) * scale).round_ties_even().max(1.0) as u32,
                )
            }
            Model::Quality => (side, side),
        };
        let resized = resize(
            &image.pixels,
            image.width,
            image.height,
            width,
            height,
            PixelType::U8x3,
        )?;
        let plane = area(side, side)?;
        let count = plane.checked_mul(3).ok_or(AppError::Internal)?;
        let mut tensor = zeroed::<f32>(count)?;
        let mean = [0.485_f32, 0.456, 0.406];
        let deviation = [0.229_f32, 0.224, 0.225];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let source = (y * width as usize + x) * 3;
                let target = y * side as usize + x;
                for channel in 0..3 {
                    let value = f32::from(resized[source + channel]) / 255.0;
                    tensor[channel * plane + target] = match spec.model {
                        Model::Fast => value,
                        Model::Quality => (value - mean[channel]) / deviation[channel],
                    };
                }
            }
        }
        let tensor = match spec.precision {
            Precision::Fp32 => InputTensor::F32(tensor),
            Precision::Fp16 => {
                let mut half = zeroed::<half::f16>(count)?;
                for (target, source) in half.iter_mut().zip(tensor) {
                    *target = half::f16::from_f32(source);
                }
                InputTensor::F16(half)
            }
        };
        Ok(PreparedImage {
            image,
            tensor,
            geometry: MaskGeometry {
                crop_width: width,
                crop_height: height,
            },
        })
    }

    pub fn encode(
        &self,
        image: DecodedImage,
        mask: Mask,
        geometry: MaskGeometry,
        spec: &ModelSpec,
    ) -> Result<EncodedImage> {
        let postprocess_started = Instant::now();
        self.check_rgb(&image)?;
        if mask.width == 0
            || mask.height == 0
            || mask.width > 1024
            || mask.height > 1024
            || mask.values.len() != area(mask.width, mask.height)?
            || spec.input_size == 0
            || geometry.crop_width == 0
            || geometry.crop_height == 0
            || geometry.crop_width > spec.input_size
            || geometry.crop_height > spec.input_size
            || mask.values.iter().any(|value| !value.is_finite())
        {
            return Err(AppError::Internal);
        }
        let (crop_width, crop_height) = match spec.model {
            Model::Fast => (
                (f64::from(geometry.crop_width) * f64::from(mask.width)
                    / f64::from(spec.input_size))
                .round_ties_even()
                .max(1.0) as u32,
                (f64::from(geometry.crop_height) * f64::from(mask.height)
                    / f64::from(spec.input_size))
                .round_ties_even()
                .max(1.0) as u32,
            ),
            Model::Quality => (mask.width, mask.height),
        };
        if crop_width > mask.width || crop_height > mask.height {
            return Err(AppError::Internal);
        }
        let mut alpha = zeroed::<u8>(area(crop_width, crop_height)?)?;
        for y in 0..crop_height as usize {
            for x in 0..crop_width as usize {
                // NumPy clips/scales float32, truncates to u8, THEN Pillow resizes.
                alpha[y * crop_width as usize + x] =
                    (mask.values[y * mask.width as usize + x] * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
        drop(mask);
        let alpha = resize(
            &alpha,
            crop_width,
            crop_height,
            image.width,
            image.height,
            PixelType::U8,
        )?;
        let mut rgba = zeroed::<u8>(
            area(image.width, image.height)?
                .checked_mul(4)
                .ok_or(AppError::Internal)?,
        )?;
        for ((target, rgb), alpha) in rgba
            .chunks_exact_mut(4)
            .zip(image.pixels.chunks_exact(3))
            .zip(alpha)
        {
            target[..3].copy_from_slice(rgb);
            target[3] = alpha;
        }
        let (width, height) = (image.width, image.height);
        drop(image);
        let postprocess_ms = postprocess_started.elapsed().as_secs_f64() * 1000.0;
        let encode_started = Instant::now();
        let bytes = encode_png(&rgba, width, height, self.limits.max_output_bytes)?;
        Ok(EncodedImage {
            bytes,
            postprocess_ms,
            encode_ms: encode_started.elapsed().as_secs_f64() * 1000.0,
        })
    }

    fn check_dimensions(&self, width: u32, height: u32) -> Result<usize> {
        if width == 0 || height == 0 {
            return Err(AppError::InvalidImage);
        }
        if width > MAX_SIDE
            || height > MAX_SIDE
            || u64::from(width) * u64::from(height) > self.limits.max_pixels
        {
            return Err(AppError::TooLarge);
        }
        area(width, height)
    }

    fn check_rgb(&self, image: &DecodedImage) -> Result<()> {
        let pixels = self.check_dimensions(image.width, image.height)?;
        if image.pixels.len() != pixels.checked_mul(3).ok_or(AppError::Internal)? {
            return Err(AppError::Internal);
        }
        Ok(())
    }

    fn raster_header(&self, bytes: &[u8], format: InputFormat) -> Result<(u32, u32, u8)> {
        let mut orientation = 1;
        let (width, height) = match format {
            InputFormat::Png => {
                if bytes.get(..8) != Some(b"\x89PNG\r\n\x1a\n")
                    || bytes.get(12..16) != Some(b"IHDR")
                    || be32(bytes, 8)? != 13
                {
                    return Err(AppError::InvalidImage);
                }
                let dimensions = (be32(bytes, 16)?, be32(bytes, 20)?);
                self.check_dimensions(dimensions.0, dimensions.1)?;
                let mut offset = 8usize;
                while offset < bytes.len() {
                    let size = be32(bytes, offset)? as usize;
                    let end = offset
                        .checked_add(12)
                        .and_then(|n| n.checked_add(size))
                        .ok_or(AppError::TooLarge)?;
                    let payload = bytes
                        .get(offset + 8..end - 4)
                        .ok_or(AppError::InvalidImage)?;
                    let kind = bytes
                        .get(offset + 4..offset + 8)
                        .ok_or(AppError::InvalidImage)?;
                    if kind == b"eXIf" {
                        orientation = exif_orientation(payload);
                    }
                    if kind == b"fcTL" {
                        self.check_dimensions(be32(payload, 4)?, be32(payload, 8)?)?;
                    }
                    offset = end;
                    if kind == b"IEND" {
                        break;
                    }
                }
                dimensions
            }
            InputFormat::Jpeg => {
                if bytes.get(..2) != Some(b"\xff\xd8") {
                    return Err(AppError::InvalidImage);
                }
                let mut offset = 2usize;
                let mut dimensions = None;
                while offset < bytes.len() {
                    if bytes[offset] != 0xff {
                        return Err(AppError::InvalidImage);
                    }
                    while bytes.get(offset) == Some(&0xff) {
                        offset += 1;
                    }
                    let marker = *bytes.get(offset).ok_or(AppError::InvalidImage)?;
                    offset += 1;
                    if marker == 0xda || marker == 0xd9 {
                        break;
                    }
                    if marker == 0x01 || (0xd0..=0xd7).contains(&marker) {
                        continue;
                    }
                    let size = be16(bytes, offset)? as usize;
                    if size < 2 {
                        return Err(AppError::InvalidImage);
                    }
                    let payload = bytes
                        .get(offset + 2..offset.checked_add(size).ok_or(AppError::TooLarge)?)
                        .ok_or(AppError::InvalidImage)?;
                    if matches!(marker, 0xc0..=0xc3 | 0xc5..=0xc7 | 0xc9..=0xcb | 0xcd..=0xcf) {
                        let found = (u32::from(be16(payload, 3)?), u32::from(be16(payload, 1)?));
                        self.check_dimensions(found.0, found.1)?;
                        if dimensions.is_some_and(|prior| prior != found) {
                            return Err(AppError::InvalidImage);
                        }
                        dimensions = Some(found);
                    }
                    if marker == 0xe1 && payload.starts_with(b"Exif\0\0") {
                        orientation = exif_orientation(&payload[6..]);
                    }
                    offset += size;
                }
                dimensions.ok_or(AppError::InvalidImage)?
            }
            InputFormat::Webp => {
                if bytes.get(..4) != Some(b"RIFF") || bytes.get(8..12) != Some(b"WEBP") {
                    return Err(AppError::InvalidImage);
                }
                let end = (le32(bytes, 4)? as usize)
                    .checked_add(8)
                    .ok_or(AppError::TooLarge)?;
                if end > bytes.len() || end < 12 {
                    return Err(AppError::InvalidImage);
                }
                let mut offset = 12usize;
                let mut dimensions: Option<(u32, u32)> = None;
                while offset < end {
                    let size = le32(bytes, offset + 4)? as usize;
                    let payload_end = offset
                        .checked_add(8)
                        .and_then(|n| n.checked_add(size))
                        .ok_or(AppError::TooLarge)?;
                    let payload = bytes
                        .get(offset + 8..payload_end)
                        .filter(|_| payload_end <= end)
                        .ok_or(AppError::InvalidImage)?;
                    let kind = &bytes[offset..offset + 4];
                    let found = match kind {
                        b"VP8X" => Some((le24(payload, 4)? + 1, le24(payload, 7)? + 1)),
                        b"VP8 " => Some(webp_lossy_dimensions(payload)?),
                        b"VP8L" => Some(webp_lossless_dimensions(payload)?),
                        b"ANMF" => {
                            let width = le24(payload, 6)? + 1;
                            let height = le24(payload, 9)? + 1;
                            self.check_dimensions(width, height)?;
                            let canvas = dimensions.ok_or(AppError::InvalidImage)?;
                            if le24(payload, 0)?
                                .checked_mul(2)
                                .and_then(|x| x.checked_add(width))
                                .is_none_or(|x| x > canvas.0)
                                || le24(payload, 3)?
                                    .checked_mul(2)
                                    .and_then(|y| y.checked_add(height))
                                    .is_none_or(|y| y > canvas.1)
                            {
                                return Err(AppError::InvalidImage);
                            }
                            check_webp_frame(
                                payload.get(16..).ok_or(AppError::InvalidImage)?,
                                width,
                                height,
                            )?;
                            None
                        }
                        b"EXIF" => {
                            orientation = exif_orientation(payload);
                            None
                        }
                        _ => None,
                    };
                    if let Some(found) = found {
                        self.check_dimensions(found.0, found.1)?;
                        if dimensions.is_some_and(|prior| prior != found) {
                            return Err(AppError::InvalidImage);
                        }
                        dimensions = Some(found);
                    }
                    offset = payload_end
                        .checked_add(size & 1)
                        .ok_or(AppError::TooLarge)?;
                }
                dimensions.ok_or(AppError::InvalidImage)?
            }
            _ => return Err(AppError::Internal),
        };
        self.check_dimensions(width, height)?;
        Ok((width, height, orientation))
    }

    fn decode_raster(&self, bytes: &[u8], format: InputFormat) -> Result<Raster> {
        let (width, height, _) = self.raster_header(bytes, format)?;
        let allocation = self
            .check_dimensions(width, height)?
            .checked_mul(16)
            .and_then(|n| n.checked_add(16 * MIB))
            .ok_or(AppError::TooLarge)?;
        let mut limits = Limits::default();
        limits.max_image_width = Some(width);
        limits.max_image_height = Some(height);
        limits.max_alloc = Some(allocation as u64);
        let source = Cursor::new(bytes);
        match format {
            InputFormat::Png => decode_decoder(
                image::codecs::png::PngDecoder::with_limits(source, limits)
                    .map_err(invalid_image)?,
                width,
                height,
                true,
                false,
            ),
            InputFormat::Jpeg => {
                let mut decoder =
                    image::codecs::jpeg::JpegDecoder::new(source).map_err(invalid_image)?;
                decoder.set_limits(limits).map_err(invalid_image)?;
                decode_decoder(decoder, width, height, false, false)
            }
            InputFormat::Webp => decode_decoder(
                image::codecs::webp::WebPDecoder::new(source).map_err(invalid_image)?,
                width,
                height,
                false,
                false,
            ),
            _ => Err(AppError::Internal),
        }
    }

    fn decode_gif(&self, bytes: &[u8]) -> Result<Raster> {
        let max_frame = self
            .limits
            .max_pixels
            .checked_mul(4)
            .ok_or(AppError::TooLarge)?;
        let mut decoder = gif_reader(bytes, max_frame)?;
        let (mut width, mut height) = (u32::from(decoder.width()), u32::from(decoder.height()));
        self.check_dimensions(width, height)?;
        let frame = decoder
            .next_frame_info()
            .map_err(invalid_image)?
            .ok_or(AppError::InvalidImage)?;
        let (left, top, fw, fh, transparent) = (
            u32::from(frame.left),
            u32::from(frame.top),
            u32::from(frame.width),
            u32::from(frame.height),
            frame.transparent,
        );
        let frame_pixels = self.check_dimensions(fw, fh)?;
        width = width.max(left + fw);
        height = height.max(top + fh);
        let pixels = self.check_dimensions(width, height)?;
        let palette = decoder.palette().map_err(invalid_image)?.to_vec();
        let background = transparent.unwrap_or(0);
        let color = palette_color(&palette, background)?;
        let mut rgba = zeroed::<u8>(pixels.checked_mul(4).ok_or(AppError::TooLarge)?)?;
        for pixel in rgba.chunks_exact_mut(4) {
            pixel.copy_from_slice(&[
                color[0],
                color[1],
                color[2],
                if transparent == Some(background) {
                    0
                } else {
                    255
                },
            ]);
        }
        let mut indices = zeroed::<u8>(frame_pixels)?;
        decoder
            .read_into_buffer(&mut indices)
            .map_err(invalid_image)?;
        for y in 0..fh as usize {
            for x in 0..fw as usize {
                let index = indices[y * fw as usize + x];
                let color = palette_color(&palette, index)?;
                let offset = ((top as usize + y) * width as usize + left as usize + x) * 4;
                rgba[offset..offset + 4].copy_from_slice(&[
                    color[0],
                    color[1],
                    color[2],
                    if transparent == Some(index) { 0 } else { 255 },
                ]);
            }
        }
        Ok(Raster {
            width,
            height,
            pixels: rgba,
            channels: 4,
        })
    }

    fn avif_context(&self, bytes: &[u8]) -> Result<mp4parse::AvifContext> {
        preflight_avif_container(bytes)?;
        let context =
            mp4parse::read_avif(&mut Cursor::new(bytes), mp4parse::ParseStrictness::Normal)
                .map_err(invalid_image)?;
        let (width, height) = avif_dimensions(&context)?;
        self.check_dimensions(width, height)?;
        // image's AVIF constructor decodes immediately, before ImageDecoder::set_limits.
        // Inspect EVERY OBU rather than trusting ispe or only the last sequence header.
        let primary = self.check_av1(
            context
                .primary_item_coded_data()
                .ok_or(AppError::InvalidImage)?,
            width,
            height,
        )?;
        if let Some(alpha) = context.alpha_item_coded_data() {
            let alpha = self.check_av1(alpha, width, height)?;
            // The pinned image decoder interprets the alpha plane using the
            // primary plane's storage depth. Reject incompatible planes, not RGB
            // silently read with the wrong sample width.
            if alpha.0 != primary.0 || !alpha.1 {
                return Err(AppError::InvalidImage);
            }
        }
        Ok(context)
    }

    fn check_av1(&self, bytes: &[u8], width: u32, height: u32) -> Result<(u8, bool)> {
        let mut offset = 0usize;
        let mut sequences = 0usize;
        let mut frames = 0usize;
        let mut sample_format = (0, false);
        while offset < bytes.len() {
            let start = offset;
            let header = bytes[offset];
            offset += 1;
            if header & 0x81 != 0 || header & 2 == 0 {
                return Err(AppError::InvalidImage);
            }
            let kind = (header >> 3) & 15;
            if header & 4 != 0 {
                let extension = *bytes.get(offset).ok_or(AppError::InvalidImage)?;
                if extension & 7 != 0 {
                    return Err(AppError::InvalidImage);
                }
                offset += 1;
            }
            let size = leb128(bytes, &mut offset)?;
            offset = offset
                .checked_add(size)
                .filter(|end| *end <= bytes.len())
                .ok_or(AppError::InvalidImage)?;
            if kind == 1 {
                let mut sequence =
                    std::mem::MaybeUninit::<dav1d_sys::Dav1dSequenceHeader>::uninit();
                // SAFETY: dav1d writes the complete sequence header on success; the
                // bounded OBU slice is live for the entire synchronous native call.
                let status = unsafe {
                    dav1d_sys::dav1d_parse_sequence_header(
                        sequence.as_mut_ptr(),
                        bytes[start..offset].as_ptr(),
                        offset - start,
                    )
                };
                if status < 0 {
                    return Err(AppError::InvalidImage);
                }
                let sequence = unsafe { sequence.assume_init() };
                let sw = u32::try_from(sequence.max_width).map_err(|_| AppError::InvalidImage)?;
                let sh = u32::try_from(sequence.max_height).map_err(|_| AppError::InvalidImage)?;
                self.check_dimensions(sw, sh)?;
                if sw != width || sh != height {
                    return Err(AppError::InvalidImage);
                }
                sample_format = (sequence.hbd, sequence.monochrome != 0);
                sequences += 1;
            }
            if kind == 3 || kind == 6 {
                frames += 1;
            }
            // AVIF primary/alpha items are still pictures (including the primary
            // picture of an animation). No inter-frame reference history is needed.
            if sequences > 1 || frames > 1 {
                return Err(AppError::InvalidImage);
            }
        }
        if sequences != 1 || frames != 1 {
            return Err(AppError::InvalidImage);
        }
        Ok(sample_format)
    }

    fn decode_avif(&self, bytes: &[u8]) -> Result<Raster> {
        let context = self.avif_context(bytes)?;
        let (width, height) = avif_dimensions(&context)?;
        let decoder =
            image::codecs::avif::AvifDecoder::new(Cursor::new(bytes)).map_err(invalid_image)?;
        let mut raster =
            decode_decoder(decoder, width, height, false, context.premultiplied_alpha)?;
        let rotation = match context.image_rotation().map_err(invalid_image)? {
            mp4parse::ImageRotation::D0 => 1,
            mp4parse::ImageRotation::D90 => 8,
            mp4parse::ImageRotation::D180 => 3,
            mp4parse::ImageRotation::D270 => 6,
        };
        raster = orient_raster(raster, rotation)?;
        let mirror = context.image_mirror_ptr().map_err(invalid_image)?;
        if !mirror.is_null() {
            // SAFETY: pointer is borrowed from the live immutable AvifContext.
            raster = orient_raster(
                raster,
                match unsafe { &*mirror } {
                    mp4parse::ImageMirror::TopBottom => 4,
                    mp4parse::ImageMirror::LeftRight => 2,
                },
            )?;
        }
        Ok(raster)
    }

    fn svg_document<'a>(&self, bytes: &'a [u8]) -> Result<roxmltree::Document<'a>> {
        let text = std::str::from_utf8(bytes).map_err(invalid_image)?;
        let document = roxmltree::Document::parse_with_options(
            text,
            roxmltree::ParsingOptions {
                allow_dtd: false,
                nodes_limit: u32::try_from(self.limits.max_svg_nodes).unwrap_or(u32::MAX),
                ..Default::default()
            },
        )
        .map_err(|error| match error {
            roxmltree::Error::NodesLimitReached => AppError::TooLarge,
            _ => invalid_image(error),
        })?;
        let root = document.root_element();
        if root.tag_name().name() != "svg"
            || !matches!(root.tag_name().namespace(), None | Some(SVG_NAMESPACE))
        {
            return Err(AppError::InvalidImage);
        }
        Ok(document)
    }

    fn check_svg(&self, document: &roxmltree::Document<'_>) -> Result<SvgBudget> {
        let mut budget = SvgBudget::default();
        let mut ids = HashMap::new();
        let mut edges: HashMap<roxmltree::NodeId, Vec<roxmltree::NodeId>> = HashMap::new();
        let mut references = Vec::new();
        let mut embedded_bytes = 0usize;
        let mut embedded_pixels = 0u64;
        let mut text_bytes = 0usize;
        let mut drawing_bytes = 0usize;
        for node in document.descendants() {
            let depth = node.ancestors().count();
            if depth > self.limits.max_svg_depth {
                return Err(AppError::TooLarge);
            }
            budget.depth = budget.depth.max(depth);
            if node.is_pi() {
                return Err(AppError::InvalidImage);
            }
            if node.is_text() {
                text_bytes = text_bytes
                    .checked_add(node.text().unwrap_or("").len())
                    .ok_or(AppError::TooLarge)?;
                if text_bytes > self.limits.max_svg_nodes {
                    return Err(AppError::TooLarge);
                }
            }
            if !node.is_element() {
                continue;
            }
            let tag = node.tag_name().name();
            if matches!(
                tag,
                "script"
                    | "foreignObject"
                    | "animate"
                    | "animateMotion"
                    | "animateTransform"
                    | "set"
                    | "discard"
                    | "font"
                    | "font-face"
            ) {
                return Err(AppError::InvalidImage);
            }
            if tag.starts_with("fe") {
                budget.filter_primitives += 1;
            }
            if let Some(id) = node.attribute("id")
                && ids.insert(id, node.id()).is_some()
            {
                return Err(AppError::InvalidImage);
            }
            let children = node
                .children()
                .filter(|child| child.is_element())
                .map(|child| child.id())
                .collect();
            edges.insert(node.id(), children);
            if tag == "style" {
                check_css(node.text().unwrap_or(""), &mut references, node.id())?;
            }
            for attribute in node.attributes() {
                let name = attribute.name();
                let value = attribute.value().trim();
                if matches!(name, "d" | "points" | "kernelMatrix" | "tableValues") {
                    drawing_bytes = drawing_bytes
                        .checked_add(value.len())
                        .ok_or(AppError::TooLarge)?;
                    if drawing_bytes > self.limits.max_svg_nodes.saturating_mul(64) {
                        return Err(AppError::TooLarge);
                    }
                }
                if matches!(name, "order" | "numOctaves") {
                    for number in value
                        .split(|character: char| {
                            character.is_ascii_whitespace() || character == ','
                        })
                        .filter(|part| !part.is_empty())
                    {
                        let number: f64 = number.parse().map_err(invalid_image)?;
                        if !number.is_finite()
                            || number < 0.0
                            || number > if name == "order" { 15.0 } else { 8.0 }
                        {
                            return Err(AppError::TooLarge);
                        }
                    }
                }
                if name.starts_with("on") || name == "base" {
                    return Err(AppError::InvalidImage);
                }
                if name == "href" {
                    if let Some(reference) = value.strip_prefix('#') {
                        references.push((node.id(), reference.to_owned()));
                    } else if tag == "image" || tag == "feImage" {
                        let (mime, data) =
                            embedded_data(value, self.limits.max_svg_embedded_bytes)?;
                        embedded_bytes = embedded_bytes
                            .checked_add(data.len())
                            .ok_or(AppError::TooLarge)?;
                        if embedded_bytes > self.limits.max_svg_embedded_bytes {
                            return Err(AppError::TooLarge);
                        }
                        let info = self.inspect(&data, &mime)?;
                        if matches!(info.format, InputFormat::Svg) {
                            return Err(AppError::InvalidImage);
                        }
                        embedded_pixels = embedded_pixels
                            .checked_add(u64::from(info.width) * u64::from(info.height))
                            .ok_or(AppError::TooLarge)?;
                        if embedded_pixels > self.limits.max_svg_embedded_pixels {
                            return Err(AppError::TooLarge);
                        }
                        // Embedded decodes are serial; only their normalized PNGs
                        // accumulate. Reserve each pixel's retained and decode peak.
                        let memory = (u64::from(info.width) * u64::from(info.height))
                            .checked_mul(if matches!(info.format, InputFormat::Avif) {
                                80
                            } else {
                                40
                            })
                            .and_then(|n| n.checked_add(data.len() as u64 * 3))
                            .ok_or(AppError::TooLarge)?;
                        budget.embedded_memory = budget
                            .embedded_memory
                            .checked_add(usize::try_from(memory).map_err(|_| AppError::TooLarge)?)
                            .ok_or(AppError::TooLarge)?;
                    } else {
                        return Err(AppError::InvalidImage);
                    }
                } else if name == "style" || value.to_ascii_lowercase().contains("url") {
                    check_css(value, &mut references, node.id())?;
                }
            }
        }
        for (from, reference) in references {
            if let Some(to) = ids.get(reference.as_str()) {
                edges.entry(from).or_default().push(*to);
            }
        }
        // Count the expanded reference DAG with memoization. A small XML file can
        // otherwise cause exponential <use>, pattern, marker or filter expansion.
        let mut visiting = HashSet::new();
        let mut costs = HashMap::new();
        for id in edges.keys().copied() {
            expanded_cost(
                id,
                &edges,
                &mut visiting,
                &mut costs,
                0,
                self.limits.max_svg_depth,
                self.limits.max_svg_nodes,
            )?;
        }
        Ok(budget)
    }

    fn svg_options<'a>(
        &'a self,
        decode_resources: bool,
        failure: Arc<Mutex<Option<AppError>>>,
    ) -> usvg::Options<'a> {
        let mut options = usvg::Options {
            fontdb: Arc::clone(&self.fonts),
            font_family: "Noto Sans".to_owned(),
            resources_dir: None,
            ..usvg::Options::default()
        };
        let cache: Mutex<HashMap<Arc<Vec<u8>>, usvg::ImageKind>> = Mutex::new(HashMap::new());
        options.image_href_resolver = usvg::ImageHrefResolver {
            resolve_string: Box::new(|_, _| None),
            resolve_data: Box::new(move |mime, data, _| {
                if !decode_resources {
                    return None;
                }
                let result = (|| {
                    if let Some(image) = cache.lock().map_err(|_| AppError::Internal)?.get(&data) {
                        return Ok(image.clone());
                    }
                    let info = self.inspect(&data, mime)?;
                    let raster = match info.format {
                        InputFormat::Svg => return Err(AppError::InvalidImage),
                        InputFormat::Gif => self.decode_gif(&data)?,
                        InputFormat::Avif => self.decode_avif(&data)?,
                        format => self.decode_raster(&data, format)?,
                    };
                    let raster = raster.into_rgba()?;
                    // Normalize through our budgeted decoder. resvg only sees trusted
                    // RGBA PNG, including AVIF, and cannot invoke another resource loader.
                    let maximum = raster
                        .pixels
                        .len()
                        .checked_add(raster.height as usize)
                        .and_then(|n| n.checked_add(MIB))
                        .ok_or(AppError::TooLarge)?;
                    let png = encode_png(&raster.pixels, raster.width, raster.height, maximum)?;
                    let image = usvg::ImageKind::PNG(Arc::new(png));
                    cache
                        .lock()
                        .map_err(|_| AppError::Internal)?
                        .insert(Arc::clone(&data), image.clone());
                    Ok(image)
                })();
                match result {
                    Ok(image) => Some(image),
                    Err(error) => {
                        if let Ok(mut slot) = failure.lock() {
                            *slot = Some(error);
                        }
                        None
                    }
                }
            }),
        };
        options
    }

    fn decode_svg(&self, bytes: &[u8], reservation: usize) -> Result<DecodedImage> {
        let document = self.svg_document(bytes)?;
        let budget = self.check_svg(&document)?;
        let failure = Arc::new(Mutex::new(None));
        let options = self.svg_options(true, Arc::clone(&failure));
        let tree = usvg::Tree::from_xmltree(&document, &options).map_err(invalid_image)?;
        if let Some(error) = *failure.lock().map_err(|_| AppError::Internal)? {
            return Err(error);
        }
        let size = tree.size().to_int_size();
        let pixels = self.check_dimensions(size.width(), size.height())?;
        let scratch = self.svg_scratch(
            tree.root(),
            0,
            &mut 0usize,
            tiny_skia::Transform::identity(),
        )?;
        let required = scratch
            .checked_add(pixels.checked_mul(16).ok_or(AppError::TooLarge)?)
            .and_then(|n| n.checked_add(budget.embedded_memory))
            .and_then(|n| n.checked_add(bytes.len().checked_mul(32)?))
            .and_then(|n| n.checked_add(64 * MIB))
            .ok_or(AppError::TooLarge)?;
        if required > reservation {
            return Err(AppError::TooLarge);
        }
        let data = zeroed::<u8>(pixels.checked_mul(4).ok_or(AppError::TooLarge)?)?;
        let mut pixmap = tiny_skia::Pixmap::from_vec(data, size).ok_or(AppError::Internal)?;
        resvg::render(
            &tree,
            tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );
        let mut rgba = pixmap.take();
        // tiny-skia emits premultiplied bytes. Recover straight RGB exactly once;
        // input alpha is then deliberately ignored by the common RGB model path.
        for pixel in rgba.chunks_exact_mut(4) {
            unpremultiply(pixel);
        }
        Raster {
            width: size.width(),
            height: size.height(),
            pixels: rgba,
            channels: 4,
        }
        .into_rgb()
    }

    fn svg_scratch(
        &self,
        group: &usvg::Group,
        depth: usize,
        count: &mut usize,
        outer: tiny_skia::Transform,
    ) -> Result<usize> {
        if depth > self.limits.max_svg_depth {
            return Err(AppError::TooLarge);
        }
        let bounds = group
            .abs_layer_bounding_box()
            .transform(outer)
            .ok_or(AppError::InvalidImage)?;
        let mut scratch = self.svg_surface(bounds.width() + 4.0, bounds.height() + 4.0)?;
        for filter in group.filters() {
            scratch = scratch
                .checked_add(
                    self.svg_surface(bounds.width(), bounds.height())?
                        .checked_mul(
                            filter
                                .primitives()
                                .len()
                                .checked_mul(4)
                                .ok_or(AppError::TooLarge)?,
                        )
                        .ok_or(AppError::TooLarge)?,
                )
                .ok_or(AppError::TooLarge)?;
        }
        for node in group.children() {
            *count = count.checked_add(1).ok_or(AppError::TooLarge)?;
            if *count > self.limits.max_svg_nodes {
                return Err(AppError::TooLarge);
            }
            if let usvg::Node::Group(child) = node {
                scratch = scratch
                    .checked_add(self.svg_scratch(child, depth + 1, count, outer)?)
                    .ok_or(AppError::TooLarge)?;
            }
            let mut pattern_scale = outer;
            if let usvg::Node::Path(path) = node {
                let paints = [
                    path.fill().map(|fill| fill.paint()),
                    path.stroke().map(|stroke| stroke.paint()),
                ];
                let (mut max_x, mut max_y) = outer.get_scale();
                for paint in paints.into_iter().flatten() {
                    if let usvg::Paint::Pattern(pattern) = paint {
                        let transform = outer
                            .pre_concat(node.abs_transform())
                            .pre_concat(pattern.transform());
                        let (sx, sy) = transform.get_scale();
                        scratch = scratch
                            .checked_add(self.svg_surface(
                                pattern.rect().width() * sx,
                                pattern.rect().height() * sy,
                            )?)
                            .ok_or(AppError::TooLarge)?;
                        max_x = max_x.max(sx);
                        max_y = max_y.max(sy);
                    }
                }
                pattern_scale = tiny_skia::Transform::from_scale(max_x, max_y);
            }
            let mut failure = None;
            node.subroots(|child| {
                if failure.is_none() {
                    match self.svg_scratch(child, depth + 1, count, pattern_scale) {
                        Ok(bytes) => match scratch.checked_add(bytes) {
                            Some(total) => scratch = total,
                            None => failure = Some(AppError::TooLarge),
                        },
                        Err(error) => failure = Some(error),
                    }
                }
            });
            if let Some(error) = failure {
                return Err(error);
            }
        }
        Ok(scratch)
    }

    fn svg_surface(&self, width: f32, height: f32) -> Result<usize> {
        if !width.is_finite()
            || !height.is_finite()
            || width < 0.0
            || height < 0.0
            || width.ceil() > MAX_SIDE as f32
            || height.ceil() > MAX_SIDE as f32
        {
            return Err(AppError::TooLarge);
        }
        self.check_dimensions(width.ceil().max(1.0) as u32, height.ceil().max(1.0) as u32)?
            .checked_mul(4)
            .ok_or(AppError::TooLarge)
    }
}

#[derive(Default)]
struct SvgBudget {
    depth: usize,
    filter_primitives: usize,
    embedded_memory: usize,
}

struct Raster {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    channels: usize,
}

impl Raster {
    fn into_rgb(mut self) -> Result<DecodedImage> {
        let pixels = area(self.width, self.height)?;
        if self.pixels.len()
            != pixels
                .checked_mul(self.channels)
                .ok_or(AppError::Internal)?
        {
            return Err(AppError::Internal);
        }
        if self.channels == 4 {
            // Compact in place without destroying RGB underneath input alpha.
            for index in 0..pixels {
                self.pixels.copy_within(index * 4..index * 4 + 3, index * 3);
            }
            self.pixels.truncate(pixels * 3);
        } else if self.channels != 3 {
            return Err(AppError::Internal);
        }
        Ok(DecodedImage {
            width: self.width,
            height: self.height,
            pixels: self.pixels,
        })
    }

    fn into_rgba(mut self) -> Result<Self> {
        if self.channels == 4 {
            return Ok(self);
        }
        if self.channels != 3 {
            return Err(AppError::Internal);
        }
        let pixels = area(self.width, self.height)?;
        let length = pixels.checked_mul(4).ok_or(AppError::TooLarge)?;
        self.pixels
            .try_reserve_exact(length - self.pixels.len())
            .map_err(internal_image)?;
        self.pixels.resize(length, 255);
        for index in (0..pixels).rev() {
            self.pixels.copy_within(index * 3..index * 3 + 3, index * 4);
            self.pixels[index * 4 + 3] = 255;
        }
        self.channels = 4;
        Ok(self)
    }
}

fn decode_decoder(
    decoder: impl ImageDecoder,
    width: u32,
    height: u32,
    pillow_png: bool,
    premultiplied: bool,
) -> Result<Raster> {
    if decoder.dimensions() != (width, height) {
        return Err(AppError::InvalidImage);
    }
    let color = decoder.color_type();
    let bytes = usize::try_from(decoder.total_bytes()).map_err(|_| AppError::TooLarge)?;
    let pixels = area(width, height)?;
    if bytes > pixels.checked_mul(8).ok_or(AppError::TooLarge)? {
        return Err(AppError::TooLarge);
    }
    let mut source = zeroed::<u8>(bytes)?;
    decoder.read_image(&mut source).map_err(invalid_image)?;
    if color == ColorType::Rgb8 {
        return Ok(Raster {
            width,
            height,
            pixels: source,
            channels: 3,
        });
    }
    if color == ColorType::Rgba8 {
        if premultiplied {
            for pixel in source.chunks_exact_mut(4) {
                unpremultiply(pixel);
            }
        }
        return Ok(Raster {
            width,
            height,
            pixels: source,
            channels: 4,
        });
    }
    let output_channels = if color.has_alpha() { 4 } else { 3 };
    let mut output = zeroed::<u8>(
        pixels
            .checked_mul(output_channels)
            .ok_or(AppError::TooLarge)?,
    )?;
    for (index, target) in output.chunks_exact_mut(output_channels).enumerate() {
        if output_channels == 4 {
            target[3] = 255;
        }
        match color {
            ColorType::L8 => target[..3].fill(source[index]),
            ColorType::La8 => {
                target[..3].fill(source[index * 2]);
                target[3] = source[index * 2 + 1];
            }
            ColorType::L16 | ColorType::La16 | ColorType::Rgb16 | ColorType::Rgba16 => {
                let channels = usize::from(color.channel_count());
                let channel = |channel: usize| {
                    let offset = (index * channels + channel) * 2;
                    let mut value = u16::from_ne_bytes([source[offset], source[offset + 1]]);
                    if premultiplied && channels == 4 && channel < 3 {
                        let alpha_offset = (index * channels + 3) * 2;
                        let alpha =
                            u16::from_ne_bytes([source[alpha_offset], source[alpha_offset + 1]]);
                        value = if alpha == 0 {
                            0
                        } else {
                            ((u64::from(value) * 65_535 + u64::from(alpha) / 2) / u64::from(alpha))
                                .min(65_535) as u16
                        };
                    }
                    if pillow_png && color == ColorType::L16 {
                        value.min(255) as u8
                    } else {
                        (value >> 8) as u8
                    }
                };
                if channels <= 2 {
                    target[..3].fill(channel(0));
                } else {
                    for (index, output) in target[..3].iter_mut().enumerate() {
                        *output = channel(index);
                    }
                }
                if channels == 2 || channels == 4 {
                    target[3] = channel(channels - 1);
                }
            }
            _ => return Err(AppError::InvalidImage),
        }
    }
    Ok(Raster {
        width,
        height,
        pixels: output,
        channels: output_channels,
    })
}

fn gif_reader(bytes: &[u8], max_bytes: u64) -> Result<gif::Decoder<Cursor<&[u8]>>> {
    let mut options = gif::DecodeOptions::new();
    options.set_color_output(gif::ColorOutput::Indexed);
    options.set_memory_limit(gif::MemoryLimit::Bytes(
        NonZeroU64::new(max_bytes).ok_or(AppError::Internal)?,
    ));
    options.read_info(Cursor::new(bytes)).map_err(invalid_image)
}

fn palette_color(palette: &[u8], index: u8) -> Result<&[u8]> {
    let offset = usize::from(index) * 3;
    palette
        .get(offset..offset + 3)
        .ok_or(AppError::InvalidImage)
}

// mp4parse concatenates iloc extents, even when they repeatedly reference the
// same source bytes. Bound the *resolved* sum before calling it: file length by
// itself does not bound the allocations triggered by an extent table.
fn preflight_avif_container(bytes: &[u8]) -> Result<()> {
    const MAX_BOXES: usize = 16_384;
    const MAX_DATA_BOXES: usize = 64;
    let mut data_boxes = [(0usize, 0usize); MAX_DATA_BOXES];
    let mut data_count = 0usize;
    let mut box_count = 0usize;
    let mut offset = 0usize;
    let mut meta = None;
    while let Some(part) = next_bmff_box(bytes, &mut offset, 0)? {
        box_count += 1;
        if box_count > MAX_BOXES {
            return Err(AppError::TooLarge);
        }
        match &part.kind {
            b"meta" if meta.is_some() => return Err(AppError::InvalidImage),
            b"meta" => meta = Some(part),
            b"mdat" => {
                if data_count == MAX_DATA_BOXES {
                    return Err(AppError::TooLarge);
                }
                data_boxes[data_count] = (
                    part.payload_offset,
                    part.payload_offset + part.payload.len(),
                );
                data_count += 1;
            }
            _ => {}
        }
    }
    let meta = meta.ok_or(AppError::InvalidImage)?;
    if meta.payload.get(..4) != Some(&[0, 0, 0, 0]) {
        return Err(AppError::InvalidImage);
    }
    let mut offset = 4usize;
    let mut iloc = None;
    let mut idat = None;
    while let Some(part) = next_bmff_box(meta.payload, &mut offset, meta.payload_offset)? {
        box_count += 1;
        if box_count > MAX_BOXES {
            return Err(AppError::TooLarge);
        }
        match &part.kind {
            b"iloc" if iloc.is_some() => return Err(AppError::InvalidImage),
            b"iloc" => iloc = Some(part.payload),
            b"idat" => {
                let range = (
                    part.payload_offset,
                    part.payload_offset + part.payload.len(),
                );
                if idat.replace(range).is_some() {
                    return Err(AppError::InvalidImage);
                }
            }
            _ => {}
        }
    }
    let iloc = iloc.ok_or(AppError::InvalidImage)?;
    let version = *iloc.first().ok_or(AppError::InvalidImage)?;
    if version > 2 || iloc.get(1..4) != Some(&[0, 0, 0]) {
        return Err(AppError::InvalidImage);
    }
    let sizes = *iloc.get(4).ok_or(AppError::InvalidImage)?;
    let sizes2 = *iloc.get(5).ok_or(AppError::InvalidImage)?;
    let offset_size = sizes >> 4;
    let length_size = sizes & 15;
    let base_size = sizes2 >> 4;
    let index_size = if version == 0 { 0 } else { sizes2 & 15 };
    for size in [offset_size, length_size, base_size, index_size] {
        if !matches!(size, 0 | 4 | 8) {
            return Err(AppError::InvalidImage);
        }
    }
    let mut cursor = 6usize;
    let item_id_size = if version == 2 { 4 } else { 2 };
    let items = bmff_integer(iloc, &mut cursor, item_id_size)?;
    // In particular, reject a forged v2 item_count before mp4parse reserves its
    // HashMap from that count, independently of whether entries follow it.
    if items > 4096 {
        return Err(AppError::TooLarge);
    }
    let mut extent_count_total = 0u64;
    let mut resolved_total = 0u64;
    for _ in 0..items {
        bmff_integer(iloc, &mut cursor, item_id_size)?;
        let construction = if version == 0 {
            0
        } else {
            let value = bmff_integer(iloc, &mut cursor, 2)?;
            if value & !15 != 0 {
                return Err(AppError::InvalidImage);
            }
            value & 15
        };
        if construction > 1 || bmff_integer(iloc, &mut cursor, 2)? != 0 {
            return Err(AppError::InvalidImage);
        }
        let base = bmff_integer(iloc, &mut cursor, base_size)?;
        let extents = bmff_integer(iloc, &mut cursor, 2)?;
        if extents == 0 || (extents != 1 && (offset_size == 0 || length_size == 0)) {
            return Err(AppError::InvalidImage);
        }
        extent_count_total = extent_count_total
            .checked_add(extents)
            .ok_or(AppError::TooLarge)?;
        if extent_count_total > 16_384 {
            return Err(AppError::TooLarge);
        }
        for _ in 0..extents {
            bmff_integer(iloc, &mut cursor, index_size)?;
            let offset = base
                .checked_add(bmff_integer(iloc, &mut cursor, offset_size)?)
                .ok_or(AppError::TooLarge)?;
            let length = bmff_integer(iloc, &mut cursor, length_size)?;
            let offset = usize::try_from(offset).map_err(|_| AppError::TooLarge)?;
            let (start, end) = if construction == 1 {
                let (start, end) = idat.ok_or(AppError::InvalidImage)?;
                (start.checked_add(offset).ok_or(AppError::TooLarge)?, end)
            } else {
                let (_, end) = data_boxes[..data_count]
                    .iter()
                    .find(|(start, end)| offset >= *start && offset <= *end)
                    .ok_or(AppError::InvalidImage)?;
                (offset, *end)
            };
            let available = end.checked_sub(start).ok_or(AppError::InvalidImage)? as u64;
            // A zero extent length means through the end of its mdat/idat, not
            // an empty extent. Count each occurrence, including overlaps.
            let resolved = if length == 0 { available } else { length };
            if resolved > available {
                return Err(AppError::InvalidImage);
            }
            resolved_total = resolved_total
                .checked_add(resolved)
                .ok_or(AppError::TooLarge)?;
            if resolved_total > bytes.len() as u64 {
                return Err(AppError::TooLarge);
            }
        }
    }
    if cursor != iloc.len() {
        return Err(AppError::InvalidImage);
    }
    Ok(())
}

struct BmffBox<'a> {
    kind: [u8; 4],
    payload: &'a [u8],
    payload_offset: usize,
}

fn next_bmff_box<'a>(
    bytes: &'a [u8],
    offset: &mut usize,
    base: usize,
) -> Result<Option<BmffBox<'a>>> {
    if *offset == bytes.len() {
        return Ok(None);
    }
    let start = *offset;
    let short_size = bmff_integer(bytes, offset, 4)?;
    let kind = bytes
        .get(*offset..offset.checked_add(4).ok_or(AppError::TooLarge)?)
        .ok_or(AppError::InvalidImage)?;
    let kind = [kind[0], kind[1], kind[2], kind[3]];
    *offset += 4;
    let size = match short_size {
        0 => bytes.len() - start,
        1 => usize::try_from(bmff_integer(bytes, offset, 8)?).map_err(|_| AppError::TooLarge)?,
        size => usize::try_from(size).map_err(|_| AppError::TooLarge)?,
    };
    let end = start.checked_add(size).ok_or(AppError::TooLarge)?;
    if end < *offset || end > bytes.len() {
        return Err(AppError::InvalidImage);
    }
    let payload_offset = base.checked_add(*offset).ok_or(AppError::TooLarge)?;
    let payload = &bytes[*offset..end];
    *offset = end;
    Ok(Some(BmffBox {
        kind,
        payload,
        payload_offset,
    }))
}

fn bmff_integer(bytes: &[u8], cursor: &mut usize, size: u8) -> Result<u64> {
    let end = cursor
        .checked_add(usize::from(size))
        .ok_or(AppError::TooLarge)?;
    let bytes = bytes.get(*cursor..end).ok_or(AppError::InvalidImage)?;
    let mut value = 0u64;
    for byte in bytes {
        value = (value << 8) | u64::from(*byte);
    }
    *cursor = end;
    Ok(value)
}

fn avif_dimensions(context: &mp4parse::AvifContext) -> Result<(u32, u32)> {
    let coded = context
        .primary_item_coded_data()
        .ok_or(AppError::InvalidImage)?;
    let mut sequence = std::mem::MaybeUninit::<dav1d_sys::Dav1dSequenceHeader>::uninit();
    // SAFETY: the immutable coded slice is live; success initializes the complete
    // header. No pixel allocation is performed by dav1d_parse_sequence_header.
    let status = unsafe {
        dav1d_sys::dav1d_parse_sequence_header(sequence.as_mut_ptr(), coded.as_ptr(), coded.len())
    };
    if status < 0 {
        return Err(AppError::InvalidImage);
    }
    let sequence = unsafe { sequence.assume_init() };
    Ok((
        u32::try_from(sequence.max_width).map_err(|_| AppError::InvalidImage)?,
        u32::try_from(sequence.max_height).map_err(|_| AppError::InvalidImage)?,
    ))
}

fn leb128(bytes: &[u8], offset: &mut usize) -> Result<usize> {
    let mut value = 0u64;
    for shift in (0..56).step_by(7) {
        let byte = *bytes.get(*offset).ok_or(AppError::InvalidImage)?;
        *offset += 1;
        value |= u64::from(byte & 127) << shift;
        if byte & 128 == 0 {
            return usize::try_from(value).map_err(|_| AppError::TooLarge);
        }
    }
    Err(AppError::InvalidImage)
}

fn webp_lossy_dimensions(bytes: &[u8]) -> Result<(u32, u32)> {
    if bytes.get(3..6) != Some(b"\x9d\x01\x2a") {
        return Err(AppError::InvalidImage);
    }
    Ok((
        u32::from(le16(bytes, 6)? & 0x3fff),
        u32::from(le16(bytes, 8)? & 0x3fff),
    ))
}

fn webp_lossless_dimensions(bytes: &[u8]) -> Result<(u32, u32)> {
    if bytes.first() != Some(&0x2f) {
        return Err(AppError::InvalidImage);
    }
    let bits = le32(bytes, 1)?;
    Ok(((bits & 0x3fff) + 1, ((bits >> 14) & 0x3fff) + 1))
}

fn check_webp_frame(bytes: &[u8], width: u32, height: u32) -> Result<()> {
    let mut offset = 0usize;
    let mut seen = false;
    while offset < bytes.len() {
        let size = le32(bytes, offset + 4)? as usize;
        let end = offset
            .checked_add(8)
            .and_then(|n| n.checked_add(size))
            .ok_or(AppError::TooLarge)?;
        let payload = bytes.get(offset + 8..end).ok_or(AppError::InvalidImage)?;
        let dimensions = match &bytes[offset..offset + 4] {
            b"VP8 " => Some(webp_lossy_dimensions(payload)?),
            b"VP8L" => Some(webp_lossless_dimensions(payload)?),
            _ => None,
        };
        if let Some(dimensions) = dimensions {
            if seen || dimensions != (width, height) {
                return Err(AppError::InvalidImage);
            }
            seen = true;
        }
        offset = end.checked_add(size & 1).ok_or(AppError::TooLarge)?;
    }
    if !seen {
        return Err(AppError::InvalidImage);
    }
    Ok(())
}

fn exif_orientation(bytes: &[u8]) -> u8 {
    let bytes = bytes.strip_prefix(b"Exif\0\0").unwrap_or(bytes);
    let read16 = |offset| match bytes.get(..2) {
        Some(b"II") => le16(bytes, offset),
        Some(b"MM") => be16(bytes, offset),
        _ => Err(AppError::InvalidImage),
    };
    let read32 = |offset| match bytes.get(..2) {
        Some(b"II") => le32(bytes, offset),
        Some(b"MM") => be32(bytes, offset),
        _ => Err(AppError::InvalidImage),
    };
    let parsed = (|| {
        if read16(2)? != 42 {
            return Err(AppError::InvalidImage);
        }
        let ifd = read32(4)? as usize;
        let entries = read16(ifd)? as usize;
        for index in 0..entries {
            let offset = ifd
                .checked_add(2)
                .and_then(|n| n.checked_add(index.checked_mul(12)?))
                .ok_or(AppError::InvalidImage)?;
            if read16(offset)? == 0x112 && read16(offset + 2)? == 3 && read32(offset + 4)? == 1 {
                let orientation = read16(offset + 8)?;
                return Ok(if (1..=8).contains(&orientation) {
                    orientation as u8
                } else {
                    1
                });
            }
        }
        Ok(1)
    })();
    // withoutbg deliberately leaves malformed/missing EXIF unrotated.
    parsed.unwrap_or(1)
}

fn orient_rgb(image: DecodedImage, orientation: u8) -> Result<DecodedImage> {
    let (width, height, pixels) = orient(image.width, image.height, image.pixels, 3, orientation)?;
    Ok(DecodedImage {
        width,
        height,
        pixels,
    })
}

fn orient_raster(image: Raster, orientation: u8) -> Result<Raster> {
    let (width, height, pixels) = orient(
        image.width,
        image.height,
        image.pixels,
        image.channels,
        orientation,
    )?;
    Ok(Raster {
        width,
        height,
        pixels,
        channels: image.channels,
    })
}

fn orient(
    width: u32,
    height: u32,
    source: Vec<u8>,
    channels: usize,
    orientation: u8,
) -> Result<(u32, u32, Vec<u8>)> {
    if !(2..=8).contains(&orientation) {
        return Ok((width, height, source));
    }
    let (out_width, out_height) = if orientation >= 5 {
        (height, width)
    } else {
        (width, height)
    };
    let mut output = zeroed::<u8>(source.len())?;
    for y in 0..height {
        for x in 0..width {
            let (tx, ty) = match orientation {
                2 => (width - 1 - x, y),
                3 => (width - 1 - x, height - 1 - y),
                4 => (x, height - 1 - y),
                5 => (y, x),
                6 => (height - 1 - y, x),
                7 => (height - 1 - y, width - 1 - x),
                8 => (y, width - 1 - x),
                _ => unreachable!(),
            };
            let src = (y as usize * width as usize + x as usize) * channels;
            let dst = (ty as usize * out_width as usize + tx as usize) * channels;
            output[dst..dst + channels].copy_from_slice(&source[src..src + channels]);
        }
    }
    Ok((out_width, out_height, output))
}

fn unpremultiply(pixel: &mut [u8]) {
    let alpha = u32::from(pixel[3]);
    if alpha == 0 {
        pixel[..3].fill(0);
    } else if alpha != 255 {
        for channel in &mut pixel[..3] {
            *channel = ((u32::from(*channel) * 255 + alpha / 2) / alpha).min(255) as u8;
        }
    }
}

fn resize(
    source: &[u8],
    width: u32,
    height: u32,
    out_width: u32,
    out_height: u32,
    pixel_type: PixelType,
) -> Result<Vec<u8>> {
    let input = ImageRef::new(width, height, source, pixel_type).map_err(internal_image)?;
    let length = area(out_width, out_height)?
        .checked_mul(pixel_type.size())
        .ok_or(AppError::Internal)?;
    let mut output = zeroed::<u8>(length)?;
    let mut target = Image::from_slice_u8(out_width, out_height, &mut output, pixel_type)
        .map_err(internal_image)?;
    Resizer::new()
        .resize(
            &input,
            &mut target,
            &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(ResizeFilter::Bilinear)),
        )
        .map_err(internal_image)?;
    Ok(output)
}

fn embedded_data(value: &str, limit: usize) -> Result<(String, Vec<u8>)> {
    let url = data_url::DataUrl::process(value).map_err(invalid_image)?;
    let mime = format!("{}/{}", url.mime_type().type_, url.mime_type().subtype);
    if !matches!(
        mime.as_str(),
        "image/jpeg" | "image/png" | "image/gif" | "image/webp" | "image/avif"
    ) {
        return Err(AppError::InvalidImage);
    }
    let mut data = BoundedWriter::new(limit);
    let mut overflow = false;
    let result = url.decode(|bytes| {
        if data
            .bytes
            .len()
            .checked_add(bytes.len())
            .is_none_or(|length| length > limit)
        {
            overflow = true;
        }
        data.write_all(bytes)
    });
    if overflow {
        return Err(AppError::TooLarge);
    }
    result.map_err(invalid_image)?;
    Ok((mime, data.bytes))
}

fn check_css(
    value: &str,
    references: &mut Vec<(roxmltree::NodeId, String)>,
    node: roxmltree::NodeId,
) -> Result<()> {
    // CSS escape/comment obfuscation is not needed for the static subset; reject
    // it rather than accidentally admitting an escaped external URL or import.
    if value.contains('\\') || value.contains("/*") || value.contains('@') {
        return Err(AppError::InvalidImage);
    }
    let lower = value.to_ascii_lowercase();
    let mut remaining = lower.as_str();
    while let Some(start) = remaining.find("url(") {
        remaining = &remaining[start + 4..];
        let end = remaining.find(')').ok_or(AppError::InvalidImage)?;
        let target = remaining[..end].trim().trim_matches(['\'', '"']);
        if !target.starts_with('#') {
            return Err(AppError::InvalidImage);
        }
        remaining = &remaining[end + 1..];
    }
    // Preserve the case of IDs; CSS function names are case-insensitive.
    let mut offset = 0usize;
    while let Some(start) = lower[offset..].find("url(") {
        let start = offset + start + 4;
        let end = start + lower[start..].find(')').ok_or(AppError::InvalidImage)?;
        let target = value[start..end].trim().trim_matches(['\'', '"']);
        references.push((node, target[1..].to_owned()));
        offset = end + 1;
    }
    Ok(())
}

fn expanded_cost(
    id: roxmltree::NodeId,
    edges: &HashMap<roxmltree::NodeId, Vec<roxmltree::NodeId>>,
    visiting: &mut HashSet<roxmltree::NodeId>,
    costs: &mut HashMap<roxmltree::NodeId, (usize, usize)>,
    depth: usize,
    max_depth: usize,
    limit: usize,
) -> Result<(usize, usize)> {
    if depth > max_depth {
        return Err(AppError::TooLarge);
    }
    if let Some(&(cost, height)) = costs.get(&id) {
        if depth.checked_add(height).is_none_or(|n| n > max_depth) {
            return Err(AppError::TooLarge);
        }
        return Ok((cost, height));
    }
    if !visiting.insert(id) {
        return Err(AppError::InvalidImage);
    }
    let mut cost = 1usize;
    let mut height = 0usize;
    if let Some(children) = edges.get(&id) {
        for child in children {
            let (child_cost, child_height) =
                expanded_cost(*child, edges, visiting, costs, depth + 1, max_depth, limit)?;
            cost = cost.checked_add(child_cost).ok_or(AppError::TooLarge)?;
            height = height.max(child_height.checked_add(1).ok_or(AppError::TooLarge)?);
            if cost > limit || depth.checked_add(height).is_none_or(|n| n > max_depth) {
                return Err(AppError::TooLarge);
            }
        }
    }
    visiting.remove(&id);
    costs.insert(id, (cost, height));
    Ok((cost, height))
}

struct BoundedWriter {
    bytes: Vec<u8>,
    limit: usize,
}

impl BoundedWriter {
    fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            limit,
        }
    }
}

impl Write for BoundedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let length = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .filter(|n| *n <= self.limit)
            .ok_or_else(|| io::Error::other("PNG-Ausgabe überschreitet das Speicherlimit"))?;
        // Explicit growth never reserves a geometrically doubled buffer beyond
        // the configured cap. The scheduler retains the actual final capacity.
        if length > self.bytes.capacity() {
            let capacity = length
                .max(self.bytes.capacity().saturating_mul(2))
                .min(self.limit);
            self.bytes
                .try_reserve_exact(capacity - self.bytes.len())
                .map_err(io::Error::other)?;
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn encode_png(rgba: &[u8], width: u32, height: u32, limit: usize) -> Result<Vec<u8>> {
    let mut writer = BoundedWriter::new(limit);
    PngEncoder::new_with_quality(&mut writer, CompressionType::Level(3), FilterType::Adaptive)
        .write_image(rgba, width, height, image::ExtendedColorType::Rgba8)
        .map_err(internal_image)?;
    Ok(writer.bytes)
}

fn zeroed<T: Default + Clone>(length: usize) -> Result<Vec<T>> {
    length
        .checked_mul(std::mem::size_of::<T>())
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or(AppError::TooLarge)?;
    let mut values = Vec::new();
    values.try_reserve_exact(length).map_err(internal_image)?;
    values.resize(length, T::default());
    Ok(values)
}

fn area(width: u32, height: u32) -> Result<usize> {
    (width as usize)
        .checked_mul(height as usize)
        .ok_or(AppError::TooLarge)
}

fn be16(bytes: &[u8], offset: usize) -> Result<u16> {
    let value = bytes
        .get(offset..offset.checked_add(2).ok_or(AppError::InvalidImage)?)
        .ok_or(AppError::InvalidImage)?;
    Ok(u16::from_be_bytes([value[0], value[1]]))
}
fn le16(bytes: &[u8], offset: usize) -> Result<u16> {
    let value = bytes
        .get(offset..offset.checked_add(2).ok_or(AppError::InvalidImage)?)
        .ok_or(AppError::InvalidImage)?;
    Ok(u16::from_le_bytes([value[0], value[1]]))
}
fn be32(bytes: &[u8], offset: usize) -> Result<u32> {
    let value = bytes
        .get(offset..offset.checked_add(4).ok_or(AppError::InvalidImage)?)
        .ok_or(AppError::InvalidImage)?;
    Ok(u32::from_be_bytes([value[0], value[1], value[2], value[3]]))
}
fn le32(bytes: &[u8], offset: usize) -> Result<u32> {
    let value = bytes
        .get(offset..offset.checked_add(4).ok_or(AppError::InvalidImage)?)
        .ok_or(AppError::InvalidImage)?;
    Ok(u32::from_le_bytes([value[0], value[1], value[2], value[3]]))
}
fn le24(bytes: &[u8], offset: usize) -> Result<u32> {
    let value = bytes
        .get(offset..offset.checked_add(3).ok_or(AppError::InvalidImage)?)
        .ok_or(AppError::InvalidImage)?;
    Ok(u32::from(value[0]) | (u32::from(value[1]) << 8) | (u32::from(value[2]) << 16))
}

fn invalid_image(error: impl std::fmt::Display) -> AppError {
    // Raw codec/XML diagnostics may contain input fragments or SVG URLs.
    let _ = error;
    tracing::debug!("Ungültige Bild- oder Metadatenstruktur");
    AppError::InvalidImage
}
fn internal_image(error: impl std::fmt::Display) -> AppError {
    tracing::error!(%error, "Native Bildverarbeitung fehlgeschlagen");
    AppError::Internal
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    fn pipeline() -> ImagePipeline {
        ImagePipeline {
            limits: ImageLimits {
                max_file_bytes: 20 * MIB,
                max_multipart_bytes: 20 * MIB + 64 * 1024,
                max_pixels: 40_000_000,
                max_output_bytes: 32 * MIB,
                max_svg_nodes: 20_000,
                max_svg_depth: 64,
                max_svg_embedded_bytes: 20 * MIB,
                max_svg_embedded_pixels: 40_000_000,
            },
            // These raster/shape tests intentionally do not exercise font loading.
            fonts: Arc::new(usvg::fontdb::Database::new()),
        }
    }

    #[test]
    fn avif_native_sequence_header_and_alpha_plane_decode_safely() {
        let pipeline = pipeline();
        let input = include_bytes!("test-data/rgba.avif");
        let info = pipeline.inspect(input, "image/avif").unwrap();
        let image = pipeline.decode(input, info, Model::Fast).unwrap();
        assert_eq!((image.width, image.height), (16, 8));
        assert_eq!(image.pixels.len(), 16 * 8 * 3);
        for pixel in image.pixels.chunks_exact(3) {
            for (actual, expected) in pixel.iter().zip([72u8, 133, 206]) {
                assert!(actual.abs_diff(expected) <= 1);
            }
        }
    }

    #[test]
    fn transparent_input_rgb_survives_replacement_and_png_encoding() {
        let pipeline = pipeline();
        let input = encode_png(&[213, 71, 19, 0, 10, 20, 30, 128], 2, 1, MIB).unwrap();
        let info = pipeline.inspect(&input, "image/png").unwrap();
        let image = pipeline.decode(&input, info, Model::Fast).unwrap();
        assert_eq!(image.pixels, [213, 71, 19, 10, 20, 30]);
        let output = pipeline
            .encode(
                image,
                Mask {
                    width: 2,
                    height: 1,
                    values: vec![0.0, 1.0],
                },
                MaskGeometry {
                    crop_width: 2,
                    crop_height: 1,
                },
                &ModelSpec {
                    model: Model::Fast,
                    input_size: 2,
                    precision: Precision::Fp32,
                },
            )
            .unwrap();
        let decoded = image::load_from_memory(&output.bytes).unwrap().into_rgba8();
        assert_eq!(decoded.as_raw(), &[213, 71, 19, 0, 10, 20, 30, 255]);
    }

    #[test]
    fn alpha_is_quantized_before_interpolating() {
        let pipeline = pipeline();
        let output = pipeline
            .encode(
                DecodedImage {
                    width: 3,
                    height: 1,
                    pixels: vec![99; 9],
                },
                Mask {
                    width: 2,
                    height: 1,
                    values: vec![0.1 / 255.0, 1.1 / 255.0],
                },
                MaskGeometry {
                    crop_width: 2,
                    crop_height: 1,
                },
                &ModelSpec {
                    model: Model::Quality,
                    input_size: 2,
                    precision: Precision::Fp32,
                },
            )
            .unwrap();
        let decoded = image::load_from_memory(&output.bytes).unwrap().into_rgba8();
        let alpha: Vec<_> = decoded.pixels().map(|pixel| pixel[3]).collect();
        assert_eq!(alpha, [0, 1, 1]);
    }

    #[test]
    fn fast_letterbox_uses_ties_even_and_top_left_black_padding() {
        let prepared = pipeline()
            .prepare(
                DecodedImage {
                    width: 8,
                    height: 5,
                    pixels: vec![255; 8 * 5 * 3],
                },
                &ModelSpec {
                    model: Model::Fast,
                    input_size: 4,
                    precision: Precision::Fp32,
                },
            )
            .unwrap();
        assert_eq!(
            (prepared.geometry.crop_width, prepared.geometry.crop_height),
            (4, 2)
        );
        let InputTensor::F32(tensor) = prepared.tensor else {
            panic!("FP32 erwartet");
        };
        for plane in tensor.chunks_exact(16) {
            assert_eq!(&plane[..8], &[1.0; 8]);
            assert_eq!(&plane[8..], &[0.0; 8]);
        }
    }

    #[test]
    fn fast_exif_changes_geometry_but_quality_keeps_storage_order() {
        let pipeline = pipeline();
        let input = encode_png(
            &[
                1, 0, 0, 255, 2, 0, 0, 255, 3, 0, 0, 255, 4, 0, 0, 255, 5, 0, 0, 255, 6, 0, 0, 255,
            ],
            2,
            3,
            MIB,
        )
        .unwrap();
        let expected = [
            vec![1, 2, 3, 4, 5, 6],
            vec![2, 1, 4, 3, 6, 5],
            vec![6, 5, 4, 3, 2, 1],
            vec![5, 6, 3, 4, 1, 2],
            vec![1, 3, 5, 2, 4, 6],
            vec![5, 3, 1, 6, 4, 2],
            vec![6, 4, 2, 5, 3, 1],
            vec![2, 4, 6, 1, 3, 5],
        ];
        for orientation in 1..=8 {
            let mut info = pipeline.inspect(&input, "image/png").unwrap();
            info.orientation = orientation;
            let fast = pipeline.decode(&input, info, Model::Fast).unwrap();
            assert_eq!(
                fast.pixels
                    .chunks_exact(3)
                    .map(|pixel| pixel[0])
                    .collect::<Vec<_>>(),
                expected[usize::from(orientation - 1)]
            );
            assert_eq!(
                (fast.width, fast.height),
                if orientation >= 5 { (3, 2) } else { (2, 3) }
            );
        }
        let mut info = pipeline.inspect(&input, "image/png").unwrap();
        info.orientation = 6;
        let quality = pipeline.decode(&input, info, Model::Quality).unwrap();
        assert_eq!((quality.width, quality.height), (2, 3));
        assert_eq!(
            quality
                .pixels
                .chunks_exact(3)
                .map(|pixel| pixel[0])
                .collect::<Vec<_>>(),
            expected[0]
        );
    }

    #[test]
    fn gif_uses_first_frame_without_erasing_transparent_palette_rgb() {
        let mut input = Vec::new();
        {
            let mut encoder = gif::Encoder::new(&mut input, 2, 1, &[200, 30, 40, 4, 5, 6]).unwrap();
            let first = gif::Frame {
                width: 2,
                height: 1,
                transparent: Some(0),
                buffer: Cow::Borrowed(&[0, 1]),
                ..Default::default()
            };
            encoder.write_frame(&first).unwrap();
            let second = gif::Frame {
                width: 2,
                height: 1,
                buffer: Cow::Borrowed(&[1, 1]),
                ..Default::default()
            };
            encoder.write_frame(&second).unwrap();
        }
        let pipeline = pipeline();
        let info = pipeline.inspect(&input, "image/gif").unwrap();
        let output = pipeline.decode(&input, info, Model::Fast).unwrap();
        assert_eq!(output.pixels, [200, 30, 40, 4, 5, 6]);
    }

    #[test]
    fn oversized_png_is_rejected_from_header_without_raster_data() {
        let mut input = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR".to_vec();
        input.extend_from_slice(&10_000u32.to_be_bytes());
        input.extend_from_slice(&10_000u32.to_be_bytes());
        assert!(matches!(
            pipeline().inspect(&input, "image/png"),
            Err(AppError::TooLarge)
        ));
    }

    #[test]
    fn svg_demultiplies_once_and_forbids_active_external_recursive_resources() {
        let pipeline = pipeline();
        let input = br##"<svg xmlns="http://www.w3.org/2000/svg" width="2" height="1"><rect width="2" height="1" fill="#c86432" fill-opacity=".5"/></svg>"##;
        let info = pipeline.inspect(input, "image/svg+xml").unwrap();
        let image = pipeline.decode(input, info, Model::Fast).unwrap();
        for pixel in image.pixels.chunks_exact(3) {
            for (actual, expected) in pixel.iter().zip([200u8, 100, 50]) {
                assert!(actual.abs_diff(expected) <= 1);
            }
        }
        for body in [
            r#"<image href="file:///private/image.png"/>"#,
            r#"<script>alert(1)</script>"#,
            r#"<rect width="1" height="1" onclick="run()"/>"#,
            r#"<style>@import 'https://example.invalid/style.css';</style>"#,
            r##"<g id="a"><use href="#a"/></g>"##,
            r#"<image href="data:image/svg+xml,%3Csvg%2F%3E"/>"#,
        ] {
            let svg = format!(
                r#"<svg xmlns="http://www.w3.org/2000/svg" width="2" height="1">{body}</svg>"#
            );
            assert!(matches!(
                pipeline.inspect(svg.as_bytes(), "image/svg+xml"),
                Err(AppError::InvalidImage)
            ));
        }
        let entity = br#"<!DOCTYPE svg [<!ENTITY x SYSTEM "file:///secret">]><svg xmlns="http://www.w3.org/2000/svg" width="2" height="1">&x;</svg>"#;
        assert!(matches!(
            pipeline.inspect(entity, "image/svg+xml"),
            Err(AppError::InvalidImage)
        ));
    }

    #[test]
    fn embedded_svg_images_share_the_aggregate_pixel_budget() {
        let mut pipeline = pipeline();
        let png = encode_png(&[240, 30, 60, 255], 1, 1, MIB).unwrap();
        let encoded: String = png.iter().map(|byte| format!("%{byte:02X}")).collect();
        let svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="2" height="1"><image width="1" height="1" href="data:image/png,{encoded}"/><image x="1" width="1" height="1" href="data:image/png,{encoded}"/></svg>"#
        );
        let info = pipeline.inspect(svg.as_bytes(), "image/svg+xml").unwrap();
        let image = pipeline.decode(svg.as_bytes(), info, Model::Fast).unwrap();
        assert_eq!(image.pixels, [240, 30, 60, 240, 30, 60]);
        pipeline.limits.max_svg_embedded_pixels = 1;
        assert!(matches!(
            pipeline.inspect(svg.as_bytes(), "image/svg+xml"),
            Err(AppError::TooLarge)
        ));
    }

    #[test]
    fn png_output_overflow_never_returns_a_partial_success() {
        let mut pipeline = pipeline();
        pipeline.limits.max_output_bytes = 16;
        let result = pipeline.encode(
            DecodedImage {
                width: 1,
                height: 1,
                pixels: vec![1, 2, 3],
            },
            Mask {
                width: 1,
                height: 1,
                values: vec![1.0],
            },
            MaskGeometry {
                crop_width: 1,
                crop_height: 1,
            },
            &ModelSpec {
                model: Model::Fast,
                input_size: 1,
                precision: Precision::Fp32,
            },
        );
        assert!(matches!(result, Err(AppError::Internal)));
    }

    #[test]
    fn avif_repeated_extents_are_bounded_before_container_materialization() {
        fn part(kind: &[u8; 4], payload: &[u8], wide: bool) -> Vec<u8> {
            let mut output = Vec::new();
            if wide {
                output.extend_from_slice(&1u32.to_be_bytes());
                output.extend_from_slice(kind);
                output.extend_from_slice(&(payload.len() as u64 + 16).to_be_bytes());
            } else {
                output.extend_from_slice(&(payload.len() as u32 + 8).to_be_bytes());
                output.extend_from_slice(kind);
            }
            output.extend_from_slice(payload);
            output
        }
        fn container(repeat: u16, to_end: bool, idat: bool, wide: bool) -> Vec<u8> {
            let ftyp = part(b"ftyp", b"avif\0\0\0\0mif1avif", false);
            let mut iloc = vec![1, 0, 0, 0, 0x44, 0]; // v1, four-byte offsets/lengths
            iloc.extend_from_slice(&1u16.to_be_bytes()); // one item
            iloc.extend_from_slice(&1u16.to_be_bytes()); // item id
            iloc.extend_from_slice(&u16::from(idat).to_be_bytes());
            iloc.extend_from_slice(&0u16.to_be_bytes()); // local data reference
            iloc.extend_from_slice(&repeat.to_be_bytes());
            let header = if wide { 16 } else { 8 };
            let meta_size = header + 4 + 8 + iloc.len() + usize::from(repeat) * 8;
            let data_offset = if idat {
                0
            } else {
                ftyp.len() + meta_size + header
            };
            for _ in 0..repeat {
                iloc.extend_from_slice(&(data_offset as u32).to_be_bytes());
                iloc.extend_from_slice(&(if to_end { 0u32 } else { 4096u32 }).to_be_bytes());
            }
            let mut meta = vec![0, 0, 0, 0];
            meta.extend(part(b"iloc", &iloc, false));
            let mut output = ftyp;
            if idat {
                meta.extend(part(b"idat", &[7; 4096], wide));
            }
            output.extend(part(b"meta", &meta, wide));
            if !idat {
                output.extend(part(b"mdat", &[7; 4096], wide));
            }
            output
        }
        for (idat, wide, to_end) in [
            (false, false, false),
            (false, true, true),
            (true, false, true),
        ] {
            let ordinary = container(1, to_end, idat, wide);
            assert!(preflight_avif_container(&ordinary).is_ok());
            let repeated = container(1000, to_end, idat, wide);
            assert!(matches!(
                preflight_avif_container(&repeated),
                Err(AppError::TooLarge)
            ));
            // The public inspect path must reject at preflight, before the
            // deeper decoder would report this metadata-only fixture invalid.
            assert!(matches!(
                pipeline().inspect(&repeated, "image/avif"),
                Err(AppError::TooLarge)
            ));
        }
    }
}
