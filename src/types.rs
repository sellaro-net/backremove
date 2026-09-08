use half::f16;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Model {
    Fast,
    Quality,
}
impl Model {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Quality => "quality",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    Fp32,
    Fp16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum WorkClass {
    Foreground,
    Background,
}
impl WorkClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Foreground => "foreground",
            Self::Background => "background",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub model: Model,
    pub input_size: u32,
    pub precision: Precision,
}
pub enum InputTensor {
    F32(Vec<f32>),
    F16(Vec<f16>),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFormat {
    Jpeg,
    Png,
    Webp,
    Gif,
    Avif,
    Svg,
}
#[derive(Debug)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    pub format: InputFormat,
    pub orientation: u8,
    pub estimated_memory_bytes: usize,
}
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}
#[derive(Debug, Clone, Copy)]
pub struct MaskGeometry {
    pub crop_width: u32,
    pub crop_height: u32,
}
pub struct PreparedImage {
    pub image: DecodedImage,
    pub tensor: InputTensor,
    pub geometry: MaskGeometry,
}
pub struct Mask {
    pub width: u32,
    pub height: u32,
    pub values: Vec<f32>,
}
pub struct EncodedImage {
    pub bytes: Vec<u8>,
    pub postprocess_ms: f64,
    pub encode_ms: f64,
}
#[derive(Debug, Default, Clone, Serialize)]
pub struct Timings {
    pub admission_ms: f64,
    pub decode_ms: f64,
    pub queue_ms: f64,
    pub inference_ms: f64,
    pub encode_ms: f64,
}
