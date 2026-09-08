use std::{env, fmt::Write, fs, mem::{align_of, offset_of, size_of}, path::PathBuf};

// Layout constants are compiled from the same Rust definitions used by callers.
// No native functions are linked or invoked by this build-time module.
#[allow(dead_code)]
#[path = "src/lib.rs"]
mod ffi;

fn main() {
    assert_eq!(env::var("HOST").unwrap(), env::var("TARGET").unwrap(),
        "Die dav1d-ABI-Prüfung benötigt einen nativen Build für die Zielplattform.");
    let dependencies = system_deps::Config::new().probe().expect("Das gepinnte dav1d-Paket muss vor dem Rust-Build bereitstehen.");
    let library = dependencies.get_by_name("dav1d").expect("dav1d fehlt");
    assert!(!library.include_paths.is_empty(), "Der dav1d-Headerpfad fehlt für die ABI-Prüfung.");
    let mut source = String::from("#include <stddef.h>\n#include <dav1d/dav1d.h>\n#include <dav1d/headers.h>\n");
    macro_rules! check {
        ($rust:ident, $c:literal, [$($field:ident => $cfield:literal),* $(,)?]) => {{
            writeln!(source, "_Static_assert(sizeof({}) == {}, \"{}: ABI-Groesse\");", $c, size_of::<ffi::$rust>(), $c).unwrap();
            writeln!(source, "_Static_assert(_Alignof({}) == {}, \"{}: ABI-Ausrichtung\");", $c, align_of::<ffi::$rust>(), $c).unwrap();
            $(writeln!(source, "_Static_assert(offsetof({}, {}) == {}, \"{}.{}: ABI-Offset\");", $c, $cfield, offset_of!(ffi::$rust, $field), $c, $cfield).unwrap();)*
        }};
    }
    check!(Dav1dUserData, "Dav1dUserData", [data => "data", ref_ => "ref"]);
    check!(Dav1dDataProps, "Dav1dDataProps", [timestamp => "timestamp", duration => "duration", offset => "offset", size => "size", user_data => "user_data"]);
    check!(Dav1dWarpedMotionParams, "Dav1dWarpedMotionParams", [type_ => "type", matrix => "matrix"]);
    check!(Dav1dContentLightLevel, "Dav1dContentLightLevel", []);
    check!(Dav1dMasteringDisplay, "Dav1dMasteringDisplay", []);
    check!(Dav1dITUTT35, "Dav1dITUTT35", [payload_size => "payload_size", payload => "payload"]);
    check!(Dav1dSequenceHeader, "Dav1dSequenceHeader", [
        profile => "profile", max_width => "max_width", max_height => "max_height", layout => "layout",
        pri => "pri", trc => "trc", mtrx => "mtrx", chr => "chr", hbd => "hbd", color_range => "color_range",
        num_operating_points => "num_operating_points", operating_points => "operating_points",
        still_picture => "still_picture", reduced_still_picture_header => "reduced_still_picture_header",
        timing_info_present => "timing_info_present", num_units_in_tick => "num_units_in_tick", time_scale => "time_scale",
        equal_picture_interval => "equal_picture_interval", num_ticks_per_picture => "num_ticks_per_picture",
        decoder_model_info_present => "decoder_model_info_present", encoder_decoder_buffer_delay_length => "encoder_decoder_buffer_delay_length",
        num_units_in_decoding_tick => "num_units_in_decoding_tick", buffer_removal_delay_length => "buffer_removal_delay_length",
        frame_presentation_delay_length => "frame_presentation_delay_length", display_model_info_present => "display_model_info_present",
        width_n_bits => "width_n_bits", height_n_bits => "height_n_bits", frame_id_numbers_present => "frame_id_numbers_present",
        delta_frame_id_n_bits => "delta_frame_id_n_bits", frame_id_n_bits => "frame_id_n_bits", sb128 => "sb128",
        filter_intra => "filter_intra", intra_edge_filter => "intra_edge_filter", inter_intra => "inter_intra",
        masked_compound => "masked_compound", warped_motion => "warped_motion", dual_filter => "dual_filter",
        order_hint => "order_hint", jnt_comp => "jnt_comp", ref_frame_mvs => "ref_frame_mvs",
        screen_content_tools => "screen_content_tools", force_integer_mv => "force_integer_mv",
        order_hint_n_bits => "order_hint_n_bits", super_res => "super_res", cdef => "cdef", restoration => "restoration",
        ss_hor => "ss_hor", ss_ver => "ss_ver", monochrome => "monochrome", color_description_present => "color_description_present",
        separate_uv_delta_q => "separate_uv_delta_q", film_grain_present => "film_grain_present", operating_parameter_info => "operating_parameter_info"
    ]);
    check!(Dav1dSequenceHeaderOperatingPoint, "struct Dav1dSequenceHeaderOperatingPoint", [major_level => "major_level", minor_level => "minor_level", initial_display_delay => "initial_display_delay", idc => "idc", tier => "tier", decoder_model_param_present => "decoder_model_param_present", display_model_param_present => "display_model_param_present"]);
    check!(Dav1dSequenceHeaderOperatingParameterInfo, "struct Dav1dSequenceHeaderOperatingParameterInfo", []);
    check!(Dav1dSegmentationData, "Dav1dSegmentationData", []);
    check!(Dav1dSegmentationDataSet, "Dav1dSegmentationDataSet", []);
    check!(Dav1dLoopfilterModeRefDeltas, "Dav1dLoopfilterModeRefDeltas", []);
    check!(Dav1dFilmGrainData, "Dav1dFilmGrainData", []);
    check!(Dav1dFrameHeader, "Dav1dFrameHeader", [film_grain => "film_grain", frame_type => "frame_type", width => "width", height => "height"]);
    check!(Dav1dPictureParameters, "Dav1dPictureParameters", [w => "w", h => "h", layout => "layout", bpc => "bpc"]);
    check!(Dav1dPicture, "Dav1dPicture", [seq_hdr => "seq_hdr", frame_hdr => "frame_hdr", data => "data", stride => "stride", p => "p", m => "m", content_light => "content_light", mastering_display => "mastering_display", allocator_data => "allocator_data"]);
    check!(Dav1dPicAllocator, "Dav1dPicAllocator", [cookie => "cookie", alloc_picture_callback => "alloc_picture_callback", release_picture_callback => "release_picture_callback"]);
    check!(Dav1dData, "Dav1dData", [data => "data", sz => "sz", ref_ => "ref", m => "m"]);
    check!(Dav1dLogger, "Dav1dLogger", [cookie => "cookie", callback => "callback"]);
    check!(Dav1dSettings, "Dav1dSettings", [n_threads => "n_threads", max_frame_delay => "max_frame_delay", apply_grain => "apply_grain", operating_point => "operating_point", all_layers => "all_layers", frame_size_limit => "frame_size_limit", allocator => "allocator", logger => "logger", strict_std_compliance => "strict_std_compliance", output_invisible_frames => "output_invisible_frames", inloop_filters => "inloop_filters", decode_frame_type => "decode_frame_type", reserved => "reserved"]);
    source.push_str("int backremove_dav1d_abi_verified;\n");
    let path = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("dav1d-abi.c");
    fs::write(&path, source).unwrap();
    cc::Build::new().file(path).includes(&library.include_paths).std("c11").warnings_into_errors(true).cargo_metadata(false).compile("dav1d_abi");
    println!("cargo:rerun-if-changed=src/lib.rs");
    for directory in &library.include_paths {
        println!("cargo:rerun-if-changed={}", directory.join("dav1d").display());
    }
}
