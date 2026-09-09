use crate::{
    artifacts::{ArtifactSet, ModelManifest, Provider, RUNTIME_API, RUNTIME_VERSION},
    config::Device,
    error::{AppError, Result},
    types::{InputTensor, Mask, Model, ModelSpec, Precision},
};
use half::f16;
use ort::{
    ep::{ArenaExtendStrategy, CPU, CUDA, cuda::ConvAlgorithmSearch},
    session::{Session, builder::GraphOptimizationLevel},
    value::{Tensor, TensorElementType, ValueType},
};
use serde_json::{Value, json};
use std::{
    ffi::{CStr, c_char, c_void},
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

/// One owner moves this engine to the dedicated inference worker. Sessions are
/// never cloned, replaced on a request timeout, or called concurrently.
pub struct InferenceEngine {
    fast: Resident,
    quality: Option<Resident>,
    quality_enabled: bool,
    provider: Provider,
}

struct Resident {
    spec: ModelSpec,
    session: Session,
    profile_enabled: bool,
    placement_audited: bool,
}

impl InferenceEngine {
    pub fn load(artifacts: ArtifactSet, device: Device, quality_enabled: bool) -> Result<Self> {
        let provider = artifacts.manifest.runtime.provider;
        provider.check_device(device)?;
        if quality_enabled != artifacts.quality_enabled {
            tracing::error!("Modellauswahl wurde nach der Artefaktprüfung geändert");
            return Err(AppError::Unavailable);
        }
        let library = artifacts.resolve(&artifacts.manifest.runtime.library)?;
        prepare_native_loader(&artifacts, &library)?;
        check_runtime_api(&library)?;
        let initialized = ort::init_from(&library).map_err(|error| {
            tracing::error!(%error, "Die geprüfte ONNX Runtime kann nicht initialisiert werden");
            AppError::Unavailable
        })?.with_name("backremove").with_telemetry(false).commit();
        if !initialized {
            tracing::error!(
                "ORT wurde bereits initialisiert; private Runtime-Auswahl ist nicht gewährleistet"
            );
            return Err(AppError::Unavailable);
        }
        tracing::info!(
            runtime_version = RUNTIME_VERSION,
            api = RUNTIME_API,
            provider = provider.execution_provider(),
            build = ort::info(),
            "Native ONNX Runtime initialisiert"
        );
        let profile_directory = profile_directory()?;
        let fast = Resident::load(
            &artifacts,
            &artifacts.manifest.fast,
            Model::Fast,
            profile_directory.as_deref(),
        )?;
        let quality = if quality_enabled {
            artifacts
                .manifest
                .quality
                .as_ref()
                .map(|manifest| {
                    Resident::load(
                        &artifacts,
                        manifest,
                        Model::Quality,
                        profile_directory.as_deref(),
                    )
                })
                .transpose()?
        } else {
            None
        };
        // Both selected sessions exist before either warm-up. This exercises
        // their combined resident allocation, rather than two isolated loads.
        let mut engine = Self {
            fast,
            quality,
            quality_enabled,
            provider,
        };
        for resident in std::iter::once(&mut engine.fast).chain(engine.quality.iter_mut()) {
            let start = Instant::now();
            resident.warm_up().map_err(|error| {
                tracing::error!(model = resident.spec.model.as_str(), %error, "Modell-Warm-up fehlgeschlagen");
                AppError::Unavailable
            })?;
            resident.finish_profile(provider)?;
            tracing::info!(
                model = resident.spec.model.as_str(),
                warmup_ms = start.elapsed().as_secs_f64() * 1000.0,
                placement_audited = resident.placement_audited,
                "Residentes Modell ist aufgewärmt"
            );
        }
        verify_loaded_libraries(&artifacts, true)?;
        Ok(engine)
    }

    pub fn spec(&self, model: Model) -> Option<ModelSpec> {
        self.resident(model).map(|resident| resident.spec.clone())
    }

    pub fn provider(&self) -> &str {
        self.provider.execution_provider()
    }

    pub fn status(&self) -> Value {
        let unavailable = if self.quality_enabled {
            "unavailable"
        } else {
            "disabled"
        };
        json!({
            "fast": self.fast.status(self.provider),
            "quality": self.quality.as_ref().map(|resident| resident.status(self.provider)).unwrap_or_else(|| json!({
                "enabled": self.quality_enabled,
                "loaded": false,
                "ready": false,
                "status": unavailable,
                "runtime": "onnxruntime",
                "runtime_version": RUNTIME_VERSION,
                "provider": null,
                "placement_audited": false
            }))
        })
    }

    pub fn infer(&mut self, model: Model, tensor: InputTensor) -> Result<Mask> {
        match model {
            Model::Fast => self.fast.infer(tensor),
            Model::Quality => self
                .quality
                .as_mut()
                .ok_or(AppError::Unavailable)?
                .infer(tensor),
        }
    }

    fn resident(&self, model: Model) -> Option<&Resident> {
        match model {
            Model::Fast => Some(&self.fast),
            Model::Quality => self.quality.as_ref(),
        }
    }
}

impl Resident {
    fn load(
        artifacts: &ArtifactSet,
        manifest: &ModelManifest,
        model: Model,
        profile: Option<&Path>,
    ) -> Result<Self> {
        let provider = artifacts.manifest.runtime.provider;
        let model_path = artifacts.resolve(&manifest.path)?;
        let create = || -> ort::Result<Session> {
            let mut builder = Session::builder()?
                .with_intra_threads(2)?
                .with_inter_threads(2)?
                .with_parallel_execution(false)?
                .with_memory_pattern(false)?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_log_id(format!("backremove-{}", model.as_str()))?;
            builder = match provider {
                Provider::Cuda => builder
                    .with_execution_providers([CUDA::default()
                        .with_device_id(0)
                        .with_conv_algorithm_search(ConvAlgorithmSearch::Heuristic)
                        .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested)
                        .with_conv_max_workspace(false)
                        .with_tf32(false)
                        .build()
                        .error_on_failure()])?
                    .with_disable_cpu_fallback()?,
                Provider::Cpu => builder.with_execution_providers([CPU::default()
                    .with_arena_allocator(true)
                    .build()
                    .error_on_failure()])?,
            };
            if let Some(directory) = profile {
                builder =
                    builder.with_profiling(directory.join(format!("{}-warmup", model.as_str())))?;
            }
            // The path was hash-verified before native loading. No remote model
            // lookup, graph rewrite, or runtime optimization artifact is emitted.
            builder.commit_from_file(&model_path)
        };
        let session = create().map_err(|error| {
            tracing::error!(%error, model = model.as_str(), provider = provider.execution_provider(),
                "Native Modellsitzung konnte nicht erstellt werden; kein Provider-Fallback");
            AppError::Unavailable
        })?;
        let resident = Self {
            spec: manifest.spec(model),
            session,
            profile_enabled: profile.is_some(),
            placement_audited: false,
        };
        if resident.session.inputs().len() != 1 || resident.session.outputs().len() != 1 {
            tracing::error!(
                model = model.as_str(),
                "Modell muss genau einen Eingang und einen Maskenausgang haben"
            );
            return Err(AppError::Unavailable);
        }
        let input = &resident.session.inputs()[0];
        let output = &resident.session.outputs()[0];
        if input.name() != manifest.input_name || output.name() != manifest.output_name {
            tracing::error!(
                model = model.as_str(),
                "Tensor-Namen stimmen nicht mit dem Manifest überein"
            );
            return Err(AppError::Unavailable);
        }
        let side = i64::from(manifest.input_size);
        let dtype = match manifest.precision {
            Precision::Fp32 => TensorElementType::Float32,
            Precision::Fp16 => TensorElementType::Float16,
        };
        check_tensor_type(input.dtype(), dtype, &[1, 3, side, side], model, "input")?;
        check_tensor_type(output.dtype(), dtype, &[1, 1, side, side], model, "output")?;
        Ok(resident)
    }

    fn warm_up(&mut self) -> Result<()> {
        let count = 3 * self.spec.input_size as usize * self.spec.input_size as usize;
        let tensor = match self.spec.precision {
            Precision::Fp32 => InputTensor::F32(vec![0.0; count]),
            Precision::Fp16 => InputTensor::F16(vec![f16::ZERO; count]),
        };
        self.infer(tensor).map(drop)
    }

    fn infer(&mut self, input: InputTensor) -> Result<Mask> {
        let model = self.spec.model.as_str();
        let side = self.spec.input_size as usize;
        let count = 3 * side * side;
        let outputs = match (input, self.spec.precision) {
            (InputTensor::F32(values), Precision::Fp32) => {
                if values.len() != count || values.iter().any(|value| !value.is_finite()) {
                    tracing::error!(model, "FP32-Eingabetensor hat ungültige Länge oder nichtendliche Werte");
                    return Err(AppError::Internal);
                }
                let tensor = Tensor::from_array(([1_usize, 3, side, side], values)).map_err(|error| {
                    tracing::error!(%error, model, "FP32-Eingabetensor kann nicht erstellt werden");
                    AppError::Internal
                })?;
                self.session.run(ort::inputs![tensor])
            }
            (InputTensor::F16(values), Precision::Fp16) => {
                if values.len() != count || values.iter().any(|value| !value.is_finite()) {
                    tracing::error!(model, "FP16-Eingabetensor hat ungültige Länge oder nichtendliche Werte");
                    return Err(AppError::Internal);
                }
                let tensor = Tensor::from_array(([1_usize, 3, side, side], values)).map_err(|error| {
                    tracing::error!(%error, model, "FP16-Eingabetensor kann nicht erstellt werden");
                    AppError::Internal
                })?;
                self.session.run(ort::inputs![tensor])
            }
            _ => {
                tracing::error!(model, "Eingabepräzision stimmt nicht mit der Modellsitzung überein");
                return Err(AppError::Internal);
            }
        }.map_err(|error| {
            tracing::error!(%error, model, "Synchroner ONNX-Modellaufruf fehlgeschlagen");
            AppError::Internal
        })?;
        // Session::run returns host output only after native work completes. No
        // timeout starts a replacement call; the scheduler retains ownership.
        if outputs.len() != 1 {
            tracing::error!(
                model,
                "ONNX-Aufruf lieferte eine unerwartete Anzahl Ausgänge"
            );
            return Err(AppError::Internal);
        }
        let output = &outputs[0];
        let values = match self.spec.precision {
            Precision::Fp32 => {
                let (shape, data) = output.try_extract_tensor::<f32>().map_err(|error| {
                    tracing::error!(%error, model, "FP32-Maske kann nicht gelesen werden");
                    AppError::Internal
                })?;
                check_output_shape(shape, data.len(), side, model)?;
                data.to_vec()
            }
            Precision::Fp16 => {
                let (shape, data) = output.try_extract_tensor::<f16>().map_err(|error| {
                    tracing::error!(%error, model, "FP16-Maske kann nicht gelesen werden");
                    AppError::Internal
                })?;
                check_output_shape(shape, data.len(), side, model)?;
                data.iter().map(|value| value.to_f32()).collect::<Vec<_>>()
            }
        };
        if values.iter().any(|value| !is_valid_probability(*value)) {
            // Aggregate only rejected output; never log individual mask pixels or
            // input content. Exact endpoint bits distinguish rounding from failure.
            let mut finite_min = None::<f32>;
            let mut finite_max = None::<f32>;
            let mut nonfinite_count = 0_usize;
            let mut below_zero_count = 0_usize;
            let mut above_one_count = 0_usize;
            for &value in &values {
                if !value.is_finite() {
                    nonfinite_count += 1;
                    continue;
                }
                finite_min = Some(finite_min.map_or(value, |minimum| minimum.min(value)));
                finite_max = Some(finite_max.map_or(value, |maximum| maximum.max(value)));
                below_zero_count += usize::from(value < 0.0);
                above_one_count += usize::from(value > 1.0);
            }
            tracing::error!(
                model,
                ?finite_min,
                ?finite_max,
                finite_min_bits = ?finite_min.map(f32::to_bits),
                finite_max_bits = ?finite_max.map(f32::to_bits),
                nonfinite_count,
                below_zero_count,
                above_one_count,
                "Modell lieferte keine endlichen Wahrscheinlichkeiten im Bereich 0 bis 1"
            );
            return Err(AppError::Internal);
        }
        // Both released graphs already include their semantic sigmoid. Only
        // binary16 -> binary32 conversion is applied: no sigmoid/clamp/min-max.
        Ok(Mask {
            width: self.spec.input_size,
            height: self.spec.input_size,
            values,
        })
    }

    fn status(&self, provider: Provider) -> Value {
        json!({ "enabled": true, "loaded": true, "ready": true, "status": "ready",
            "runtime": "onnxruntime", "runtime_version": RUNTIME_VERSION,
            "provider": provider.execution_provider(), "input_size": self.spec.input_size,
            "precision": match self.spec.precision { Precision::Fp32 => "fp32", Precision::Fp16 => "fp16" },
            "cpu_fallback_disabled": provider == Provider::Cuda,
            "placement_audited": self.placement_audited })
    }

    fn finish_profile(&mut self, provider: Provider) -> Result<()> {
        if !self.profile_enabled {
            return Ok(());
        }
        let path = self.session.end_profiling().map_err(|error| {
            tracing::error!(%error, model = self.spec.model.as_str(), "ORT-Profil kann nicht abgeschlossen werden");
            AppError::Unavailable
        })?;
        self.profile_enabled = false;
        audit_profile(Path::new(&path), provider, self.spec.model)?;
        self.placement_audited = true;
        Ok(())
    }
}

impl Drop for Resident {
    fn drop(&mut self) {
        // Preserve diagnostic traces even if a later model load/warm-up fails.
        if self.profile_enabled {
            match self.session.end_profiling() {
                Ok(path) => tracing::info!(
                    profile = path,
                    model = self.spec.model.as_str(),
                    "Unvollständiges ORT-Diagnoseprofil gespeichert"
                ),
                Err(error) => {
                    tracing::warn!(%error, "ORT-Diagnoseprofil konnte nicht abgeschlossen werden")
                }
            }
        }
    }
}

fn is_valid_probability(value: f32) -> bool {
    // ORT 1.26's MLAS FMA3 sigmoid can exceed one by one FP32 ULP
    // (observed 0x3f800001). PNG conversion saturates that endpoint like
    // the reference model; larger excursions and nonfinite values are errors.
    value.is_finite() && (0.0..=1.0_f32.next_up()).contains(&value)
}

fn check_tensor_type(
    dtype: &ValueType,
    expected: TensorElementType,
    shape: &[i64],
    model: Model,
    tensor: &str,
) -> Result<()> {
    if dtype.tensor_type() != Some(expected)
        || dtype
            .tensor_shape()
            .is_none_or(|declared| &declared[..] != shape)
    {
        tracing::error!(model = model.as_str(), tensor, declared_type = %dtype,
            "Modell hat nicht den deklarierten statischen Tensor-Typ und die exakte Form");
        return Err(AppError::Unavailable);
    }
    Ok(())
}

fn check_output_shape(
    shape: &ort::value::Shape,
    length: usize,
    side: usize,
    model: &str,
) -> Result<()> {
    if &shape[..] != [1_i64, 1, side as i64, side as i64].as_slice() || length != side * side {
        tracing::error!(model, "Berechnete Maske hat eine unerwartete Form");
        return Err(AppError::Internal);
    }
    Ok(())
}

fn profile_directory() -> Result<Option<PathBuf>> {
    let Some(path) = std::env::var_os("BACKREMOVE_ORT_PROFILE_DIR") else {
        return Ok(None);
    };
    if path.is_empty() {
        tracing::error!("BACKREMOVE_ORT_PROFILE_DIR darf nicht leer sein");
        return Err(AppError::Unavailable);
    }
    let path = PathBuf::from(path);
    fs::create_dir_all(&path).map_err(|error| {
        tracing::error!(%error, "ORT-Profilverzeichnis kann nicht erstellt werden");
        AppError::Unavailable
    })?;
    Ok(Some(path))
}

fn audit_profile(path: &Path, provider: Provider, model: Model) -> Result<()> {
    let file = fs::File::open(path).map_err(|error| {
        tracing::error!(%error, "ORT-Profil kann nicht geöffnet werden");
        AppError::Unavailable
    })?;
    let profile: Value =
        serde_json::from_reader(std::io::BufReader::new(file)).map_err(|error| {
            tracing::error!(%error, "ORT-Profil ist ungültig");
            AppError::Unavailable
        })?;
    let events = profile
        .as_array()
        .or_else(|| profile.get("traceEvents").and_then(Value::as_array))
        .ok_or_else(|| {
            tracing::error!("ORT-Profil enthält keine Ereignisliste");
            AppError::Unavailable
        })?;
    let mut matched = 0_usize;
    let mut deform_conv = 0_usize;
    for event in events {
        if event["cat"] != "Node" {
            continue;
        }
        let Some(actual) = event["args"]["provider"]
            .as_str()
            .filter(|name| !name.is_empty())
        else {
            continue;
        };
        if actual != provider.execution_provider() {
            tracing::error!(
                model = model.as_str(),
                actual_provider = actual,
                "ORT-Profil zeigt nicht erlaubte Graph-Ausführung auf anderem Provider"
            );
            return Err(AppError::Unavailable);
        }
        matched += 1;
        if event["args"]["op_name"] == "DeformConv" {
            deform_conv += 1;
        }
    }
    if matched == 0 {
        tracing::error!(
            model = model.as_str(),
            "ORT-Profil belegt keine Modellberechnung"
        );
        return Err(AppError::Unavailable);
    }
    tracing::info!(model = model.as_str(), provider = provider.execution_provider(),
        node_events = matched, deform_conv_events = deform_conv, profile = %path.display(),
        "Provider-Platzierung des vollständigen Warm-ups geprüft");
    Ok(())
}

// Stable public two-function OrtApiBase ABI, checked before the Rust binding can
// dereference its API26 table. No dependency on a separately linked ort-sys DLL.
#[repr(C)]
struct ApiBase {
    get_api: unsafe extern "system" fn(u32) -> *const c_void,
    get_version_string: unsafe extern "system" fn() -> *const c_char,
}

fn check_runtime_api(path: &Path) -> Result<()> {
    let library = unsafe { libloading::Library::new(path) }.map_err(|error| {
        tracing::error!(%error, "Native Runtime-Bibliothek kann nicht geöffnet werden");
        AppError::Unavailable
    })?;
    let get_base =
        unsafe { library.get::<unsafe extern "system" fn() -> *const ApiBase>(b"OrtGetApiBase\0") }
            .map_err(|error| {
                tracing::error!(%error, "OrtGetApiBase fehlt in der Runtime");
                AppError::Unavailable
            })?;
    let base = unsafe { get_base().as_ref() }.ok_or_else(|| {
        tracing::error!("OrtGetApiBase lieferte einen Nullzeiger");
        AppError::Unavailable
    })?;
    let version = unsafe { (base.get_version_string)() };
    if version.is_null() {
        tracing::error!("ORT-Runtime meldet keine Version");
        return Err(AppError::Unavailable);
    }
    let version = unsafe { CStr::from_ptr(version) }.to_string_lossy();
    if version != RUNTIME_VERSION || unsafe { (base.get_api)(RUNTIME_API) }.is_null() {
        tracing::error!(actual_version = %version, expected_version = RUNTIME_VERSION, api = RUNTIME_API,
            "Runtime-Version oder API stimmt nicht mit dem geprüften Paket überein");
        return Err(AppError::Unavailable);
    }
    // ORT holds a process-global environment. Runtime/dependency references must
    // remain valid through global teardown, not merely until this engine drops.
    std::mem::forget(library);
    Ok(())
}

#[cfg(windows)]
fn wide(path: &Path) -> Result<Vec<u16>> {
    use std::os::windows::ffi::OsStrExt;
    let mut value: Vec<u16> = path.as_os_str().encode_wide().collect();
    if value.contains(&0) {
        tracing::error!("NUL-Zeichen im nativen Bibliothekspfad");
        return Err(AppError::Unavailable);
    }
    value.push(0);
    Ok(value)
}

#[cfg(windows)]
fn check_loaded_path(
    handle: windows_sys::Win32::Foundation::HMODULE,
    requested: &Path,
) -> Result<()> {
    use std::os::windows::ffi::OsStringExt;
    use windows_sys::Win32::System::LibraryLoader::GetModuleFileNameW;
    let mut path = vec![0_u16; 32768];
    let length =
        unsafe { GetModuleFileNameW(handle, path.as_mut_ptr(), path.len() as u32) } as usize;
    if length == 0 || length >= path.len() {
        tracing::error!(error = %std::io::Error::last_os_error(), "Geladener DLL-Pfad kann nicht geprüft werden");
        return Err(AppError::Unavailable);
    }
    let actual = PathBuf::from(std::ffi::OsString::from_wide(&path[..length]));
    let actual = fs::canonicalize(actual).map_err(|error| {
        tracing::error!(%error, "Geladener DLL-Pfad ist nicht auflösbar");
        AppError::Unavailable
    })?;
    if actual.as_os_str().to_string_lossy().to_lowercase()
        != requested.as_os_str().to_string_lossy().to_lowercase()
    {
        tracing::error!(expected = %requested.display(), actual = %actual.display(), "Fremde gleichnamige DLL wurde bereits geladen");
        return Err(AppError::Unavailable);
    }
    Ok(())
}

#[cfg(windows)]
fn verify_loaded_libraries(artifacts: &ArtifactSet, require_provider: bool) -> Result<()> {
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    for entry in &artifacts.manifest.runtime.files {
        let path = artifacts.resolve(&entry.path)?;
        if !path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("dll"))
        {
            continue;
        }
        let name = path.file_name().ok_or(AppError::Unavailable)?;
        let encoded = wide(Path::new(name))?;
        let handle = unsafe { GetModuleHandleW(encoded.as_ptr()) };
        if handle.is_null() {
            if require_provider
                && artifacts.manifest.runtime.provider == Provider::Cuda
                && name
                    .to_str()
                    .is_some_and(|name| name.eq_ignore_ascii_case("onnxruntime_providers_cuda.dll"))
            {
                tracing::error!("CUDA-Sitzungen haben keine private CUDA-Provider-DLL geladen");
                return Err(AppError::Unavailable);
            }
            // Some verified dependencies are loaded lazily by vendor libraries.
            continue;
        }
        check_loaded_path(handle, &path)?;
    }
    Ok(())
}

#[cfg(windows)]
fn prepare_native_loader(artifacts: &ArtifactSet, library: &Path) -> Result<()> {
    use std::{collections::HashSet, ptr};
    use windows_sys::Win32::System::LibraryLoader::{
        AddDllDirectory, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32,
        LOAD_LIBRARY_SEARCH_USER_DIRS, LoadLibraryExW, SetDefaultDllDirectories,
    };
    // Exclude global PATH and CWD. System32 supplies the NVIDIA driver/MSVC
    // system runtime; every model-runtime dependency comes from this pack.
    let search = LOAD_LIBRARY_SEARCH_SYSTEM32 | LOAD_LIBRARY_SEARCH_USER_DIRS;
    if unsafe { SetDefaultDllDirectories(search) } == 0 {
        tracing::error!(error = %std::io::Error::last_os_error(), "Private DLL-Suchrichtlinie kann nicht gesetzt werden");
        return Err(AppError::Unavailable);
    }
    verify_loaded_libraries(artifacts, false)?;
    let mut directories = HashSet::new();
    for entry in &artifacts.manifest.runtime.files {
        let path = artifacts.resolve(&entry.path)?;
        let directory = path.parent().ok_or(AppError::Unavailable)?;
        if directories.insert(directory.to_path_buf()) {
            let encoded = wide(directory)?;
            if unsafe { AddDllDirectory(encoded.as_ptr()) }.is_null() {
                tracing::error!(error = %std::io::Error::last_os_error(), "Privates DLL-Verzeichnis kann nicht registriert werden");
                return Err(AppError::Unavailable);
            }
        }
    }
    // Directory cookies and library references intentionally survive until
    // process exit, matching the lifetime of ort's process-global runtime.
    for entry in &artifacts.manifest.runtime.preload {
        let path = artifacts.resolve(entry)?;
        load_private_library(&path, search)?;
    }
    load_private_library(library, search)?;

    fn load_private_library(path: &Path, search: u32) -> Result<()> {
        let encoded = wide(path)?;
        let handle = unsafe {
            LoadLibraryExW(
                encoded.as_ptr(),
                ptr::null_mut(),
                search | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
            )
        };
        if handle.is_null() {
            tracing::error!(error = %std::io::Error::last_os_error(), library = %path.display(), "Private native DLL kann nicht geladen werden");
            return Err(AppError::Unavailable);
        }
        check_loaded_path(handle, path)
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn prepare_native_loader(artifacts: &ArtifactSet, library: &Path) -> Result<()> {
    use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};
    verify_loaded_libraries(artifacts, false)?;
    // Loading absolute private paths first registers their SONAMEs globally.
    // DT_NEEDED and vendor dlopen calls then reuse these references regardless
    // of LD_LIBRARY_PATH. Provider bridges must still be initialized by ORT.
    let load = |path: &Path| -> Result<()> {
        let loaded = unsafe { Library::open(Some(path), RTLD_NOW | RTLD_GLOBAL) }
            .map_err(|error| {
                tracing::error!(%error, library = %path.display(), "Private native Bibliothek kann nicht geladen werden");
                AppError::Unavailable
            })?;
        // into_raw deliberately retains a process-lifetime reference, including
        // on startup errors: ORT and vendor libraries keep process-global state.
        linux_loader::check_handle(loaded.into_raw(), path)
    };
    for entry in &artifacts.manifest.runtime.preload {
        let path = artifacts.resolve(entry)?;
        load(&path)?;
        let name = path.file_name().ok_or(AppError::Unavailable)?;
        if !linux_loader::check_soname(name, &path)? {
            tracing::error!(library = %path.display(), "Private CUDA-Bibliothek registriert nicht den erforderlichen SONAME");
            return Err(AppError::Unavailable);
        }
    }
    load(library)?;
    verify_loaded_libraries(artifacts, false)
}

#[cfg(target_os = "linux")]
fn verify_loaded_libraries(artifacts: &ArtifactSet, require_provider: bool) -> Result<()> {
    linux_loader::verify(artifacts, require_provider)
}

#[cfg(target_os = "linux")]
mod linux_loader {
    use super::*;
    use libloading::os::unix::{Library, RTLD_NOW};
    use std::{ffi::OsStr, os::unix::ffi::OsStrExt, ptr};

    // GNU/Linux dlfcn/link ABI. Only the documented common prefix is accessed.
    const RTLD_NOLOAD: i32 = 4;
    const RTLD_DI_LINKMAP: i32 = 2;

    #[repr(C)]
    struct LinkMap {
        address: usize,
        name: *const c_char,
    }

    #[repr(C)]
    struct PhdrInfo {
        address: usize,
        name: *const c_char,
    }

    #[link(name = "dl")]
    unsafe extern "C" {
        fn dlinfo(handle: *mut c_void, request: i32, info: *mut *mut LinkMap) -> i32;
        fn dl_iterate_phdr(
            callback: unsafe extern "C" fn(*mut PhdrInfo, usize, *mut c_void) -> i32,
            data: *mut c_void,
        ) -> i32;
    }

    fn check_path(actual: &Path, expected: &Path) -> Result<()> {
        let actual = fs::canonicalize(actual).map_err(|error| {
            tracing::error!(%error, library = %actual.display(), "Geladener Bibliothekspfad ist nicht auflösbar");
            AppError::Unavailable
        })?;
        if actual != expected {
            tracing::error!(actual = %actual.display(), expected = %expected.display(),
                "Fremde gleichnamige native Bibliothek wurde geladen");
            return Err(AppError::Unavailable);
        }
        Ok(())
    }

    pub(super) fn check_handle(handle: *mut c_void, expected: &Path) -> Result<()> {
        let mut map = ptr::null_mut();
        if unsafe { dlinfo(handle, RTLD_DI_LINKMAP, &mut map) } != 0 || map.is_null() {
            tracing::error!("Geladene ELF-Bibliothek kann nicht geprüft werden");
            return Err(AppError::Unavailable);
        }
        let name = unsafe { (*map).name };
        if name.is_null() {
            return Err(AppError::Unavailable);
        }
        // The dlopen reference keeps this link-map entry and name alive.
        let name = unsafe { CStr::from_ptr(name) };
        check_path(Path::new(OsStr::from_bytes(name.to_bytes())), expected)
    }

    pub(super) fn check_soname(name: &OsStr, expected: &Path) -> Result<bool> {
        // NOLOAD never searches by loading an ambient file or runs constructors.
        // It also finds foreign files loaded under a different basename but
        // advertising the expected SONAME (for example through LD_PRELOAD).
        let Ok(loaded) = (unsafe { Library::open(Some(name), RTLD_NOW | RTLD_NOLOAD) }) else {
            return Ok(false);
        };
        let handle = loaded.into_raw();
        let loaded = unsafe { Library::from_raw(handle) };
        let result = check_handle(handle, expected);
        drop(loaded);
        result.map(|()| true)
    }

    fn loaded_paths() -> Vec<PathBuf> {
        unsafe extern "C" fn collect(info: *mut PhdrInfo, size: usize, data: *mut c_void) -> i32 {
            if size >= std::mem::size_of::<PhdrInfo>() && !info.is_null() {
                let name = unsafe { (*info).name };
                if !name.is_null() {
                    let name = unsafe { CStr::from_ptr(name) }.to_bytes();
                    if !name.is_empty() {
                        // dl_iterate_phdr holds the loader lock while invoking
                        // us; copy the path before returning from the callback.
                        let paths = unsafe { &mut *data.cast::<Vec<PathBuf>>() };
                        paths.push(PathBuf::from(OsStr::from_bytes(name)));
                    }
                }
            }
            0
        }
        let mut paths = Vec::new();
        unsafe { dl_iterate_phdr(collect, ptr::from_mut(&mut paths).cast()) };
        paths
    }

    pub(super) fn verify(artifacts: &ArtifactSet, require_provider: bool) -> Result<()> {
        let runtime = &artifacts.manifest.runtime;
        let library = artifacts.resolve(&runtime.library)?;
        let paths = runtime
            .files
            .iter()
            .filter(|entry| {
                Path::new(&entry.path)
                    .file_name()
                    .and_then(OsStr::to_str)
                    .is_some_and(|name| name.contains(".so"))
            })
            .map(|entry| artifacts.resolve(&entry.path))
            .collect::<Result<Vec<_>>>()?;
        // Enumerate as well as query SONAMEs: dlopen(NOLOAD) returns only the
        // first match and must not hide a second same-name loaded object.
        for actual in loaded_paths() {
            let Some(name) = actual.file_name() else {
                continue;
            };
            let expected = if name
                .to_str()
                .is_some_and(|name| name.starts_with("libonnxruntime.so"))
            {
                Some(&library)
            } else {
                paths.iter().find(|path| path.file_name() == Some(name))
            };
            if let Some(expected) = expected {
                check_path(&actual, expected)?;
            }
        }
        for path in paths {
            let name = path.file_name().ok_or(AppError::Unavailable)?;
            check_soname(name, &path)?;
            // The CUDA provider has no SONAME and ORT opens its absolute path.
            let loaded = check_soname(path.as_os_str(), &path)?;
            let required = path == library
                || (runtime.provider == Provider::Cuda
                    && (name == "libonnxruntime_providers_cuda.so"
                        || name == "libonnxruntime_providers_shared.so"
                        || name
                            .to_str()
                            .is_some_and(|name| crate::artifacts::CUDA_PRELOAD.contains(&name))));
            if require_provider && required && !loaded {
                tracing::error!(library = %path.display(), "Native Sitzung hat keine geprüfte private Bibliothek geladen");
                return Err(AppError::Unavailable);
            }
            if require_provider && loaded {
                tracing::info!(library = %path.display(), "Private native Bibliotheksherkunft nach Warm-up geprüft");
            }
        }
        // The wheel's main library is copied to libonnxruntime.so, while ELF
        // dependency lookup may use its versioned SONAME.
        for alias in ["libonnxruntime.so.1", "libonnxruntime.so.1.26.0"] {
            check_soname(OsStr::new(alias), &library)?;
        }
        Ok(())
    }

    #[cfg(all(test, target_env = "gnu"))]
    mod tests {
        use super::*;

        #[test]
        fn loaded_soname_rejects_a_different_private_origin() {
            let actual = loaded_paths()
                .into_iter()
                .find(|path| path.file_name() == Some(OsStr::new("libc.so.6")))
                .expect("the GNU/Linux process has libc loaded");
            let actual = fs::canonicalize(actual).unwrap();
            assert!(check_soname(OsStr::new("libc.so.6"), &actual).unwrap());
            let private = std::env::temp_dir().join("backremove-private/libc.so.6");
            assert!(matches!(
                check_soname(OsStr::new("libc.so.6"), &private),
                Err(AppError::Unavailable)
            ));
        }
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
fn prepare_native_loader(_: &ArtifactSet, _: &Path) -> Result<()> {
    Ok(())
}

#[cfg(not(any(windows, target_os = "linux")))]
fn verify_loaded_libraries(_: &ArtifactSet, _: bool) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::is_valid_probability;

    #[test]
    fn sigmoid_rounding_allows_only_one_upper_endpoint_ulp() {
        assert!(is_valid_probability(0.0));
        assert!(is_valid_probability(1.0));
        assert!(is_valid_probability(1.0_f32.next_up()));
        assert!(!is_valid_probability(1.0_f32.next_up().next_up()));
        assert!(!is_valid_probability(-f32::from_bits(1)));
        assert!(!is_valid_probability(f32::NAN));
        assert!(!is_valid_probability(f32::INFINITY));
        assert!(!is_valid_probability(f32::NEG_INFINITY));
    }
}
