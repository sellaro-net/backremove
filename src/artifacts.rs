use crate::{
    config::Device,
    error::{AppError, Result},
    types::{Model, ModelSpec, Precision},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

pub const RUNTIME_VERSION: &str = "1.26.0";
pub const RUNTIME_API: u32 = 26;

// This is the order exercised by the full-model GTX 1060 probe. Provider DLLs
// are deliberately absent: ORT must initialize its bridge before loading them.
pub(crate) const CUDA_PRELOAD: &[&str] = &[
    "cudart64_12.dll",
    "cublasLt64_12.dll",
    "cublas64_12.dll",
    "cufft64_11.dll",
    "nvrtc-builtins64_128.dll",
    "nvrtc64_120_0.dll",
    "cudnn64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_heuristic64_9.dll",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    Cpu,
    Cuda,
}

impl Provider {
    pub const fn execution_provider(self) -> &'static str {
        match self {
            Self::Cpu => "CPUExecutionProvider",
            Self::Cuda => "CUDAExecutionProvider",
        }
    }

    pub(crate) fn check_device(self, device: Device) -> Result<()> {
        if matches!(
            (device, self),
            (Device::Cpu, Self::Cuda) | (Device::Cuda, Self::Cpu)
        ) {
            tracing::error!(
                provider = self.execution_provider(),
                "Artefaktpaket passt nicht zum angeforderten Gerät"
            );
            return Err(AppError::Unavailable);
        }
        if self == Self::Cuda && !cfg!(all(windows, target_arch = "x86_64")) {
            tracing::error!("Das geprüfte CUDA-Paket erfordert Windows x64");
            return Err(AppError::Unavailable);
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
pub struct HashedFile {
    pub path: String,
    pub sha256: String,
}

#[derive(Debug, Deserialize)]
pub struct RuntimeManifest {
    pub version: String,
    pub provider: Provider,
    pub library: String,
    pub files: Vec<HashedFile>,
    pub preload: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct ModelManifest {
    pub path: String,
    pub sha256: String,
    pub input_name: String,
    pub output_name: String,
    pub input_size: u32,
    pub precision: Precision,
}

impl ModelManifest {
    pub fn spec(&self, model: Model) -> ModelSpec {
        ModelSpec {
            model,
            input_size: self.input_size,
            precision: self.precision,
        }
    }

    fn validate(&self, model: Model) -> Result<()> {
        let valid = match model {
            Model::Fast => {
                self.input_name == "rgb"
                    && self.output_name == "alpha"
                    && self.input_size == 448
                    && self.precision == Precision::Fp32
            }
            Model::Quality => {
                self.input_name == "input"
                    && self.output_name == "mask"
                    && self.input_size == 1024
                    && self.precision == Precision::Fp16
            }
        };
        if !valid {
            tracing::error!(
                model = model.as_str(),
                "Modellmetadaten entsprechen nicht dem freigegebenen Tensorvertrag"
            );
            return Err(AppError::Unavailable);
        }
        validate_relative_path(&self.path)?;
        parse_hash(&self.sha256)?;
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub version: u32,
    pub runtime: RuntimeManifest,
    pub fast: ModelManifest,
    pub quality: Option<ModelManifest>,
    pub fonts: Vec<HashedFile>,
    #[serde(default)]
    pub provenance: serde_json::Value,
}

pub struct ArtifactSet {
    pub root: PathBuf,
    pub manifest: Manifest,
    pub font_paths: Vec<PathBuf>,
    pub(crate) quality_enabled: bool,
}

impl ArtifactSet {
    pub fn load(manifest_path: &Path, device: Device, quality_enabled: bool) -> Result<Self> {
        let manifest_path = fs::canonicalize(manifest_path).map_err(|error| {
            tracing::error!(%error, path = %manifest_path.display(), "Artefaktmanifest ist nicht erreichbar");
            AppError::Unavailable
        })?;
        let root = manifest_path
            .parent()
            .ok_or_else(|| {
                tracing::error!("Artefaktmanifest hat kein Stammverzeichnis");
                AppError::Unavailable
            })?
            .to_path_buf();
        let file = File::open(&manifest_path).map_err(|error| {
            tracing::error!(%error, "Artefaktmanifest kann nicht geöffnet werden");
            AppError::Unavailable
        })?;
        let manifest: Manifest =
            serde_json::from_reader(std::io::BufReader::new(file)).map_err(|error| {
                tracing::error!(%error, "Artefaktmanifest ist ungültig");
                AppError::Unavailable
            })?;
        if manifest.version != 1 || manifest.runtime.version != RUNTIME_VERSION {
            tracing::error!(manifest_version = manifest.version, runtime_version = %manifest.runtime.version,
                "Nicht unterstützte Artefakt- oder Runtime-Version");
            return Err(AppError::Unavailable);
        }
        manifest.runtime.provider.check_device(device)?;
        manifest.fast.validate(Model::Fast)?;
        if let Some(quality) = &manifest.quality {
            quality.validate(Model::Quality)?;
            if manifest.runtime.provider == Provider::Cpu {
                tracing::error!("Das CPU-Paket darf kein GPU-Quality-Modell anbieten");
                return Err(AppError::Unavailable);
            }
        }
        let mut artifacts = Self {
            root,
            manifest,
            font_paths: Vec::new(),
            quality_enabled,
        };
        artifacts.validate_runtime()?;
        artifacts.verify_file(
            &artifacts.manifest.fast.path,
            &artifacts.manifest.fast.sha256,
        )?;
        if quality_enabled && let Some(quality) = &artifacts.manifest.quality {
            artifacts.verify_file(&quality.path, &quality.sha256)?;
        }
        if artifacts.manifest.fonts.is_empty() {
            tracing::error!("Das Artefaktpaket enthält keinen versionierten SVG-Schriftbestand");
            return Err(AppError::Unavailable);
        }
        let mut fonts = HashSet::new();
        for font in &artifacts.manifest.fonts {
            let path = artifacts.verify_file(&font.path, &font.sha256)?;
            if !fonts.insert(path.clone()) {
                tracing::error!("Doppelter Eintrag im SVG-Schriftbestand");
                return Err(AppError::Unavailable);
            }
            artifacts.font_paths.push(path);
        }
        tracing::info!(
            provider = artifacts.manifest.runtime.provider.execution_provider(),
            runtime_version = RUNTIME_VERSION,
            "Ausgewählte Modelle, Runtime und Schriften sind SHA-256-geprüft"
        );
        Ok(artifacts)
    }

    /// Manifest paths are portable relative paths. Canonical containment also
    /// rejects symlinks/junctions that lead out of the immutable artifact pack.
    pub fn resolve(&self, path: &str) -> Result<PathBuf> {
        validate_relative_path(path)?;
        let resolved = fs::canonicalize(self.root.join(path)).map_err(|error| {
            tracing::error!(%error, artifact = path, "Artefaktpfad ist nicht erreichbar");
            AppError::Unavailable
        })?;
        if !resolved.starts_with(&self.root) || !resolved.is_file() {
            tracing::error!(
                artifact = path,
                "Artefakt liegt außerhalb des Pakets oder ist keine Datei"
            );
            return Err(AppError::Unavailable);
        }
        Ok(resolved)
    }

    fn verify_file(&self, path: &str, expected: &str) -> Result<PathBuf> {
        let expected = parse_hash(expected)?;
        let resolved = self.resolve(path)?;
        let mut file = File::open(&resolved).map_err(|error| {
            tracing::error!(%error, artifact = path, "Artefakt kann nicht geöffnet werden");
            AppError::Unavailable
        })?;
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let count = file.read(&mut buffer).map_err(|error| {
                tracing::error!(%error, artifact = path, "Artefakt kann nicht vollständig geprüft werden");
                AppError::Unavailable
            })?;
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
        }
        let actual: [u8; 32] = hasher.finalize().into();
        if actual != expected {
            tracing::error!(
                artifact = path,
                "SHA-256-Prüfung des Artefakts fehlgeschlagen"
            );
            return Err(AppError::Unavailable);
        }
        Ok(resolved)
    }

    fn validate_runtime(&self) -> Result<()> {
        let runtime = &self.manifest.runtime;
        let expected_library = if cfg!(windows) {
            "onnxruntime.dll"
        } else {
            "libonnxruntime.so"
        };
        if !cfg!(any(windows, target_os = "linux"))
            || Path::new(&runtime.library)
                .file_name()
                .and_then(|v| v.to_str())
                != Some(expected_library)
        {
            tracing::error!("Runtime-Bibliothek passt nicht zum Betriebssystem");
            return Err(AppError::Unavailable);
        }
        let library = self.resolve(&runtime.library)?;
        let mut verified = HashSet::new();
        let mut names = HashSet::new();
        for file in &runtime.files {
            let resolved = self.verify_file(&file.path, &file.sha256)?;
            let name = resolved
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or(AppError::Unavailable)?;
            if !verified.insert(resolved.clone()) || !names.insert(name.to_ascii_lowercase()) {
                tracing::error!(artifact = %file.path, "Doppelte Runtime-Datei oder kollidierender Bibliotheksname");
                return Err(AppError::Unavailable);
            }
        }
        if !verified.contains(&library) {
            tracing::error!("Runtime-Hauptbibliothek fehlt in der Hashliste");
            return Err(AppError::Unavailable);
        }
        let mut preload_names = Vec::with_capacity(runtime.preload.len());
        for entry in &runtime.preload {
            let path = self.resolve(entry)?;
            if !verified.contains(&path) {
                tracing::error!(
                    artifact = entry,
                    "Vorgeladene Bibliothek fehlt in der Hashliste"
                );
                return Err(AppError::Unavailable);
            }
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or(AppError::Unavailable)?;
            preload_names.push(name.to_owned());
        }
        match runtime.provider {
            Provider::Cuda => {
                if !preload_names
                    .iter()
                    .map(String::as_str)
                    .eq(CUDA_PRELOAD.iter().copied())
                {
                    tracing::error!(
                        "CUDA-Preloadliste entspricht nicht der geprüften DLL-Reihenfolge"
                    );
                    return Err(AppError::Unavailable);
                }
                for required in [
                    "curand64_10.dll",
                    "nvJitLink_120_0.dll",
                    "onnxruntime_providers_shared.dll",
                    "onnxruntime_providers_cuda.dll",
                ] {
                    if !names.contains(&required.to_ascii_lowercase()) {
                        tracing::error!(
                            dependency = required,
                            "Erforderliche CUDA-Abhängigkeit fehlt im Manifest"
                        );
                        return Err(AppError::Unavailable);
                    }
                }
                let directory = library.parent().ok_or(AppError::Unavailable)?;
                for provider in [
                    "onnxruntime_providers_shared.dll",
                    "onnxruntime_providers_cuda.dll",
                ] {
                    let path = fs::canonicalize(directory.join(provider)).map_err(|error| {
                        tracing::error!(%error, dependency = provider, "ORT-Provider liegt nicht neben der Runtime");
                        AppError::Unavailable
                    })?;
                    if !verified.contains(&path) {
                        tracing::error!(dependency = provider, "ORT-Provider wurde nicht geprüft");
                        return Err(AppError::Unavailable);
                    }
                }
            }
            Provider::Cpu if !runtime.preload.is_empty() => {
                tracing::error!("Das native CPU-Paket benötigt keine CUDA-Preloads");
                return Err(AppError::Unavailable);
            }
            Provider::Cpu => {}
        }
        Ok(())
    }
}

fn validate_relative_path(path: &str) -> Result<()> {
    if path.is_empty()
        || path.contains(['\\', ':', '\0'])
        || path
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
        || Path::new(path).is_absolute()
    {
        tracing::error!("Artefaktpfad ist kein gültiger relativer Paketpfad");
        return Err(AppError::Unavailable);
    }
    Ok(())
}

fn parse_hash(value: &str) -> Result<[u8; 32]> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        tracing::error!("Artefaktmanifest enthält keinen gültigen SHA-256-Wert");
        return Err(AppError::Unavailable);
    }
    let mut hash = [0; 32];
    for (index, byte) in hash.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| AppError::Unavailable)?;
    }
    Ok(hash)
}
