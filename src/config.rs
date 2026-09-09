use crate::types::Model;
use ipnet::IpNet;
use std::{
    env,
    net::{IpAddr, SocketAddr},
    path::PathBuf,
    time::Duration,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Auto,
    Cpu,
    Cuda,
}

#[derive(Debug, Clone)]
pub struct ImageLimits {
    pub max_file_bytes: usize,
    pub max_multipart_bytes: usize,
    pub max_pixels: u64,
    pub max_output_bytes: usize,
    pub max_svg_nodes: usize,
    pub max_svg_depth: usize,
    pub max_svg_embedded_bytes: usize,
    pub max_svg_embedded_pixels: u64,
}
impl Default for ImageLimits {
    fn default() -> Self {
        Self {
            max_file_bytes: 20 * 1024 * 1024,
            max_multipart_bytes: 20 * 1024 * 1024 + 64 * 1024,
            max_pixels: 40_000_000,
            max_output_bytes: 32 * 1024 * 1024,
            max_svg_nodes: 20_000,
            max_svg_depth: 64,
            max_svg_embedded_bytes: 20 * 1024 * 1024,
            max_svg_embedded_pixels: 40_000_000,
        }
    }
}

// Deliberately no Debug: configuration contains the authentication secret.
pub struct Config {
    pub bind: SocketAddr,
    pub api_key: Vec<u8>,
    pub cors_origins: Vec<String>,
    pub trusted_proxies: Vec<IpNet>,
    pub manifest: PathBuf,
    pub device: Device,
    pub quality_enabled: bool,
    pub queue_capacity: usize,
    pub max_jobs: usize,
    pub input_budget_bytes: usize,
    pub memory_budget_bytes: usize,
    pub fast_timeout: Duration,
    pub quality_timeout: Duration,
    pub shutdown_grace: Duration,
    pub image: ImageLimits,
    pub auth_max_entries: usize,
}
impl Config {
    pub fn load() -> std::result::Result<Self, String> {
        let api_key = env::var("API_KEY").map_err(|_| "API_KEY muss gesetzt sein.".to_owned())?;
        if api_key.is_empty()
            || !api_key.is_ascii()
            || api_key.len() > 1024
            || api_key.bytes().any(|b| b.is_ascii_control())
        {
            return Err("API_KEY muss 1 bis 1024 druckbare ASCII-Zeichen enthalten.".into());
        }
        let host: IpAddr = value("HOST", "0.0.0.0")
            .parse()
            .map_err(|_| "HOST muss eine gültige IP-Adresse sein.")?;
        let port = integer("PORT", 8585, 1, 65535)? as u16;
        let device = match value("INFERENCE_DEVICE", "auto").as_str() {
            "auto" => Device::Auto,
            "cpu" => Device::Cpu,
            "cuda" => Device::Cuda,
            _ => return Err("INFERENCE_DEVICE muss auto, cpu oder cuda sein.".into()),
        };
        let quality_enabled = boolean("QUALITY_MODEL_ENABLED", false)?;
        let default_manifest = if cfg!(windows) {
            if device == Device::Cpu {
                "artifacts/windows-cpu/manifest.json"
            } else {
                "artifacts/windows-cuda/manifest.json"
            }
        } else if cfg!(target_os = "linux") && device == Device::Cuda {
            "artifacts/linux-cuda/manifest.json"
        } else {
            "artifacts/linux-cpu/manifest.json"
        };
        let manifest = PathBuf::from(value("ARTIFACT_MANIFEST", default_manifest));
        let cors_origins = csv("CORS_ORIGINS");
        if cors_origins.iter().any(|v| v == "*") && cors_origins.len() != 1 {
            return Err("CORS_ORIGINS darf * nur als einzigen Ursprung enthalten.".into());
        }
        for origin in &cors_origins {
            axum::http::HeaderValue::from_str(origin)
                .map_err(|_| "CORS_ORIGINS enthält einen ungültigen Ursprung.")?;
            if origin != "*" {
                let uri: axum::http::Uri = origin
                    .parse()
                    .map_err(|_| "CORS_ORIGINS enthält einen ungültigen Ursprung.")?;
                if !matches!(uri.scheme_str(), Some("http" | "https"))
                    || uri.authority().is_none()
                    || uri
                        .authority()
                        .is_some_and(|authority| authority.as_str().contains('@'))
                    || uri
                        .path_and_query()
                        .is_some_and(|path| path.as_str() != "/")
                {
                    return Err(
                        "CORS_ORIGINS muss aus HTTP(S)-Ursprüngen ohne Pfad bestehen.".into(),
                    );
                }
            }
        }
        let trusted_proxies = csv("TRUSTED_PROXIES")
            .into_iter()
            .map(|v| {
                v.parse::<IpNet>()
                    .or_else(|_| v.parse::<IpAddr>().map(IpNet::from))
                    .map_err(|_| {
                        "TRUSTED_PROXIES muss gültige IP-Adressen oder CIDR-Netze enthalten."
                            .to_owned()
                    })
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let queue_capacity = integer("QUEUE_CAPACITY", 8, 1, 1024)?;
        let image = ImageLimits::default();
        let input_budget_bytes = mebibytes("INPUT_BUDGET_MB", 256, 21, 16_384)?;
        let memory_budget_bytes = mebibytes("MEMORY_BUDGET_MB", 1024, 128, 65_536)?;
        Ok(Self {
            bind: SocketAddr::new(host, port),
            api_key: api_key.into_bytes(),
            cors_origins,
            trusted_proxies,
            manifest,
            device,
            quality_enabled,
            queue_capacity,
            max_jobs: queue_capacity + 1,
            input_budget_bytes,
            memory_budget_bytes,
            fast_timeout: seconds("FAST_TIMEOUT", 9.0, 300.0)?,
            quality_timeout: seconds("QUALITY_TIMEOUT", 29.0, 300.0)?,
            shutdown_grace: seconds("SHUTDOWN_GRACE", 30.0, 600.0)?,
            image,
            auth_max_entries: integer("AUTH_MAX_ENTRIES", 10_000, 1, 100_000)?,
        })
    }
    pub fn timeout(&self, model: Model) -> Duration {
        match model {
            Model::Fast => self.fast_timeout,
            Model::Quality => self.quality_timeout,
        }
    }
}
fn value(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_owned())
}
fn csv(name: &str) -> Vec<String> {
    env::var(name)
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_owned)
        .collect()
}
fn integer(
    name: &str,
    default: usize,
    min: usize,
    max: usize,
) -> std::result::Result<usize, String> {
    let parsed = match env::var(name) {
        Ok(v) => v
            .parse()
            .map_err(|_| format!("{name} muss eine ganze Zahl sein."))?,
        Err(env::VarError::NotPresent) => default,
        Err(_) => return Err(format!("{name} enthält ungültige Zeichen.")),
    };
    if !(min..=max).contains(&parsed) {
        return Err(format!("{name} muss zwischen {min} und {max} liegen."));
    }
    Ok(parsed)
}
fn mebibytes(
    name: &str,
    default: usize,
    min: usize,
    max: usize,
) -> std::result::Result<usize, String> {
    integer(name, default, min, max)?
        .checked_mul(1024 * 1024)
        .ok_or_else(|| format!("{name} überschreitet den unterstützten Bereich."))
}
fn seconds(name: &str, default: f64, max: f64) -> std::result::Result<Duration, String> {
    let parsed: f64 = match env::var(name) {
        Ok(v) => v
            .parse()
            .map_err(|_| format!("{name} muss eine Sekundenzahl sein."))?,
        Err(env::VarError::NotPresent) => default,
        Err(_) => return Err(format!("{name} enthält ungültige Zeichen.")),
    };
    if !parsed.is_finite() || parsed < 0.001 || parsed > max {
        return Err(format!(
            "{name} muss zwischen 0,001 und {max} Sekunden liegen."
        ));
    }
    Ok(Duration::from_secs_f64(parsed))
}
fn boolean(name: &str, default: bool) -> std::result::Result<bool, String> {
    match env::var(name) {
        Err(env::VarError::NotPresent) => Ok(default),
        Ok(v) => match v.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => Err(format!("{name} muss ein boolescher Wert sein.")),
        },
        Err(_) => Err(format!("{name} enthält ungültige Zeichen.")),
    }
}
