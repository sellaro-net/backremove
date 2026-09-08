use std::fmt;

pub type Result<T> = std::result::Result<T, AppError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppError {
    Busy,
    Deadline,
    UploadTimeout,
    Unavailable,
    InvalidImage,
    TooLarge,
    UnsupportedMedia,
    InvalidRequest,
    Unauthorized,
    RateLimited,
    Internal,
}
impl AppError {
    pub const fn code(self) -> &'static str {
        match self {
            Self::Busy => "busy",
            Self::Deadline => "deadline_exceeded",
            Self::UploadTimeout => "upload_timeout",
            Self::Unavailable => "model_unavailable",
            Self::InvalidImage => "invalid_image",
            Self::TooLarge => "payload_too_large",
            Self::UnsupportedMedia => "unsupported_media_type",
            Self::InvalidRequest => "invalid_request",
            Self::Unauthorized => "unauthorized",
            Self::RateLimited => "rate_limited",
            Self::Internal => "internal_error",
        }
    }
    pub const fn status(self) -> u16 {
        match self {
            Self::Busy | Self::Unavailable => 503,
            Self::Deadline => 504,
            Self::UploadTimeout => 408,
            Self::InvalidImage => 400,
            Self::TooLarge => 413,
            Self::UnsupportedMedia => 415,
            Self::InvalidRequest => 422,
            Self::Unauthorized => 401,
            Self::RateLimited => 429,
            Self::Internal => 500,
        }
    }
    pub const fn message(self) -> &'static str {
        match self {
            Self::Busy => "Die Bildverarbeitung ist derzeit ausgelastet. Bitte erneut versuchen.",
            Self::Deadline => "Die Zeit für die Bildverarbeitung wurde überschritten.",
            Self::UploadTimeout => "Die Zeit für die Bildübertragung wurde überschritten.",
            Self::Unavailable => "Das angeforderte Freistellungsmodell ist nicht verfügbar.",
            Self::InvalidImage => "Die Bilddatei ist beschädigt oder ungültig.",
            Self::TooLarge => {
                "Die Bilddatei überschreitet die erlaubten Größen- oder Ressourcengrenzen."
            }
            Self::UnsupportedMedia => "Dieses Bildformat wird nicht unterstützt.",
            Self::InvalidRequest => "Die Anfrage ist ungültig.",
            Self::Unauthorized => "Der API-Schlüssel fehlt oder ist ungültig.",
            Self::RateLimited => {
                "Zu viele fehlgeschlagene Anmeldungen. Bitte später erneut versuchen."
            }
            Self::Internal => "Die Hintergrundentfernung ist fehlgeschlagen.",
        }
    }
    pub const fn retry_after(self) -> Option<u64> {
        match self {
            Self::Busy => Some(2),
            Self::RateLimited => Some(900),
            _ => None,
        }
    }
}
impl fmt::Display for AppError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.message())
    }
}
impl std::error::Error for AppError {}
