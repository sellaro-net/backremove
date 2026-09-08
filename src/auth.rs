use std::{
    collections::{HashMap, hash_map::Entry},
    net::IpAddr,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use axum::http::HeaderMap;
use ipnet::IpNet;
use subtle::ConstantTimeEq;

use crate::{
    config::Config,
    error::{AppError, Result},
};

const MAX_FAILURES: usize = 5;
const ATTEMPT_WINDOW: Duration = Duration::from_secs(60);
const BLOCK_DURATION: Duration = Duration::from_secs(900);

pub(crate) struct Authenticator {
    config: Arc<Config>,
    failures: Mutex<FailureMap>,
}

impl Authenticator {
    pub(crate) fn new(config: Arc<Config>) -> Self {
        Self {
            config,
            failures: Mutex::new(FailureMap::default()),
        }
    }

    pub(crate) fn authenticate(&self, headers: &HeaderMap, peer: IpAddr) -> Result<()> {
        let client = client_ip(headers, peer, &self.config.trusted_proxies);
        let mut keys = headers.get_all("x-api-key").iter();
        let key = keys.next().map_or(&[][..], |key| key.as_bytes());
        let matches = bool::from(key.ct_eq(self.config.api_key.as_slice()));
        let valid = matches && keys.next().is_none() && !self.config.api_key.is_empty();
        let mut failures = self.failures.lock().map_err(|_| {
            tracing::error!("Authentifizierungszustand nicht verfügbar");
            AppError::Internal
        })?;
        failures.check(client, valid, Instant::now(), self.config.auth_max_entries)
    }
}

fn canonical_ip(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V6(ip) => ip.to_ipv4_mapped().map_or(IpAddr::V6(ip), IpAddr::V4),
        ip => ip,
    }
}

fn client_ip(headers: &HeaderMap, peer: IpAddr, proxies: &[IpNet]) -> IpAddr {
    let canonical_peer = canonical_ip(peer);
    if !proxies
        .iter()
        .any(|proxy| proxy.contains(&peer) || proxy.contains(&canonical_peer))
    {
        return canonical_peer;
    }
    // Only the direct, explicitly trusted peer may supply this single-IP header.
    // Forwarded/X-Forwarded-For chains are deliberately not interpreted.
    let mut values = headers.get_all("cf-connecting-ip").iter();
    let value = values.next();
    if values.next().is_some() {
        return canonical_peer;
    }
    value
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<IpAddr>().ok())
        .map(canonical_ip)
        .unwrap_or(canonical_peer)
}

#[derive(Default)]
struct FailureMap {
    clients: HashMap<IpAddr, Failures>,
    next_cleanup: Option<Instant>,
}

struct Failures {
    attempts: [Instant; MAX_FAILURES],
    count: usize,
    blocked_until: Option<Instant>,
}

impl Failures {
    fn new(now: Instant) -> Self {
        Self {
            attempts: [now; MAX_FAILURES],
            count: 1,
            blocked_until: None,
        }
    }

    fn prune(&mut self, now: Instant) {
        if let Some(until) = self.blocked_until {
            if now >= until {
                self.blocked_until = None;
                self.count = 0;
            }
            return;
        }
        let mut count = 0;
        for index in 0..self.count {
            if now.saturating_duration_since(self.attempts[index]) < ATTEMPT_WINDOW {
                self.attempts[count] = self.attempts[index];
                count += 1;
            }
        }
        self.count = count;
    }

    fn record(&mut self, now: Instant) {
        self.attempts[self.count] = now;
        self.count += 1;
        if self.count == MAX_FAILURES {
            self.blocked_until = Some(now + BLOCK_DURATION);
        }
    }
}

impl FailureMap {
    fn check(&mut self, client: IpAddr, valid: bool, now: Instant, capacity: usize) -> Result<()> {
        // Full sweeps are amortized, not performed for every hostile request.
        // Occupied identities are always pruned individually below.
        if self.next_cleanup.is_none_or(|next| now >= next) {
            self.clients.retain(|_, failures| {
                failures.prune(now);
                failures.blocked_until.is_some() || failures.count > 0
            });
            self.next_cleanup = Some(now + ATTEMPT_WINDOW);
        }
        let full = self.clients.len() >= capacity;
        match self.clients.entry(client) {
            Entry::Occupied(mut entry) => {
                let failures = entry.get_mut();
                failures.prune(now);
                if failures.blocked_until.is_some() {
                    return Err(AppError::RateLimited);
                }
                if valid {
                    if failures.count == 0 {
                        entry.remove();
                    }
                    return Ok(());
                }
                failures.record(now);
            }
            Entry::Vacant(entry) => {
                if valid {
                    return Ok(());
                }
                // Never evict a live ban or rolling window to admit new identities.
                if full {
                    return Err(AppError::RateLimited);
                }
                entry.insert(Failures::new(now));
            }
        }
        // As in the previous API, the fifth bad key is 401 and starts the ban;
        // subsequent attempts (including a correct key) are 429 for 900 seconds.
        Err(AppError::Unauthorized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    #[test]
    fn fifth_failure_blocks_even_valid_credentials_until_expiry() {
        let mut failures = FailureMap::default();
        let client = "192.0.2.1".parse().unwrap();
        let now = Instant::now();
        for second in 0..5 {
            assert!(matches!(
                failures.check(client, false, now + Duration::from_secs(second), 1),
                Err(AppError::Unauthorized)
            ));
        }
        assert!(matches!(
            failures.check(client, true, now + Duration::from_secs(903), 1),
            Err(AppError::RateLimited)
        ));
        assert!(
            failures
                .check(client, true, now + Duration::from_secs(904), 1)
                .is_ok()
        );
    }

    #[test]
    fn rolling_window_expires_at_sixty_seconds_without_clearing_recent_failures() {
        let mut failures = FailureMap::default();
        let client = "192.0.2.1".parse().unwrap();
        let now = Instant::now();
        for second in [0, 10, 20, 30, 60] {
            assert!(matches!(
                failures.check(client, false, now + Duration::from_secs(second), 1),
                Err(AppError::Unauthorized)
            ));
        }
        assert!(
            failures
                .check(client, true, now + Duration::from_secs(60), 1)
                .is_ok()
        );
        assert!(matches!(
            failures.check(client, false, now + Duration::from_secs(61), 1),
            Err(AppError::Unauthorized)
        ));
        assert!(matches!(
            failures.check(client, true, now + Duration::from_secs(61), 1),
            Err(AppError::RateLimited)
        ));
    }

    #[test]
    fn saturated_identity_map_cannot_reset_an_existing_ban() {
        let mut failures = FailureMap::default();
        let banned = "192.0.2.1".parse().unwrap();
        let other = "192.0.2.2".parse().unwrap();
        let now = Instant::now();
        for _ in 0..5 {
            assert!(matches!(
                failures.check(banned, false, now, 1),
                Err(AppError::Unauthorized)
            ));
        }
        assert!(matches!(
            failures.check(other, false, now, 1),
            Err(AppError::RateLimited)
        ));
        assert!(failures.check(other, true, now, 1).is_ok());
        assert!(matches!(
            failures.check(banned, true, now, 1),
            Err(AppError::RateLimited)
        ));
        assert!(matches!(
            failures.check(other, false, now + BLOCK_DURATION, 1),
            Err(AppError::Unauthorized)
        ));
    }

    #[test]
    fn only_trusted_direct_peers_can_supply_one_cloudflare_address() {
        let mut headers = HeaderMap::new();
        headers.insert("cf-connecting-ip", HeaderValue::from_static("198.51.100.5"));
        let trusted = ["127.0.0.0/8".parse().unwrap()];
        let direct = "192.0.2.1".parse().unwrap();
        assert_eq!(client_ip(&headers, direct, &trusted), direct);
        assert_eq!(
            client_ip(&headers, "127.0.0.1".parse().unwrap(), &[]),
            "127.0.0.1".parse::<IpAddr>().unwrap()
        );
        assert_eq!(
            client_ip(&headers, "127.0.0.1".parse().unwrap(), &trusted),
            "198.51.100.5".parse::<IpAddr>().unwrap()
        );
        headers.append("cf-connecting-ip", HeaderValue::from_static("198.51.100.6"));
        assert_eq!(
            client_ip(&headers, "::ffff:127.0.0.1".parse().unwrap(), &trusted),
            "127.0.0.1".parse::<IpAddr>().unwrap()
        );
    }
}
