use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use axum::http::{header::HeaderName, HeaderMap};
use dashmap::DashMap;
use jsonwebtoken::{decode, Algorithm, DecodingKey, Validation};
use lethe_shared::config::{JwtConfig, RateLimitConfig, SecurityConfig};
use serde::Deserialize;
use subtle::ConstantTimeEq;

use crate::error::ApiError;

/// Represents the authentication method that validated a request.
#[derive(Debug, Clone, Copy)]
pub enum AuthMethod {
    ApiKey,
    Jwt,
}

/// Authentication context that can be attached to the request extensions.
#[derive(Debug, Clone)]
pub struct AuthenticatedIdentity {
    pub method: AuthMethod,
    pub subject: Option<String>,
    pub roles: Vec<String>,
    pub permissions: Vec<String>,
}

impl AuthenticatedIdentity {
    pub fn has_role(&self, role: &str) -> bool {
        self.roles.iter().any(|candidate| candidate == role)
    }

    pub fn has_permission(&self, permission: &str) -> bool {
        self.permissions
            .iter()
            .any(|candidate| candidate == permission)
    }
}

/// Security context shared by middleware for auth and rate limiting.
#[derive(Clone)]
pub struct SecurityContext {
    require_authentication: bool,
    api_keys: Vec<ApiKey>,
    api_key_header: Option<HeaderName>,
    jwt: Option<JwtValidator>,
    rate_limiter: Option<Arc<RateLimiter>>,
    client_ip_header: Option<HeaderName>,
}

impl SecurityContext {
    pub fn from_config(config: &SecurityConfig) -> Result<Self, ApiError> {
        if config.require_authentication && config.api_keys.is_empty() && config.jwt.is_none() {
            return Err(ApiError::internal(
                "Authentication required but neither API keys nor JWT are configured",
            ));
        }

        let api_keys = config
            .api_keys
            .iter()
            .map(|key| ApiKey::new(key))
            .collect::<Vec<_>>();

        let api_key_header = config.api_key_header.as_ref().and_then(|name| {
            HeaderName::from_lowercase(name.trim().to_ascii_lowercase().as_bytes()).ok()
        });

        let jwt = match &config.jwt {
            Some(jwt_cfg) => Some(JwtValidator::new(jwt_cfg)?),
            None => None,
        };

        let rate_limiter = match &config.rate_limit {
            Some(limit_cfg) => Some(Arc::new(RateLimiter::new(limit_cfg)?)),
            None => None,
        };

        let client_ip_header = config.client_ip_header.as_ref().and_then(|name| {
            HeaderName::from_lowercase(name.trim().to_ascii_lowercase().as_bytes()).ok()
        });

        Ok(Self {
            require_authentication: config.require_authentication,
            api_keys,
            api_key_header,
            jwt,
            rate_limiter,
            client_ip_header,
        })
    }

    pub fn authentication_required(&self) -> bool {
        self.require_authentication
    }

    pub fn try_api_key(&self, candidate: &str) -> Option<AuthenticatedIdentity> {
        let token = candidate.trim();
        if token.is_empty() {
            return None;
        }

        self.api_keys
            .iter()
            .find(|key| key.matches(token))
            .map(|key| AuthenticatedIdentity {
                method: AuthMethod::ApiKey,
                subject: None,
                roles: key.roles.clone(),
                permissions: key.permissions.clone(),
            })
    }

    pub fn try_jwt(&self, token: &str) -> Result<Option<AuthenticatedIdentity>, ApiError> {
        match &self.jwt {
            Some(validator) => {
                let auth = validator.validate(token)?;
                Ok(Some(AuthenticatedIdentity {
                    method: AuthMethod::Jwt,
                    subject: auth.subject,
                    roles: auth.roles,
                    permissions: auth.permissions,
                }))
            }
            None => Ok(None),
        }
    }

    pub fn rate_limiter(&self) -> Option<Arc<RateLimiter>> {
        self.rate_limiter.clone()
    }

    pub fn extract_client_identifier(&self, headers: &HeaderMap) -> String {
        if let Some(ref header_name) = self.client_ip_header {
            if let Some(value) = headers.get(header_name) {
                if let Ok(text) = value.to_str() {
                    if let Some(identifier) = text.split(',').next() {
                        return identifier.trim().to_string();
                    }
                }
            }
        }

        headers
            .get("x-forwarded-for")
            .and_then(|value| value.to_str().ok())
            .and_then(|text| text.split(',').next())
            .map(|s| s.trim().to_string())
            .or_else(|| {
                headers
                    .get("x-real-ip")
                    .and_then(|value| value.to_str().ok())
                    .map(|s| s.trim().to_string())
            })
            .unwrap_or_else(|| "unknown".to_string())
    }

    pub fn api_key_header(&self) -> Option<&HeaderName> {
        self.api_key_header.as_ref()
    }
}

#[derive(Clone)]
struct ApiKey {
    value: Vec<u8>,
    roles: Vec<String>,
    permissions: Vec<String>,
}

impl ApiKey {
    fn new(raw: &str) -> Self {
        let mut parts = raw.split('|');
        let value_part = parts.next().unwrap_or("").trim();

        let parse_list = |input: Option<&str>, defaults: &[&str]| -> Vec<String> {
            input
                .map(|s| {
                    s.split(',')
                        .filter_map(|item| {
                            let trimmed = item.trim();
                            if trimmed.is_empty() {
                                None
                            } else {
                                Some(trimmed.to_string())
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .filter(|list: &Vec<String>| !list.is_empty())
                .unwrap_or_else(|| defaults.iter().map(|item| (*item).to_string()).collect())
        };

        let roles = parse_list(parts.next(), &["api_key"]);
        let permissions = parse_list(parts.next(), &["query:execute"]);

        Self {
            value: value_part.as_bytes().to_vec(),
            roles,
            permissions,
        }
    }

    fn matches(&self, candidate: &str) -> bool {
        self.value.as_slice().ct_eq(candidate.as_bytes()).into()
    }
}

#[derive(Clone)]
struct JwtValidator {
    decoding_key: DecodingKey,
    validation: Validation,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct Claims {
    sub: Option<String>,
    iss: Option<String>,
    aud: Option<Vec<String>>,
    exp: Option<u64>,
    iat: Option<u64>,
    #[serde(default)]
    roles: Option<Vec<String>>,
    #[serde(default)]
    permissions: Option<Vec<String>>,
    #[serde(default)]
    scope: Option<String>,
}

impl JwtValidator {
    fn new(config: &JwtConfig) -> Result<Self, ApiError> {
        if config.secret.trim().is_empty() {
            return Err(ApiError::internal("JWT secret cannot be empty"));
        }

        let mut validation = Validation::new(Algorithm::HS256);
        validation.validate_exp = true;
        validation.leeway = config.leeway_seconds;

        if let Some(ref issuer) = config.issuer {
            validation.set_issuer(&[issuer.clone()]);
        }

        if let Some(ref audience) = config.audience {
            validation.set_audience(&[audience.clone()]);
        }

        let decoding_key = DecodingKey::from_secret(config.secret.as_bytes());

        Ok(Self {
            decoding_key,
            validation,
        })
    }

    fn validate(&self, token: &str) -> Result<JwtAuthResult, ApiError> {
        let data =
            decode::<Claims>(token, &self.decoding_key, &self.validation).map_err(|err| {
                tracing::warn!(error = %err, "JWT validation failed");
                ApiError::Authentication
            })?;

        let claims = data.claims;
        let roles = claims.resolve_roles();
        let permissions = claims.resolve_permissions();
        Ok(JwtAuthResult {
            subject: claims.sub,
            roles,
            permissions,
        })
    }
}

impl Claims {
    fn resolve_roles(&self) -> Vec<String> {
        let mut roles = self.roles.clone().unwrap_or_default();
        if roles.is_empty() {
            roles.push("jwt_user".to_string());
        }
        dedup_string_list(&mut roles);
        roles
    }

    fn resolve_permissions(&self) -> Vec<String> {
        let mut permissions = self.permissions.clone().unwrap_or_default();
        if let Some(scope) = &self.scope {
            permissions.extend(
                scope
                    .split_whitespace()
                    .filter(|value| !value.trim().is_empty())
                    .map(|value| value.trim().to_string()),
            );
        }
        if permissions.is_empty() {
            permissions.push("query:execute".to_string());
        }
        dedup_string_list(&mut permissions);
        permissions
    }
}

fn dedup_string_list(list: &mut Vec<String>) {
    for item in list.iter_mut() {
        let trimmed = item.trim();
        if trimmed.len() != item.len() {
            *item = trimmed.to_string();
        }
    }
    list.retain(|item| !item.is_empty());
    list.sort();
    list.dedup();
}

#[derive(Debug)]
struct JwtAuthResult {
    subject: Option<String>,
    roles: Vec<String>,
    permissions: Vec<String>,
}

/// Simple token bucket rate limiter shared between requests.
#[derive(Debug)]
pub struct RateLimiter {
    capacity: f64,
    refill_per_second: f64,
    buckets: DashMap<String, Bucket>,
}

#[derive(Debug, Clone)]
struct Bucket {
    tokens: f64,
    last_refill: Instant,
}

/// The outcome of a rate-limit check.
#[derive(Debug, Clone, Copy)]
pub enum RateLimitOutcome {
    Allow,
    Deny { retry_after: Duration },
}

impl RateLimiter {
    fn new(config: &RateLimitConfig) -> Result<Self, ApiError> {
        if config.requests_per_minute == 0 {
            return Err(ApiError::internal(
                "Rate limit misconfigured: requests_per_minute must be greater than zero",
            ));
        }

        let burst = if config.burst == 0 {
            config.requests_per_minute
        } else {
            config.burst
        } as f64;

        let refill_per_second = config.requests_per_minute as f64 / 60.0;

        Ok(Self {
            capacity: burst,
            refill_per_second,
            buckets: DashMap::new(),
        })
    }

    pub fn check(&self, key: &str) -> RateLimitOutcome {
        if self.refill_per_second <= f64::EPSILON {
            return RateLimitOutcome::Allow;
        }

        let now = Instant::now();
        let mut entry = self
            .buckets
            .entry(key.to_string())
            .or_insert_with(|| Bucket {
                tokens: self.capacity,
                last_refill: now,
            });

        let elapsed = now.duration_since(entry.last_refill).as_secs_f64();
        if elapsed > 0.0 {
            let replenished = elapsed * self.refill_per_second;
            entry.tokens = (entry.tokens + replenished).min(self.capacity);
            entry.last_refill = now;
        }

        if entry.tokens >= 1.0 {
            entry.tokens -= 1.0;
            RateLimitOutcome::Allow
        } else {
            let missing = 1.0 - entry.tokens;
            let wait_seconds = (missing / self.refill_per_second).max(0.5);
            RateLimitOutcome::Deny {
                retry_after: Duration::from_secs_f64(wait_seconds),
            }
        }
    }
}
