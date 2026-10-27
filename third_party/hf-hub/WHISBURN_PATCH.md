# Vendored hf-hub (0.5.0)

Patched for whisburn so HTTPS does not need OpenSSL, ring, or a C compiler:

- Upstream `ureq` 3 defaults enable rustls → ring → `cc`.
- This copy sets `ureq` to `default-features = false` with `rustls-no-provider`.
- The process crypto provider is [Lyamc/rustls-rustcrypto](https://github.com/Lyamc/rustls-rustcrypto) (RustCrypto only).
- The tokio/reqwest API is not enabled by the ureq feature.
- Unix file locking calls `flock` via `extern "C"` instead of the `libc` crate.

Use via workspace path dependency in the root `Cargo.toml`.
