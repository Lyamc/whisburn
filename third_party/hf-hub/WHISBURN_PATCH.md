# Vendored hf-hub (0.4.3)

Patched for whisburn so the default build does not pull the `cc` crate:

- Upstream `ureq` dependency used default features, which enable `tls` ? rustls ? ring ? cc.
- This copy sets `ureq` to `default-features = false` with `native-tls` only.

Use via workspace path dependency in the root `Cargo.toml`.
