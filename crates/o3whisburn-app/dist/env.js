// Shims for wasm imports in the "env" module (see Rust 1.96 wasm-ld changes).
// Only needed if a dependency emits undefined extern "C" symbols.
export function now() {
    return performance.now();
}