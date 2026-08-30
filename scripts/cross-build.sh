#!/usr/bin/env bash
# Cross build helper for o3whisburn CLI (CPU friendly)
# Examples:
#   ./scripts/cross-build.sh x86_64-unknown-linux-musl
#   ./scripts/cross-build.sh aarch64-unknown-linux-gnu
set -euo pipefail
TARGET=${1:-x86_64-unknown-linux-musl}
PROFILE=${PROFILE:-release}

echo "==> Adding target $TARGET"
rustup target add "$TARGET" 2>/dev/null || true

echo "==> Building o3whisburn-cli for $TARGET ($PROFILE)"
cargo build -p o3whisburn-cli --target "$TARGET" --profile "$PROFILE"

BIN="target/$TARGET/$PROFILE/o3whisburn"
if [[ -f "$BIN" ]]; then
  echo "==> Built: $BIN"
  file "$BIN" || true
else
  echo "Binary may be at: target/$TARGET/debug/o3whisburn or release"
  ls -l "target/$TARGET/"*/* 2>/dev/null | cat || true
fi

echo "To run on target (CPU mode recommended for foreign arch):"
echo "  O3WHISBURN_DEVICE=cpu ./o3whisburn --help"
