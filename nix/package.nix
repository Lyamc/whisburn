# Build the `whisburn` CLI from this repo.
#
#   pkgs.callPackage /path/to/whisburn/nix/package.nix { }
#
# The HTTP server is `whisburn serve`. The Iced desktop client (`whisburn app`)
# is off by default so a NixOS rebuild does not pull winit/X11. Set
# `withApp = true` if you want `whisburn app` from the store path.
{
  lib,
  rustPlatform,
  pkg-config,
  makeWrapper,
  alsa-lib,
  vulkan-loader,
  wayland,
  libxkbcommon,
  libGL,
  ffmpeg,
  withApp ? false,
}:

rustPlatform.buildRustPackage {
  pname = "whisburn";
  version = "0.2.0";

  src = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ../Cargo.toml
      ../Cargo.lock
      ../.cargo
      ../crates
      ../third_party
      # Embedded in whisburn-server (`include_bytes!`).
      ../samples/jfk.wav
    ];
  };

  cargoLock = {
    lockFile = ../Cargo.lock;
    outputHashes = {
      # git+https://github.com/Lyamc/npy-rs
      "npy-0.5.0" = "sha256-urpWDhAt4BEZ3/KDlCrq5gt+qw+/iYeHadts/EXgDJM=";
      # git+https://github.com/Lyamc/rustls-rustcrypto
      "rustls-rustcrypto-0.0.2-alpha" = "sha256-adoon7zrd+rafJ9VHR4rtNq8rQBNvW/3hcD6AmxzP/g=";
    };
  };

  nativeBuildInputs = [
    pkg-config
    makeWrapper
  ];

  buildInputs = [
    alsa-lib
    vulkan-loader
    wayland
    libxkbcommon
    libGL
  ];

  cargoBuildFlags = [
    "-p"
    "whisburn-cli"
  ]
  ++ lib.optionals withApp [ "--features" "app" ];

  buildNoDefaultFeatures = !withApp;

  # Integration tests need downloaded model weights and samples.
  doCheck = false;

  postInstall = ''
    wrapProgram "$out/bin/whisburn" \
      --prefix PATH : ${lib.makeBinPath [ ffmpeg ]} \
      --prefix LD_LIBRARY_PATH : ${
        lib.makeLibraryPath [
          vulkan-loader
          libGL
          wayland
          libxkbcommon
          alsa-lib
        ]
      }
  '';

  meta = {
    description = "Burn-based speech processing: transcription, translation, diarization, and TTS";
    homepage = "https://github.com/Lyamc/whisburn";
    license = lib.licenses.mit;
    mainProgram = "whisburn";
    platforms = lib.platforms.linux;
  };
}
