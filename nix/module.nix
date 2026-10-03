# NixOS module for the whisburn HTTP server and web UI.
#
# Flake:
#
#   {
#     inputs.whisburn.url = "github:Lyamc/whisburn";
#
#     outputs = { nixpkgs, whisburn, ... }: {
#       nixosConfigurations.host = nixpkgs.lib.nixosSystem {
#         modules = [
#           whisburn.nixosModules.default
#           {
#             services.whisburn.enable = true;
#             services.whisburn.openFirewall = true;
#           }
#         ];
#       };
#     };
#   }
#
# Non-flake NixOS configurations can import this file directly:
#
#   imports = [ /path/to/whisburn/nix/module.nix ];
#
#   services.whisburn.enable = true;
#
# `whisburn serve` is the API and the web UI. It binds TCP 8787 on all
# interfaces (`0.0.0.0`). The daemon does not open a browser (`--no-open`).
# Models are downloaded on demand into `services.whisburn.modelsDir`.
{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.services.whisburn;

  quoteSystemd = text: ''"${lib.replaceStrings [ "\\" ''"'' ] [ "\\\\" ''\"'' ] text}"'';

  preloadArg =
    if cfg.preloadModels == [ ] then
      [ ]
    else if cfg.preloadModels == [ "all" ] then
      [ "--preload-models=all" ]
    else
      [ "--preload-models=${lib.concatStringsSep "," cfg.preloadModels}" ];

  execArgs = [
    (lib.getExe cfg.package)
    "serve"
    "--no-open"
    "--port"
    (toString cfg.port)
    "--model"
    cfg.model
  ]
  ++ lib.optionals (cfg.device != null) [
    "--device"
    cfg.device
  ]
  ++ lib.optionals cfg.verbose [ "--verbose" ]
  ++ lib.optionals cfg.debug [ "--debug" ]
  ++ preloadArg
  ++ cfg.extraArgs;

  boolEnv = value: if value then "true" else "false";

  summarizeDefault =
    if cfg.summary.default == null then cfg.summary.enable else cfg.summary.default;

  effectiveEnsureModels = lib.unique (
    cfg.ensureModels ++ lib.optional cfg.summary.enable cfg.summary.model
  );
in
{
  options.services.whisburn = {
    enable = lib.mkEnableOption "whisburn speech processing HTTP server and web UI";

    package = lib.mkOption {
      type = lib.types.package;
      description = "Package that provides the `whisburn` binary. The default builds this repository.";
      default = pkgs.callPackage ./package.nix { };
      defaultText = lib.literalExpression "pkgs.callPackage ./package.nix { }";
    };

    user = lib.mkOption {
      type = lib.types.str;
      default = "whisburn";
      description = "Unix account the server runs as.";
    };

    group = lib.mkOption {
      type = lib.types.str;
      default = "whisburn";
      description = "Group of the service account. Used when {option}`services.whisburn.createUser` is true.";
    };

    createUser = lib.mkOption {
      type = lib.types.bool;
      default = cfg.user == "whisburn";
      defaultText = lib.literalExpression ''user == "whisburn"'';
      description = ''
        Create {option}`services.whisburn.user` and {option}`services.whisburn.group`.
        Turn this off to run as an account that already exists.
      '';
    };

    extraGroups = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [
        "video"
        "render"
      ];
      description = ''
        Extra groups for the service account when it is created by this module.
        `video` and `render` are required for WGPU / Vulkan GPU access.
      '';
    };

    dataDir = lib.mkOption {
      type = lib.types.str;
      default = "/var/lib/whisburn";
      description = ''
        Working directory and config home.
        `settings.toml` is read from here via `WHISBURN_CONFIG`.
      '';
    };

    modelsDir = lib.mkOption {
      type = lib.types.str;
      default = "${cfg.dataDir}/models";
      defaultText = lib.literalExpression ''"''${config.services.whisburn.dataDir}/models"'';
      description = "Directory for downloaded and converted model weights (`WHISBURN_MODELS_DIR`).";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 8787;
      description = "TCP port for `whisburn serve`. The process always binds `0.0.0.0`.";
    };

    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Open {option}`services.whisburn.port` for TCP.";
    };

    model = lib.mkOption {
      type = lib.types.str;
      default = "tiny_en";
      example = "qwen3-asr-0.6b";
      description = "Default model for `/v1/transcribe` (`--model` / `WHISBURN_DEFAULT_MODEL`).";
    };

    device = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "cpu";
      description = ''
        Burn device override (`--device` / `WHISBURN_DEVICE`).
        `null` lets Burn pick a WGPU adapter. Use `"cpu"` for the ndarray path.
      '';
    };

    preloadModels = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      example = [
        "tiny_en"
        "qwen3-asr-0.6b"
      ];
      description = ''
        Models to load into GPU memory at startup (`--preload-models`).
        Empty keeps lazy load on first audio. `[ "all" ]` warms every burn-ready model.
      '';
    };

    verbose = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Pass `--verbose` to `whisburn serve`.";
    };

    debug = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Pass `--debug` to `whisburn serve`.";
    };

    logLevel = lib.mkOption {
      type = lib.types.str;
      default = "info";
      example = "whisburn=debug";
      description = "Value of `RUST_LOG` for the server process.";
    };

    environmentFile = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "/etc/whisburn.env";
      description = ''
        Environment file sourced by systemd (for `HF_TOKEN` and other secrets).
        Do not put tokens in this Nix file.
      '';
    };

    extraArgs = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      description = "Extra arguments appended to `whisburn serve`.";
    };

    ensureModels = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      example = [
        "parakeet-tdt-0.6b-v3"
        "medium"
      ];
      description = ''
        Download and convert these models into {option}`services.whisburn.modelsDir`
        before the server starts. Already-converted models are skipped.
        Conversion uses the GPU and can take a long time on first boot.
        When {option}`services.whisburn.summary.enable` is true, `qwen3-0.6b`
        is added automatically.
      '';
    };

    vramReserveMb = lib.mkOption {
      type = lib.types.ints.unsigned;
      default = 1024;
      description = ''
        Keep at least this many MiB of GPU memory free so the display
        compositor can still allocate (`WHISBURN_VRAM_RESERVE_MB`).
        Combined with {option}`services.whisburn.vramReservePercent` using
        the larger of the two floors. `0` with percent `0` disables offload.
      '';
    };

    vramReservePercent = lib.mkOption {
      type = lib.types.ints.between 0 100;
      default = 15;
      description = ''
        Keep at least this percentage of total VRAM free
        (`WHISBURN_VRAM_RESERVE_PERCENT`). The effective reserve is
        `max(vramReserveMb, total * percent / 100)`.
      '';
    };

    summary = {
      enable = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = ''
          Download {option}`services.whisburn.summary.model` and allow
          **Summarize transcript** in the web UI. Set this to false to skip
          that download and grey out the toggle.
        '';
      };

      model = lib.mkOption {
        type = lib.types.str;
        default = "qwen3-0.6b";
        example = "granite-4.2-3b-fp8";
        description = ''
          Offline English summarizer (`WHISBURN_SUMMARIZER`).
          `qwen3-0.6b` is small enough for the GPU reserve.
          `granite-4.2-3b-fp8` is a 3B FP8 model (about 4 GB on disk) and is
          offloaded to CPU when it would cross the VRAM reserve.
        '';
      };

      default = lib.mkOption {
        type = lib.types.nullOr lib.types.bool;
        default = null;
        defaultText = lib.literalExpression "config.services.whisburn.summary.enable";
        example = false;
        description = ''
          Whether the UI checkbox is checked by default.
          `null` follows {option}`services.whisburn.summary.enable`.
          Ignored for interaction when `summary.enable` is false (the toggle
          is disabled).
        '';
      };
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.port != 0;
        message = "services.whisburn.port must be a fixed TCP port so the firewall can open it.";
      }
    ];

    environment.systemPackages = [ cfg.package ];

    networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [ cfg.port ];

    users.users = lib.mkIf cfg.createUser {
      ${cfg.user} = {
        isSystemUser = true;
        inherit (cfg) group extraGroups;
        home = cfg.dataDir;
        createHome = true;
        homeMode = "0700";
      };
    };

    users.groups = lib.mkIf cfg.createUser {
      ${cfg.group} = { };
    };

    systemd.tmpfiles.rules = [
      "d ${cfg.dataDir} 0700 ${cfg.user} ${cfg.group} - -"
      "d ${cfg.modelsDir} 0700 ${cfg.user} ${cfg.group} - -"
    ];

    systemd.services.whisburn-models = lib.mkIf (effectiveEnsureModels != [ ]) {
      description = "Download whisburn model weights";
      wantedBy = [ "multi-user.target" ];
      wants = [ "network-online.target" ];
      after = [ "network-online.target" ];
      before = [ "whisburn.service" ];
      requiredBy = [ "whisburn.service" ];

      environment = {
        HOME = cfg.dataDir;
        WHISBURN_CONFIG = cfg.dataDir;
        WHISBURN_MODELS_DIR = cfg.modelsDir;
        SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
        RUST_LOG = cfg.logLevel;
        WHISBURN_SUMMARIZE = boolEnv cfg.summary.enable;
        WHISBURN_SUMMARIZE_DEFAULT = boolEnv summarizeDefault;
        WHISBURN_SUMMARIZER = cfg.summary.model;
      }
      // lib.optionalAttrs (cfg.device != null) {
        WHISBURN_DEVICE = cfg.device;
      };

      path = [
        cfg.package
        pkgs.ffmpeg
      ];

      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        User = cfg.user;
        WorkingDirectory = cfg.dataDir;
        TimeoutStartSec = "2h";
        PrivateDevices = false;
        SupplementaryGroups = cfg.extraGroups;
        ExecStart = pkgs.writeShellScript "whisburn-ensure-models" ''
          set -eu
          ${lib.concatMapStrings (name: ''
            echo "whisburn: ensuring model ${name}"
            ${lib.getExe cfg.package} models download ${lib.escapeShellArg name} --verbose
          '') effectiveEnsureModels}
        '';
      }
      // lib.optionalAttrs cfg.createUser { Group = cfg.group; }
      // lib.optionalAttrs (cfg.environmentFile != null) {
        EnvironmentFile = cfg.environmentFile;
      };
    };

    systemd.services.whisburn = {
      description = "whisburn speech processing HTTP server";
      wantedBy = [ "multi-user.target" ];
      wants = [ "network-online.target" ];
      after = [ "network-online.target" ]
        ++ lib.optional (effectiveEnsureModels != [ ]) "whisburn-models.service";

      environment = {
        HOME = cfg.dataDir;
        WHISBURN_CONFIG = cfg.dataDir;
        WHISBURN_MODELS_DIR = cfg.modelsDir;
        WHISBURN_DEFAULT_MODEL = cfg.model;
        WHISBURN_PORT = toString cfg.port;
        SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
        RUST_LOG = cfg.logLevel;
        WHISBURN_SUMMARIZE = boolEnv cfg.summary.enable;
        WHISBURN_SUMMARIZE_DEFAULT = boolEnv summarizeDefault;
        WHISBURN_SUMMARIZER = cfg.summary.model;
        WHISBURN_VRAM_RESERVE_MB = toString cfg.vramReserveMb;
        WHISBURN_VRAM_RESERVE_PERCENT = toString cfg.vramReservePercent;
      }
      // lib.optionalAttrs (cfg.device != null) {
        WHISBURN_DEVICE = cfg.device;
      };

      # ffmpeg is already wrapped into the binary; keep it on PATH for subprocesses.
      # nvidia-smi lives under /run/opengl-driver on NixOS NVIDIA hosts and is used
      # to read remaining VRAM so the server can offload to CPU.
      path = [
        pkgs.ffmpeg
        "/run/opengl-driver"
      ];

      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        ExecStart = lib.concatMapStringsSep " " quoteSystemd execArgs;
        WorkingDirectory = cfg.dataDir;
        Restart = "on-failure";
        RestartSec = 5;
        # GPU device nodes under /dev/dri and /dev/nvidia*.
        PrivateDevices = false;
        SupplementaryGroups = cfg.extraGroups;
      }
      // lib.optionalAttrs cfg.createUser { Group = cfg.group; }
      // lib.optionalAttrs (cfg.environmentFile != null) {
        EnvironmentFile = cfg.environmentFile;
      };
    };
  };
}
