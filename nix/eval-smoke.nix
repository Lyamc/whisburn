# Smoke-check: nix-instantiate --eval --strict nix/eval-smoke.nix
{ pkgs ? import <nixpkgs> { } }:
let
  inherit (pkgs) lib;
  eval = lib.evalModules {
    modules = [
      ./module.nix
      {
        _module.check = false;
        services.whisburn.enable = true;
        services.whisburn.package = pkgs.hello;
        services.whisburn.openFirewall = true;
        services.whisburn.model = "tiny_en";
        services.whisburn.preloadModels = [ "tiny_en" ];
        services.whisburn.ensureModels = [
          "parakeet-tdt-0.6b-v3"
          "medium"
        ];
      }
    ];
    specialArgs = { inherit pkgs; };
  };
in
{
  enable = eval.config.services.whisburn.enable;
  port = eval.config.services.whisburn.port;
  model = eval.config.services.whisburn.model;
  openFirewall = eval.config.services.whisburn.openFirewall;
  user = eval.config.services.whisburn.user;
  summaryEnable = eval.config.services.whisburn.summary.enable;
  summaryModel = eval.config.services.whisburn.summary.model;
  summaryDefault = eval.config.services.whisburn.summary.default;
  vramReserveMb = eval.config.services.whisburn.vramReserveMb;
  vramReservePercent = eval.config.services.whisburn.vramReservePercent;
}
