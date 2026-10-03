# Convenience entry: `nix-build nix` from the repo root (on a Nix host).
{ pkgs ? import <nixpkgs> { } }:
pkgs.callPackage ./package.nix { }
