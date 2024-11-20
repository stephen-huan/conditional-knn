{
  description = "Conditional k-nearest neighbor estimator";

  inputs = {
    nixpkgs.follows = "maipkgs/nixpkgs";
    maipkgs.url = "github:stephen-huan/maipkgs";
  };

  outputs = { self, nixpkgs, maipkgs }:
    let
      inherit (nixpkgs) lib;
      systems = lib.systems.flakeExposed;
      eachDefaultSystem = f: builtins.foldl' lib.attrsets.recursiveUpdate { }
        (map f systems);
    in
    eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        inherit (maipkgs.legacyPackages.${system}) python;
        pythonPackages = pkgs.python311Packages;
        python' = python.withPackages (ps: with ps; [
          cython
          pbbfmm3d
          pygments
          scikit-learn
          seaborn
          setuptools
        ]);
        formatters = [
          pythonPackages.black
          pythonPackages.isort
          pkgs.nixpkgs-fmt
        ];
        linters = [ pkgs.pyright pythonPackages.ruff pkgs.statix ];
      in
      {
        formatter.${system} = pkgs.writeShellApplication {
          name = "formatter";
          runtimeInputs = formatters;
          text = ''
            isort "$@"
            black "$@"
            nixpkgs-fmt "$@"
          '';
        };

        checks.${system}.lint = pkgs.stdenvNoCC.mkDerivation {
          name = "lint";
          src = ./.;
          doCheck = true;
          nativeCheckInputs = lib.singleton python' ++ formatters ++ linters;
          checkPhase = ''
            isort --check --diff .
            black --check --diff .
            nixpkgs-fmt --check .
            ruff check .
            pyright .
            statix check
          '';
          installPhase = "touch $out";
        };

        devShells.${system}.default = pkgs.mkShell {
          packages = [
            python'
            pkgs.mkl
          ]
          ++ formatters
          ++ linters;
        };
      }
    );
}
