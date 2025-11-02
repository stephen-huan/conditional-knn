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
        inherit (maipkgs.legacyPackages.${system}) python python3Packages;
        python' = python.withPackages (ps: with ps; [
          cython
          numpy
          pbbfmm3d
          pygments
          scikit-learn
          scipy
          seaborn
          setuptools
        ]);
        formatters = [
          python3Packages.black
          python3Packages.isort
          pkgs.nixpkgs-fmt
        ];
        linters = [ pkgs.pyright python3Packages.ruff pkgs.statix ];
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
            maipkgs.packages.${system}.hlibpro
          ]
          ++ formatters
          ++ linters;
        };
      }
    );
}
