{
  pkgs ? import <nixpkgs> { },
}:
let
  python = pkgs.python314;
  pythonPackages = python.pkgs;
in
pkgs.mkShell {

  shellHook = ''
    cat > pyproject.toml << 'EOF'
    [tool.ruff]
    target-version = "py313"

    [tool.ruff.lint]
    select = ["E", "F", "I"]

    [tool.marimo.language_servers.ty]
    enabled = true
    path = "${pkgs.ty}/bin/ty"

    [tool.marimo.language_servers.ruff]
    enabled = true
    path = "${pkgs.ruff}/bin/ruff"
    EOF
  '';
  packages = [
    pkgs.csv-tui
    pkgs.kaggle
    pythonPackages.kaggle
    pythonPackages.python-lsp-server
    # pythonPackages.python-lsp-ruff
    pythonPackages.marimo
    pythonPackages.pydantic-ai-slim
    pythonPackages.openai
    pythonPackages.optuna
    pythonPackages.numpy
    pythonPackages.scipy
    pythonPackages.pyarrow
    pythonPackages.scikit-learn
    pythonPackages.pandas
    pythonPackages.polars
    pythonPackages.pip
    pythonPackages.setuptools
    pythonPackages.matplotlib
    pythonPackages.seaborn
    pythonPackages.xgboost
    pythonPackages.kagglehub
    pkgs.ty
    pkgs.pyright
    pkgs.ruff
  ];
  # Required for pygame on some systems
}
