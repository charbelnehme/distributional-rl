#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but was not found on PATH." >&2
  exit 1
fi

python3 -m venv --system-site-packages "${VENV_DIR}"

# Avoid upgrading pip here. In this workspace, pip self-upgrades inside a venv
# have been unreliable on the OneDrive-mounted filesystem.
"${VENV_DIR}/bin/python" -m ensurepip --upgrade >/dev/null
MISSING_PACKAGES="$("${VENV_DIR}/bin/python" - <<'PY'
import importlib.util

packages = {
    "alpaca-py": "alpaca",
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "ngboost": "ngboost",
    "optuna": "optuna",
    "pytest": "pytest",
}

missing = [package for package, module in packages.items() if importlib.util.find_spec(module) is None]
print(" ".join(missing))
PY
)"

if [[ -n "${MISSING_PACKAGES}" ]]; then
  "${VENV_DIR}/bin/python" -m pip install ${MISSING_PACKAGES}
fi

"${VENV_DIR}/bin/python" -m pip install -e . --no-deps --no-build-isolation

"${VENV_DIR}/bin/python" - <<'PY'
import src
from src import AlpacaMarketDataStore, DistributionalStrategy, build_feature_dataset

assert src.AlpacaMarketDataStore is AlpacaMarketDataStore
assert src.DistributionalStrategy is DistributionalStrategy
assert src.build_feature_dataset is build_feature_dataset
PY

cat <<EOF
Environment setup complete.
Activate it with:
  source .venv/bin/activate
EOF
