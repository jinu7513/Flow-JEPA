#!/bin/bash
# Supply --chdir, absolute output/error paths and CPU resources at submission.
set -euo pipefail
: "${SLURM_JOB_ID:?Use sbatch}" "${FLOWJEPA_REPO:?}"
: "${FLOWJEPA_PYTHON:?}" "${FLOWJEPA_DATA_ROOT:?}"
[[ "$FLOWJEPA_REPO" = /* && "$FLOWJEPA_PYTHON" = /* && "$FLOWJEPA_DATA_ROOT" = /* ]]
cd "$FLOWJEPA_REPO"
"$FLOWJEPA_PYTHON" -u "$FLOWJEPA_REPO/scripts/download_pusht.py" --storage "$FLOWJEPA_DATA_ROOT"
