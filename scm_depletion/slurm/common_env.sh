# Shared environment setup for the SCM depletion SLURM scripts.
# Sourced (not executed) at the top of every submit_*.sh script.
#
# Configure the two required variables below for your own cluster
# before submitting anything -- export them in your shell (or add
# them to your job submission) rather than editing this file, so a
# fresh checkout of this repository needs no local modification at all.
#
#   export SCM_DEPLETION_VENV=/path/to/your/openmc-venv/bin/activate
#   export OPENMC_CROSS_SECTIONS=/path/to/your/cross_sections.xml
#
# The repository root is taken from SLURM_SUBMIT_DIR (the directory
# `sbatch` was invoked from), which Slurm sets reliably for every job
# -- including array jobs -- independent of where the script itself
# ends up (sbatch copies submitted scripts into a per-job spool
# directory before executing them, so `dirname "$0"` would NOT give a
# stable answer here; SLURM_SUBMIT_DIR does). Submit from the
# repository root:
#
#   cd /path/to/scm_depletion && sbatch slurm/submit_e1_master_table.sh

set -euo pipefail

: "${SCM_DEPLETION_VENV:?Set SCM_DEPLETION_VENV to your Python venv activate script before submitting (see slurm/common_env.sh).}"
: "${OPENMC_CROSS_SECTIONS:?Set OPENMC_CROSS_SECTIONS to your cross_sections.xml before submitting (see slurm/common_env.sh).}"

module load python 2>/dev/null || true
source "${SCM_DEPLETION_VENV}"

if [[ ! -f "${OPENMC_CROSS_SECTIONS}" ]]; then
  echo "OPENMC_CROSS_SECTIONS does not exist: ${OPENMC_CROSS_SECTIONS}" >&2
  exit 1
fi
export OPENMC_CROSS_SECTIONS

export WORKDIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unset -- submit with sbatch from the repository root, not by running this script directly.}"
export CACHE_ROOT="${WORKDIR}/cache"
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"

cd "${WORKDIR}/experiments"

# One replica per task; OMP threads per replica (no MPI in this
# project's convention -- replicas are embarrassingly parallel across
# SLURM array tasks, and each task uses OMP_NUM_THREADS worker threads
# within OpenMC).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

mkdir -p "${CACHE_ROOT}" "${WORKDIR}/logs"
