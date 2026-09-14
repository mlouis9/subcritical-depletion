#!/bin/bash
#SBATCH --job-name=scm_calibrate_alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0
#SBATCH --output=logs/calibrate_%A_%a.out
#SBATCH --error=logs/calibrate_%A_%a.err

# One-time calibration pass: for the heu_sphere_nu case, bisect for the
# nu-multiplier alpha achieving every k_target used by E1/E9, and write
# the results to a shared JSON lookup table. Run this once, before
# submit_e1_master_table.sh or submit_e9_fixed_duration.sh -- both read
# alpha from ${CACHE_ROOT}/alpha_lookup.json via --alpha-lookup-json.
#
# `mkdir -p logs` must run before sbatch is even invoked on some Slurm
# configurations (the --output/--error paths are opened before the job
# script body runs): create it once, e.g. `mkdir -p logs cache`, from
# the repository root before your first submission.

mkdir -p logs
source "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}/slurm/common_env.sh"

python3 run_calibrate_alpha.py \
  --case heu_sphere_nu \
  --k-targets 0.50 0.70 0.80 0.90 0.95 0.97 0.98 0.99 0.995 \
  --out "${CACHE_ROOT}/alpha_lookup.json"
