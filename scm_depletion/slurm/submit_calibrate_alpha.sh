#!/bin/bash
#SBATCH --job-name=scm_calibrate_alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0
#SBATCH --wckey=ne_gen
#SBATCH --output=logs/calibrate_%A_%a.out
#SBATCH --error=logs/calibrate_%A_%a.err

# One-time calibration pass: for the heu_sphere_nu case, bisect for the
# nu-multiplier alpha achieving every k_target used anywhere downstream,
# and write the results to a shared JSON lookup table. Run this once,
# before any of the submit_e1*/submit_e9* scripts -- they all read alpha
# from ${CACHE_ROOT}/alpha_lookup.json via --alpha-lookup-json.
#
# The list below is the UNION of E1/E9's 9-point k_target grid and
# E1B/E9B's widened 14-point grid (submit_e1b_master_table.sh,
# submit_e9b_fixed_duration.sh) -- calibrate once for every k_target any
# downstream script will ever request, rather than maintaining two lookup
# tables.
#
# `mkdir -p logs` must run before sbatch is even invoked on some Slurm
# configurations (the --output/--error paths are opened before the job
# script body runs): create it once, e.g. `mkdir -p logs cache`, from
# the repository root before your first submission.

mkdir -p logs
source "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}/slurm/common_env.sh"

python3 run_calibrate_alpha.py \
  --case heu_sphere_nu \
  --k-targets 0.50 0.60 0.70 0.80 0.85 0.90 0.93 0.95 0.96 0.97 0.98 0.985 0.99 0.995 \
  --out "${CACHE_ROOT}/alpha_lookup.json"
