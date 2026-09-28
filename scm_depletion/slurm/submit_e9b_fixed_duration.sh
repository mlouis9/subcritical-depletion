#!/bin/bash
#SBATCH --job-name=scm_e9b_fixed_duration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --array=0-167%16
#SBATCH --wckey=ne_gen
#SBATCH --output=logs/e9b_%A_%a.out
#SBATCH --error=logs/e9b_%A_%a.err

# E9B -- Method B (branching fixed-source) analog of E9, for a direct
# variance-scaling comparison against it. SAME T_MAX as E9 (below,
# deliberately), same k_target grid (now widened, see below), same
# replica count -- everything structural is identical; only the
# transport method differs (see
# experiments/run_e9b_fixed_duration.py's module docstring).
#
# K_TARGETS widened from 9 to 14 points, matching submit_e1b_master_table.sh,
# so the fixed-source amplification trend is resolved by more than a
# handful of points near k=1. N_REPLICAS stays at 12: more k-points was
# the requested lever here, not more per-point replica statistics (see
# submit_e1b_master_table.sh's matching comment).
#
# --time=48:00:00 (up from 24h, since there are now 14 k_targets instead
# of 9) and %16 (down from %32) are STILL STARTING GUESSES, not
# validated: branching fixed-source has no population control, so each
# of --n-particles starting histories is followed through its ENTIRE
# descendant tree, which grows with M near criticality -- unlike SCM,
# whose per-generation cost stays fixed. This array has NOT been run or
# timed in this environment. PILOT ONE near-critical task (edit --array
# to a single high-k_target index) before submitting the full array, and
# adjust --time and the %-limit (fewer concurrent tasks if each one is
# now much more expensive) based on what you actually observe.
#
# Run submit_calibrate_alpha.sh first. `mkdir -p logs cache` from the
# repository root before your first submission (see common_env.sh).

mkdir -p logs
source "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}/slurm/common_env.sh"

K_TARGETS=(0.50 0.60 0.70 0.80 0.85 0.90 0.93 0.95 0.96 0.97 0.98 0.985 0.99 0.995)
N_REPLICAS=12
T_MAX=1.728e7
N_KTARGETS=${#K_TARGETS[@]}

task=${SLURM_ARRAY_TASK_ID}
replica=$(( task % N_REPLICAS )); task=$(( task / N_REPLICAS ))
k_idx=$(( task % N_KTARGETS ))

K_TARGET=${K_TARGETS[$k_idx]}

python3 run_e9b_fixed_duration.py \
  --case heu_sphere_nu --k-target "${K_TARGET}" \
  --alpha-lookup-json "${CACHE_ROOT}/alpha_lookup.json" \
  --replica "${replica}" \
  --n-steps 200 --t-max "${T_MAX}" \
  --cache-root "${CACHE_ROOT}" \
  --summary-csv "${CACHE_ROOT}/e9b_fixed_duration_summary.csv"
