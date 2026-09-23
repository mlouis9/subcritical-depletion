#!/bin/bash
#SBATCH --job-name=scm_e9b_fixed_duration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --array=0-107%16
#SBATCH --output=logs/e9b_%A_%a.out
#SBATCH --error=logs/e9b_%A_%a.err
#SBATCH --wckey=ne_gen

# E9B -- Method B (branching fixed-source) analog of E9, for a direct
# variance-scaling comparison against it. SAME T_MAX as E9 (below,
# deliberately), same k_target grid, same replica count -- everything
# structural is identical; only the transport method differs (see
# experiments/run_e9b_fixed_duration.py's module docstring).
#
# --time=48:00:00 (vs E9's 16h) and %16 (vs E9's %64) are STILL STARTING
# GUESSES, revised upward from the original 24:00:00/%32 now that the
# full 9-point k_target grid below is meant to actually be submitted
# (only the cheap, low-M end -- k up to ~0.80 -- has been run and timed
# here so far): branching fixed-source has no population control, so
# each of --n-particles starting histories is followed through its
# ENTIRE descendant tree, which grows with M near criticality -- unlike
# SCM, whose per-generation cost stays fixed. PILOT ONE near-critical
# task (edit --array to a single high-k_target index) before submitting
# the full array, and adjust --time and the %-limit (fewer concurrent
# tasks if each one is now much more expensive) based on what you
# actually observe rather than trusting this guess.
#
# Run submit_calibrate_alpha.sh first. `mkdir -p logs cache` from the
# repository root before your first submission (see common_env.sh).

mkdir -p logs
source "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}/slurm/common_env.sh"

K_TARGETS=(0.50 0.70 0.80 0.90 0.95 0.97 0.98 0.99 0.995)
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
