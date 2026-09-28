#!/bin/bash
#SBATCH --job-name=scm_e1b_master_table
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --array=0-167%16
#SBATCH --wckey=ne_gen
#SBATCH --output=logs/e1b_%A_%a.out
#SBATCH --error=logs/e1b_%A_%a.err

# E1B -- Method B (branching fixed-source) analog of E1, for a direct
# variance-scaling comparison against it. SAME TAU_STAR as E1's own
# tau_max (below, deliberately -- see
# experiments/run_e1b_master_table.py's module docstring for how this
# converts to a per-k_target calendar window, and why the resulting
# tau_end is only approximately tau_star, not exact the way E1's own
# delivery is).
#
# K_TARGETS widened from 9 to 14 points (denser sampling approaching
# criticality) so the fixed-source amplification trend is resolved by
# more than the previous handful of points near k=1. N_REPLICAS is left
# at the original 12 (not doubled the way E1/E9 were): Method B's
# per-task cost already grows with M near criticality, so more k-points
# was the requested lever here, not more per-point replica statistics.
#
# --time=48:00:00 (up from 24h, since there are now 14 k_targets instead
# of 9) and %16 (down from %32, to keep the near-critical high-M tasks
# from all landing in the same wall-clock window) are STILL STARTING
# GUESSES, not validated -- branching fixed-source's cost grows with M
# near criticality, unlike SCM's fixed per-generation cost. This array
# has NOT been run or timed in this environment. PILOT ONE near-critical
# task before submitting the full array.
#
# Run submit_calibrate_alpha.sh first. `mkdir -p logs cache` from the
# repository root before your first submission (see common_env.sh).

mkdir -p logs
source "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}/slurm/common_env.sh"

K_TARGETS=(0.50 0.60 0.70 0.80 0.85 0.90 0.93 0.95 0.96 0.97 0.98 0.985 0.99 0.995)
N_REPLICAS=12
TAU_STAR=1.1929e24
N_KTARGETS=${#K_TARGETS[@]}

task=${SLURM_ARRAY_TASK_ID}
replica=$(( task % N_REPLICAS )); task=$(( task / N_REPLICAS ))
k_idx=$(( task % N_KTARGETS ))

K_TARGET=${K_TARGETS[$k_idx]}

python3 run_e1b_master_table.py \
  --case heu_sphere_nu --k-target "${K_TARGET}" \
  --alpha-lookup-json "${CACHE_ROOT}/alpha_lookup.json" \
  --replica "${replica}" \
  --n-steps 200 --tau-star "${TAU_STAR}" \
  --cache-root "${CACHE_ROOT}" \
  --summary-csv "${CACHE_ROOT}/e1b_master_table_summary.csv"
