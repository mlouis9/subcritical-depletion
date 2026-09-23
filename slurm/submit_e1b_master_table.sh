#!/bin/bash
#SBATCH --job-name=scm_e1b_master_table
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --array=0-107%16
#SBATCH --output=logs/e1b_%A_%a.out
#SBATCH --error=logs/e1b_%A_%a.err
#SBATCH --wckey=ne_gen

# E1B -- Method B (branching fixed-source) analog of E1, for a direct
# variance-scaling comparison against it. SAME TAU_STAR as E1's own
# tau_max (below, deliberately -- see
# experiments/run_e1b_master_table.py's module docstring for how this
# converts to a per-k_target calendar window, and why the resulting
# tau_end is only approximately tau_star, not exact the way E1's own
# delivery is).
#
# --time=48:00:00 and %16 are STILL STARTING GUESSES, revised upward
# from the original 24:00:00/%32 now that the full 9-point k_target grid
# below is meant to actually be submitted (only the cheap, low-M end of
# it -- k up to ~0.80 -- has been run and timed here so far): branching
# fixed-source's cost grows with M near criticality, unlike SCM's fixed
# per-generation cost, so the untested k=0.99/0.995 tasks could easily
# still exceed 48h. PILOT ONE near-critical task (edit --array to a
# single high-k_target index, e.g. the last one) before submitting the
# full array, and revise --time/the %-limit from what you actually
# observe rather than trusting this guess.
#
# Run submit_calibrate_alpha.sh first. `mkdir -p logs cache` from the
# repository root before your first submission (see common_env.sh).

mkdir -p logs
source "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}/slurm/common_env.sh"

K_TARGETS=(0.50 0.70 0.80 0.90 0.95 0.97 0.98 0.99 0.995)
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
