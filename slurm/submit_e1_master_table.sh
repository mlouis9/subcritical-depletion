#!/bin/bash
#SBATCH --job-name=scm_e1_master_table
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-107%64
#SBATCH --output=logs/e1_%A_%a.out
#SBATCH --error=logs/e1_%A_%a.err

# E1 -- exposure-indexed master table (paper Sec. "Variance Scaling
# Table"): each task runs one replica of one k_target, stepping a FIXED
# EXPOSURE grid (tau_max held identical across the whole sweep). Array
# size = N_KTARGETS * N_REPLICAS = 9 * 12 = 108, so --array=0-107 above.
# k_target=0.998 is deliberately excluded: it goes supercritical partway
# through this exposure window for every replica tried (see the
# project's own diagnosis of this), which breaks the strictly-increasing
# clock interpolate_at_time depends on.
#
# Run submit_calibrate_alpha.sh first. `mkdir -p logs cache` from the
# repository root before your first submission (see common_env.sh).

mkdir -p logs
source "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}/slurm/common_env.sh"

K_TARGETS=(0.50 0.70 0.80 0.90 0.95 0.97 0.98 0.99 0.995)
N_REPLICAS=12
N_KTARGETS=${#K_TARGETS[@]}

task=${SLURM_ARRAY_TASK_ID}
replica=$(( task % N_REPLICAS )); task=$(( task / N_REPLICAS ))
k_idx=$(( task % N_KTARGETS ))

K_TARGET=${K_TARGETS[$k_idx]}

python3 run_e1_master_table.py \
  --case heu_sphere_nu --k-target "${K_TARGET}" \
  --alpha-lookup-json "${CACHE_ROOT}/alpha_lookup.json" \
  --replica "${replica}" \
  --n-steps 200 --tau-max 3.0e5 \
  --cache-root "${CACHE_ROOT}" \
  --summary-csv "${CACHE_ROOT}/e1_master_table_summary.csv"
