#!/bin/bash
#SBATCH --job-name=scm_e9_fixed_duration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-107%64
#SBATCH --output=logs/e9_%A_%a.out
#SBATCH --error=logs/e9_%A_%a.err

# E9 -- calendar-indexed, fixed-calendar-duration sweep (paper Sec.
# "Variance Scaling Table"): each task runs one replica of one
# k_target, stepping a FIXED CALENDAR grid (t_max held IDENTICAL across
# the whole sweep -- that shared value is what makes this a
# fixed-calendar-duration comparison at all; see
# experiments/run_e9_fixed_duration.py's module docstring for why this
# needs its own experiment rather than reusing E1's data).
#
# T_MAX below is a genuine free parameter, not a validated constant: it
# trades off against how much extra burnup the high-M end of the sweep
# accumulates relative to the low-M end (calendar time buys exposure at
# rate S0*M*dt). RECOMMENDATION: before submitting the full array, run
# one replica each at your highest and lowest k_target (edit --array to
# a single index, or just inspect the first few tasks' logs) and check
# M_end/N_end haven't done anything absurd before trusting T_MAX.
#
# Run submit_calibrate_alpha.sh first. `mkdir -p logs cache` from the
# repository root before your first submission (see common_env.sh).

mkdir -p logs
source "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}/slurm/common_env.sh"

K_TARGETS=(0.50 0.70 0.80 0.90 0.95 0.97 0.98 0.99 0.995)
N_REPLICAS=12
T_MAX=1.0e-3
N_KTARGETS=${#K_TARGETS[@]}

task=${SLURM_ARRAY_TASK_ID}
replica=$(( task % N_REPLICAS )); task=$(( task / N_REPLICAS ))
k_idx=$(( task % N_KTARGETS ))

K_TARGET=${K_TARGETS[$k_idx]}

python3 run_e9_fixed_duration.py \
  --case heu_sphere_nu --k-target "${K_TARGET}" \
  --alpha-lookup-json "${CACHE_ROOT}/alpha_lookup.json" \
  --replica "${replica}" \
  --n-steps 200 --t-max "${T_MAX}" \
  --cache-root "${CACHE_ROOT}" \
  --summary-csv "${CACHE_ROOT}/e9_fixed_duration_summary.csv"
