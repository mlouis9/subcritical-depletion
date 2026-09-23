#!/bin/bash
#SBATCH --job-name=scm_e9_fixed_duration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --array=0-215%64
#SBATCH --output=logs/e9_%A_%a.out
#SBATCH --error=logs/e9_%A_%a.err
#SBATCH --wckey=ne_gen

# E9 -- calendar-indexed, fixed-calendar-duration sweep (paper Sec.
# "Variance Scaling Table"): each task runs one replica of one
# k_target, stepping a FIXED CALENDAR grid (t_max held IDENTICAL across
# the whole sweep -- that shared value is what makes this a
# fixed-calendar-duration comparison at all; see
# experiments/run_e9_fixed_duration.py's module docstring for why this
# needs its own experiment rather than reusing E1's data).
#
# T_MAX = 1.728e7 s (200 days), derived (not guessed) from a
# burnup-per-step target at the WORST case in the sweep: unlike E1's
# exposure grid, E9's calendar grid is uniform in TIME, so burnup-per-
# step scales with M and is largest at the highest-M k_target
# (k=0.995, M=200). Chosen so that k_target's own burnup-per-step is at
# most 0.1 MWd/kgHM (a standard, conservative depletion step size --
# cf. Serpent's own documented 0.1-1.0 MWd/kgU early-life step
# convention) over a 1-day calendar step, 200 steps total. This pins
# down heu_sphere_nu's source_strength too (see caseregistry.py,
# 3.45e14 n/s, not the earlier 1.0e7): the previous S0 gave a power of
# only ~26 uW at this same k_target, ~1.4e7x below the ~40 kW/kgHM
# power density used in standard depletion-methodology test cases
# (Isotalo, OSTI 1362197) -- meaning no fixed MWd/kg target could have
# given a sane calendar duration without raising S0 as well. At the new
# S0, per-step burnup ranges from 1.0e-3 MWd/kgHM at k=0.50 up to the
# 0.1 MWd/kgHM ceiling at k=0.995 -- all within normal reactor-adjacent
# power density (1-100 kW/kgHM across the sweep).
#
# Run submit_calibrate_alpha.sh first. `mkdir -p logs cache` from the
# repository root before your first submission (see common_env.sh).
#
# Replicas doubled 12 -> 24 and --n-particles doubled 8000 -> 16000
# below, matching submit_e1_master_table.sh, for the same reasons (the
# noise-floor decomposition in diagnose_R_variance.py); --time roughly
# doubled to match. Array size = N_KTARGETS * N_REPLICAS = 9 * 24 = 216.

mkdir -p logs
source "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}/slurm/common_env.sh"

K_TARGETS=(0.50 0.70 0.80 0.90 0.95 0.97 0.98 0.99 0.995)
N_REPLICAS=24
N_PARTICLES=16000
N_GENERATIONS=200
T_MAX=1.728e7
N_KTARGETS=${#K_TARGETS[@]}

task=${SLURM_ARRAY_TASK_ID}
replica=$(( task % N_REPLICAS )); task=$(( task / N_REPLICAS ))
k_idx=$(( task % N_KTARGETS ))

K_TARGET=${K_TARGETS[$k_idx]}

python3 run_e9_fixed_duration.py \
  --case heu_sphere_nu --k-target "${K_TARGET}" \
  --alpha-lookup-json "${CACHE_ROOT}/alpha_lookup.json" \
  --replica "${replica}" \
  --n-particles "${N_PARTICLES}" --n-generations "${N_GENERATIONS}" \
  --n-steps 200 --t-max "${T_MAX}" \
  --cache-root "${CACHE_ROOT}" \
  --summary-csv "${CACHE_ROOT}/e9_fixed_duration_summary.csv"
