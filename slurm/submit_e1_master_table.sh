#!/bin/bash
#SBATCH --job-name=scm_e1_master_table
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --array=0-215%64
#SBATCH --output=logs/e1_%A_%a.out
#SBATCH --error=logs/e1_%A_%a.err
#SBATCH --wckey=ne_gen

# E1 -- exposure-indexed master table (paper Sec. "Variance Scaling
# Table"): each task runs one replica of one k_target, stepping a FIXED
# EXPOSURE grid (tau_max held identical across the whole sweep). Array
# size = N_KTARGETS * N_REPLICAS = 9 * 24 = 216, so --array=0-215 above
# (replicas doubled from 12 -> 24 to tighten the sqrt(2/(R-1)) relative
# uncertainty on every variance point; see diagnose_R_variance.py).
# k_target=0.998 is deliberately excluded: it goes supercritical partway
# through this exposure window for every replica tried (see the
# project's own diagnosis of this), which breaks the strictly-increasing
# clock interpolate_at_time depends on.
#
# --n-particles doubled 8000 -> 16000 below: this is the direct lever on
# the per-step k-estimator noise floor identified by
# diagnose_R_variance.py as the dominant contribution to the measured
# R(t*) variance (it should shrink close to linearly in n_particles).
# --time is roughly doubled from the original 08:00:00 to cover the
# doubled particle count; re-check against actual observed wall time for
# your cluster/geometry once the first few tasks land, same as any other
# capacity planning here.
#
# Run submit_calibrate_alpha.sh first. `mkdir -p logs cache` from the
# repository root before your first submission (see common_env.sh).

mkdir -p logs
source "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}/slurm/common_env.sh"

K_TARGETS=(0.50 0.70 0.80 0.90 0.95 0.97 0.98 0.99 0.995)
N_REPLICAS=24
N_PARTICLES=16000
N_GENERATIONS=200
N_KTARGETS=${#K_TARGETS[@]}

task=${SLURM_ARRAY_TASK_ID}
replica=$(( task % N_REPLICAS )); task=$(( task / N_REPLICAS ))
k_idx=$(( task % N_KTARGETS ))

K_TARGET=${K_TARGETS[$k_idx]}

# tau_max = 1.1929e24 n, derived (not guessed) from a burnup-per-step
# target: this experiment's exposure grid is uniform across the whole
# k-sweep, so the resulting burnup-per-step is IDENTICAL at every
# k_target by construction (tau counts multiplied neutrons directly,
# independent of S0) -- 0.1 MWd/kgHM per step over 200 steps, on this
# case's 8.848 kgHM fuel mass, with kappa=200 MeV/fission and nu=2.5
# (fast-spectrum U235). This tau_max, converted to calendar time at
# heu_sphere_nu's source_strength (see caseregistry.py) and M=200 (the
# highest-M k_target), corresponds to exactly the same 200-day campaign
# E9 uses below -- a deliberate consistency check between the two
# experiments' independently-derived durations, not a coincidence.
python3 run_e1_master_table.py \
  --case heu_sphere_nu --k-target "${K_TARGET}" \
  --alpha-lookup-json "${CACHE_ROOT}/alpha_lookup.json" \
  --replica "${replica}" \
  --n-particles "${N_PARTICLES}" --n-generations "${N_GENERATIONS}" \
  --n-steps 200 --tau-max 1.1929e24 \
  --cache-root "${CACHE_ROOT}" \
  --summary-csv "${CACHE_ROOT}/e1_master_table_summary.csv"
