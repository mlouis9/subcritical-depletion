# SCM Depletion — reproducibility package

Code accompanying *Depletion in the Normalized Subcritical Multiplication
Method*. Implements the three depletion methods the paper compares (Method
C, exposure stepping; Method A, calendar stepping; Method B, ordinary
branching fixed-source with no population control and no rescaling) and
the four numerical experiments behind the paper's variance-scaling figure:

- **E1** — exposure-indexed master table (Method C): fixed exposure
  window, same `tau_max` across the whole $k$-sweep.
- **E9** — calendar-indexed, fixed-calendar-duration sweep (Method A):
  fixed calendar window, same `t_max` across the whole $k$-sweep. Exists
  as its own experiment (not a reinterpretation of E1's data) because a
  fixed-duration comparison needs every replica of every $k_\text{target}$
  landing on the *same* calendar date by construction — see
  `experiments/run_e9_fixed_duration.py`'s module docstring.
- **E1B**, **E9B** — Method B (branching fixed-source) analogs of E1/E9,
  for comparing variance scaling directly against the SCM results. E9B
  mirrors E9 exactly (same fixed `t_max`, native delivery); E1B
  *approximates* E1's exposure target via a per-$k_\text{target}$
  calendar window derived from the calibrated $k_\text{target}$ itself,
  since Method B has no native exposure coordinate to step in — see
  `experiments/run_e1b_master_table.py`'s module docstring for exactly
  what this approximates and how to check it. **Neither has been run or
  timed in this environment; pilot a single near-critical replica before
  submitting a full array** — Method B's cost grows with $M$ near
  criticality (no population control means each starting history is
  followed through its entire branching descendant tree), unlike SCM's
  fixed per-generation cost.

## Layout

```
scm_depletion/
  scm_depletion/                 the package
    scm_transport.py               SCMCoupledOperator, KFactors,
                                    isolated_transport_session
    matrix_utils.py                 exposure/calendar Bateman matrix
                                     assembly
    integrators.py                   ExposureIntegrator, CalendarIntegrator
    trajectory.py                     cache-keyed trajectory storage +
                                       the Method-C' interpolation
                                       algorithm (interpolate_at_time)
    caseregistry.py                    benchmark case table (heu_sphere_nu),
                                        nu-multiplier sweep, alpha calibration
    bench_models.py                     model builder for the sphere case
    nu_multiplier.py                     nu-Sigma_f rescaling (Remark on
                                          the approach-to-critical family)
    common.py                             cache keys, replica seeds,
                                           replica statistics
  experiments/
    run_calibrate_alpha.py         one-time: bisects the nu-multiplier
                                    achieving each k_target
    run_e1_master_table.py         E1 driver (one replica per invocation)
    run_e9_fixed_duration.py       E9 driver (one replica per invocation)
    run_e1b_master_table.py        E1B driver (Method B analog of E1)
    run_e9b_fixed_duration.py      E9B driver (Method B analog of E9)
    _expcommon.py                  shared CLI/setup boilerplate + the
                                    per-task summary file mechanism
    collect_summary.py             merge per-task summary files into one CSV
    extract_nuclide.py             pull one nuclide's inventory out of
                                    already-cached trajectories (no rerun)
    analysis_common.py             shared post-processing helpers
    make_fig_variance_scaling.py   produces the paper's variance-scaling
                                    figure
  slurm/
    common_env.sh                  shared environment setup (see below)
    submit_calibrate_alpha.sh
    submit_e1_master_table.sh
    submit_e9_fixed_duration.sh
    submit_e1b_master_table.sh
    submit_e9b_fixed_duration.sh
  openmc_capi_patch/                C-API extension this package depends
                                     on (see "OpenMC prerequisite" below)
```

## OpenMC prerequisite

Both integrators drive OpenMC's existing `SUBCRITICAL_MULTIPLICATION` run
mode (Forget's normalized source-iteration method) through the public
`openmc.deplete.CoupledOperator` API, with one addition: a small C-API
extension, `openmc_get_subcritical_k_factors`, exposing the per-cycle
$(\hat k,\hat k_q,\hat k_s)$ estimators and their standard errors to
Python. The patch is in `openmc_capi_patch/` (`capi.h.patch`,
`simulation.cpp.patch`, `core.py.patch`) — apply it to your OpenMC
checkout and rebuild before installing this package. See the paper's
Implementation section for what the extension does and why it reads
simulation state without an `initialized` guard.

## Setup

1. Apply `openmc_capi_patch/` to your OpenMC source tree and rebuild
   (`cmake --build .` / reinstall the Python package), so
   `openmc.lib.subcritical_k_factors()` is available.
2. `pip install -e .` this package from the repository root (or just
   put the repository root on `PYTHONPATH` — every script under
   `experiments/` inserts it automatically when run directly).
3. Set the two required environment variables before submitting
   anything (see `slurm/common_env.sh` for exactly what each is used
   for):
   ```bash
   export SCM_DEPLETION_VENV=/path/to/your/openmc-venv/bin/activate
   export OPENMC_CROSS_SECTIONS=/path/to/your/cross_sections.xml
   ```
4. From the repository root: `mkdir -p logs cache` (some Slurm
   configurations open the `--output`/`--error` log files before the
   job script body runs, so `logs/` must already exist at submission
   time).

## Reproducing the paper's figure

All `sbatch` commands below must be run from the repository root (the
submit scripts locate everything else via `SLURM_SUBMIT_DIR`, which
Slurm sets to the submission directory regardless of where it copies
the script to run it).

```bash
# 1. Calibrate the nu-multiplier for every k_target (once)
sbatch slurm/submit_calibrate_alpha.sh

# 2. Run E1 and E9 (can run concurrently; each is a 108-task array:
#    9 k_targets x 12 replicas)
sbatch slurm/submit_e1_master_table.sh
sbatch slurm/submit_e9_fixed_duration.sh

# 2b. Optional: Method B (branching fixed-source) comparison, E1B/E9B.
#     PILOT FIRST -- these have not been run or timed in this
#     environment, and Method B's cost grows with M near criticality
#     (no population control), unlike SCM's fixed per-generation cost.
#     Edit --array in both submit scripts to a single near-critical
#     index (e.g. k_target=0.995, one replica) and check the task
#     actually finishes in a sane time before submitting the full
#     108-task array:
sbatch slurm/submit_e1b_master_table.sh
sbatch slurm/submit_e9b_fixed_duration.sh

# 3. Merge each experiment's per-task summary files into one CSV
python experiments/collect_summary.py cache/e1_master_table_summary.csv
python experiments/collect_summary.py cache/e9_fixed_duration_summary.csv
python experiments/collect_summary.py cache/e1b_master_table_summary.csv   # if 2b was run
python experiments/collect_summary.py cache/e9b_fixed_duration_summary.csv # if 2b was run

# 4. Extract a single fission product's inventory from the cached
#    trajectories (the full-inventory sum is dominated by near-static
#    actinide mass and resolves no genuine replica-to-replica variance
#    for this case -- see extract_nuclide.py's own docstring)
python experiments/extract_nuclide.py \
  --summary-csv cache/e1_master_table_summary.csv --cache-root cache \
  --nuclide Xe135 --out-csv cache/e1_master_table_summary.xe135.csv
python experiments/extract_nuclide.py \
  --summary-csv cache/e9_fixed_duration_summary.csv --cache-root cache \
  --experiment e9_fixed_duration --nuclide Xe135 \
  --out-csv cache/e9_fixed_duration_summary.xe135.csv
# if 2b was run, same pattern with --experiment e1b_master_table /
# e9b_fixed_duration and the corresponding summary CSVs

# 5. Produce the figure (Method B overlay is optional -- add
#    --e1b-summary-csv/--e9b-summary-csv, both or either, once step 2b
#    has been run and extracted)
python experiments/make_fig_variance_scaling.py \
  --e1-summary-csv cache/e1_master_table_summary.xe135.csv \
  --e9-summary-csv cache/e9_fixed_duration_summary.xe135.csv \
  --cache-root cache --case heu_sphere_nu \
  --k-targets 0.50 0.70 0.80 0.90 0.95 0.97 0.98 0.99 0.995 \
  --n-steps 200 --nuclide Xe135 \
  --out fig_variance_scaling
```
Writes `fig_variance_scaling.pdf` (renders with real LaTeX if a working
`latex` is on `PATH` — checked with an actual test render, not just
presence on `PATH`, and falls back to matplotlib's built-in mathtext
otherwise) and `.png`.

## Things worth knowing before rerunning this on different hardware or parameters

- **E1B/E9B (Method B) have not been run or timed in this environment.**
  Branching fixed-source has no population control: each of
  `--n-particles` starting histories is followed through its entire
  branching descendant tree within one batch, and that tree's size
  grows with $M$ near criticality — where SCM's per-generation cost
  stays fixed regardless of $k$. Pilot a single near-critical replica
  (edit `--array` to one index) before submitting either full 108-task
  array, and adjust `--time`/the array's `%`-concurrency limit based on
  what you actually observe, not the placeholder values in the submit
  scripts. Separately, E1B's exposure target is an *approximation*, not
  an exact match to E1's own delivery: it derives a per-$k_\text{target}$
  calendar window from $M=1/(1-k_\text{target})$ rather than stepping in
  exposure directly (Method B has no native exposure coordinate — see
  `run_e1b_master_table.py`'s module docstring), so check the printed
  `tau_end` against `--tau-star` for a few replicas before trusting a
  full array's worth to be "at the same burnup" in the sense E1's own
  `tau_max` is. E9B has no such approximation — it mirrors E9 exactly.
- **`k_target=0.998` is excluded from both sweeps.** Every replica goes
  supercritical partway through the exposure window at this point,
  which makes the trajectory's calendar clock non-monotonic and breaks
  the interpolation both `run_e1_master_table.py`'s downstream analysis
  and `make_fig_variance_scaling.py`'s matched-burnup panel depend on.
  `analysis_common.safe_interpolate_at_time` detects and skips any
  individual replica this happens to (a transient excursion can affect
  a single replica even at lower $k_\text{target}$), but the full
  $k=0.998$ point was dropped from the sweep entirely rather than left
  in with most of its replicas silently missing.
- **`tau_max` (E1), `T_MAX` (E9), and `heu_sphere_nu`'s `source_strength`
  are derived together, from a burnup-per-step target, not guessed.**
  At this case's original source strength, the achievable power at
  even the highest-$M$ $k_\text{target}$ was ~26 $\mu$W — about $10^7$
  times below the power density standard depletion-methodology test
  cases use (Isotalo, OSTI 1362197) — so no fixed MWd/kgHM target could
  give a sane calendar duration without raising $S_0$ as well. Current
  values: $S_0=3.45\times10^{14}$ n/s, chosen so the highest-$M$
  $k_\text{target}$ ($k=0.995$, $M=200$) sits at exactly 0.1 MWd/kgHM
  per step (a standard conservative depletion step size — cf.
  Serpent's own documented 0.1–1.0 MWd/kgU early-life step convention)
  over a 1-day calendar step; `T_MAX`=200 days (E9); `tau_max` (E1) is
  independent of $S_0$ by construction (exposure directly counts
  multiplied neutrons) and converts to exactly the same 200-day
  duration at this $S_0$ and $M_\text{max}$ — a deliberate consistency
  check between the two experiments, not a coincidence. See the
  comments in both submit scripts and `caseregistry.py` for the full
  derivation. Changing $k_\text{target}$'s grid, the fuel geometry, or
  $S_0$ again all change what "0.1 MWd/kgHM at the highest $M$" works
  out to — recompute rather than reuse these numbers if any of those
  change.
- **Both experiments now use a fixed 20 inactive cycles out of 200
  generations per depletion step** (`n_inactive_schedule=lambda _: 20`
  passed to both integrators in `run_e1_master_table.py` and
  `run_e9_fixed_duration.py`). This was not always the case: earlier
  versions of this pipeline never set `openmc.lib.settings.inactive` at
  all, so every generation counted toward the multiplication estimators
  from the first cycle, with no source-convergence burn-in. If you're
  comparing against results generated before this was wired up, they
  are not directly comparable. No warm-start schedule is used (a
  `n_inactive_schedule` callable can implement one, shrinking the count
  once the source bank has settled between consecutive depletion steps
  — see `integrators.py`'s `_maybe_set_n_inactive` docstring — but this
  package uses a flat constant deliberately, not that mechanism).
- **The matched-burnup rows in the figure split each $k_\text{target}$'s
  replica pool in half** (a reference half used only to pick the
  calendar-time interpolation target, a measurement half actually
  contributing to the plotted variance) to avoid a self-referential
  bias that a single shared ensemble would introduce — see
  `make_fig_variance_scaling.py`'s `_series_matched_burnup` docstring.
  This roughly halves the effective replica count behind those two
  lines; more replicas per $k_\text{target}$ than the 12 used here will
  make them less noisy.
- **The log-log fits in the figure and in its console output are
  unweighted** — they do not use the $\sqrt{2/(R-1)}$ replica-variance
  uncertainty the error bars themselves show. Treat a fitted exponent
  as more trustworthy where the error bars are visibly small.
