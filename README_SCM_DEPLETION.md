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
  for comparing variance scaling directly against the SCM results, now
  swept over a denser 14-point $k_\text{target}$ grid than E1/E9's 9
  points (see "Reproducing the paper's figure" below). E9B mirrors E9
  exactly (same fixed `t_max`, native delivery); E1B *approximates* E1's
  exposure target via a per-$k_\text{target}$ calendar window derived
  from the calibrated $k_\text{target}$ itself, since Method B has no
  native exposure coordinate to step in — see
  `experiments/run_e1b_master_table.py`'s module docstring for exactly
  what this approximates and how to check it. **Neither has been run or
  timed in this environment; pilot a single near-critical replica before
  submitting a full array** — Method B's cost grows with $M$ near
  criticality (no population control means each starting history is
  followed through its entire branching descendant tree), unlike SCM's
  fixed per-generation cost.

Post-processing (figures and diagnostics from already-cached trajectories)
is described in its own section below — **"Post-processing: the
ergonomic workflow"** — and is the recommended starting point once you
have summary CSVs in hand; the four experiment drivers above are what
produce those CSVs in the first place.

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
                                    figure (panels a/b); --skip-nscm gives
                                    a Method-B-only rendering
    make_fig_trajectory_bands_relative.py
                                    three-panel per-nuclide trajectory
                                    figure, plotted as fractional
                                    deviation from each k_target's own
                                    ensemble mean (avoids a linear-axis
                                    plotting artifact that hides replica
                                    spread for saturating fission products)
    diagnose_R_variance.py         read-only diagnostic: decomposes
                                    Var[R(t*)] into the clock-displacement
                                    mechanism the M^4/M^6 theory rows are
                                    about vs. the local per-step
                                    k-estimator noise floor, so a
                                    below-theory fitted exponent can be
                                    attributed to one or the other
                                    directly instead of by inspection
    run_nuclide_suite.py           ERGONOMIC ENTRY POINT: runs
                                    extract_nuclide.py ->
                                    make_fig_variance_scaling.py ->
                                    make_fig_trajectory_bands_relative.py
                                    -> diagnose_R_variance.py for as many
                                    nuclides as you like, in one command
                                    (see "Post-processing" below)
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
# 1. Calibrate the nu-multiplier for every k_target used anywhere
#    downstream (the union of E1/E9's 9-point grid and E1B/E9B's
#    14-point grid; once)
sbatch slurm/submit_calibrate_alpha.sh

# 2. Run E1 and E9 (can run concurrently; each is now a 216-task array:
#    9 k_targets x 24 replicas, at 16000 particles/gen and 40 inactive
#    cycles per step -- see "Run parameters" below for what changed and
#    why)
sbatch slurm/submit_e1_master_table.sh
sbatch slurm/submit_e9_fixed_duration.sh

# 2b. Optional: Method B (branching fixed-source) comparison, E1B/E9B,
#     now swept over 14 k_targets x 12 replicas = 168 tasks each.
#     PILOT FIRST -- these have not been run or timed in this
#     environment, and Method B's cost grows with M near criticality
#     (no population control), unlike SCM's fixed per-generation cost.
#     Edit --array in both submit scripts to a single near-critical
#     index (e.g. k_target=0.995, one replica) and check the task
#     actually finishes in a sane time before submitting the full
#     168-task array:
sbatch slurm/submit_e1b_master_table.sh
sbatch slurm/submit_e9b_fixed_duration.sh

# 3. Merge each experiment's per-task summary files into one CSV
python experiments/collect_summary.py cache/e1_master_table_summary.csv
python experiments/collect_summary.py cache/e9_fixed_duration_summary.csv
python experiments/collect_summary.py cache/e1b_master_table_summary.csv   # if 2b was run
python experiments/collect_summary.py cache/e9b_fixed_duration_summary.csv # if 2b was run

# 4. Post-process: use run_nuclide_suite.py (recommended, see below) or
#    the individual scripts it wraps.
```

## Post-processing: the ergonomic workflow

`run_nuclide_suite.py` is the recommended way to go from merged summary
CSVs to finished figures. It wraps the sequence `extract_nuclide.py` ->
`make_fig_variance_scaling.py` -> `make_fig_trajectory_bands_relative.py`
-> `diagnose_R_variance.py`, looped over as many nuclides as you like, and
shells out to each (rather than importing them) so one nuclide's failure
can't corrupt another's in-process state. Each independently-tested
script it calls is still fine to run standalone — see the following
subsections for those commands.

```bash
python experiments/run_nuclide_suite.py \
  --e1-summary-csv cache/e1_master_table_summary.csv \
  --e9-summary-csv cache/e9_fixed_duration_summary.csv \
  --cache-root cache --case heu_sphere_nu \
  --k-targets 0.50 0.70 0.80 0.90 0.95 0.97 0.98 0.99 0.995 \
  --n-steps 200 --max-replicas 24 \
  --nuclides Xe135 I131 U235 \
  --out-root outputs
  # optional Method B overlay (needs step 2b's merged CSVs, on E1B/E9B's
  # own 14-point k-target grid; the figure script only plots whatever
  # nuclide-level k_targets are actually present, so passing E1/E9's
  # 9-point --k-targets above still works):
  # --e1b-summary-csv cache/e1b_master_table_summary.csv
  # --e9b-summary-csv cache/e9b_fixed_duration_summary.csv
```

This produces, per nuclide, under `outputs/<nuclide>/`:

```
e1_summary.<nuclide>.csv, e9_summary.<nuclide>.csv   (and e1b/e9b if requested)
fig_variance_scaling.pdf / .png
fig_trajectory_bands_relative.pdf / .png
clock_variance_report.txt
```

Two flags worth knowing: `--skip-trajectory-bands` skips the slowest
part of the suite (trajectory-band figures redraw every replica's full
curve) for a quick scaling-only pass; `--skip-clock-diagnostic` skips
`diagnose_R_variance.py`. `--t-star` sets a shared calendar target for
the clock diagnostic across all k_targets — if omitted, each k_target
uses its own $0.9\times\min(\text{replica } t_\text{end})$, printed to
the console.

### Standalone equivalents

Everything `run_nuclide_suite.py` does can be run by hand, per nuclide.
This is useful if you only need one output, or want different arguments
for different nuclides.

**Extract one nuclide's inventory** (no rerun — reads already-cached
trajectories):
```bash
python experiments/extract_nuclide.py \
  --summary-csv cache/e1_master_table_summary.csv --cache-root cache \
  --nuclide Xe135 --out-csv cache/e1_summary.xe135.csv
python experiments/extract_nuclide.py \
  --summary-csv cache/e9_fixed_duration_summary.csv --cache-root cache \
  --experiment e9_fixed_duration --nuclide Xe135 \
  --out-csv cache/e9_summary.xe135.csv
# same pattern for e1b_master_table / e9b_fixed_duration if you ran 2b
```

**Variance-scaling figure, NSCM+Method B combined** (the paper's figure;
needs E1+E9 extracted, plus E1B+E9B if you want the overlay):
```bash
python experiments/make_fig_variance_scaling.py \
  --e1-summary-csv cache/e1_summary.xe135.csv \
  --e9-summary-csv cache/e9_summary.xe135.csv \
  --e1b-summary-csv cache/e1b_summary.xe135.csv \
  --e9b-summary-csv cache/e9b_summary.xe135.csv \
  --cache-root cache --case heu_sphere_nu \
  --k-targets 0.50 0.70 0.80 0.90 0.95 0.97 0.98 0.99 0.995 \
  --n-steps 200 --max-replicas 24 --nuclide Xe135 \
  --out fig_variance_scaling_xe135
```

**Variance-scaling figure, Method B (fixed-source) standalone** — pass
`--skip-nscm` and omit the `--e1-summary-csv`/`--e9-summary-csv` pair
entirely; only `--e1b-summary-csv`/`--e9b-summary-csv` are required in
this mode, and the panel titles/theory annotations switch to the
Method-B-only wording automatically:
```bash
python experiments/make_fig_variance_scaling.py \
  --skip-nscm \
  --e1b-summary-csv cache/e1b_summary.xe135.csv \
  --e9b-summary-csv cache/e9b_summary.xe135.csv \
  --cache-root cache --case heu_sphere_nu \
  --k-targets 0.50 0.60 0.70 0.80 0.85 0.90 0.93 0.95 0.96 0.97 0.98 0.985 0.99 0.995 \
  --n-steps 200 --max-replicas 12 --nuclide Xe135 \
  --out fig_variance_scaling_xe135_methodB
```

**Trajectory-band figure (relative-deviation)** — plots each replica as
its fractional deviation from that k_target's own ensemble mean, which
is what actually shows k-dependent replica spread for a saturating
fission product on a shared axis; pass `--no-relative` for the original
raw-$N$ behavior:
```bash
python experiments/make_fig_trajectory_bands_relative.py \
  --cache-root cache --case heu_sphere_nu \
  --k-targets 0.50 0.70 0.80 0.90 0.95 0.97 0.98 0.99 0.995 \
  --n-steps 200 --max-replicas 24 --nuclide U235 \
  --out fig_trajectory_bands_u235
```

**Clock-variance diagnostic** — decomposes the observed $R(t^*)$
variance into the clock-displacement mechanism vs. the local
$k$-estimator noise floor, per $k_\text{target}$; read-only, touches
only cached `trajectory.pkl` files:
```bash
python experiments/diagnose_R_variance.py \
  --cache-root cache --case heu_sphere_nu \
  --k-targets 0.50 0.70 0.80 0.90 0.95 0.97 0.98 0.99 0.995 \
  --n-steps 200 --max-replicas 24 \
  --out-txt clock_variance_report.xe135.txt
```

All figures render with real LaTeX if a working `latex` is on `PATH`
(checked with an actual test render, not just presence on `PATH`), and
fall back to matplotlib's built-in mathtext otherwise.

## Run parameters (current values, and what changed)

The values below are what `slurm/submit_*.sh` currently pass. If you're
comparing against figures generated before this revision, note what
moved:

| parameter | E1 / E9 | E1B / E9B | previously (E1/E9) |
|---|---|---|---|
| particles/generation | 16000 | case default (8000) | 8000 |
| generations/step | 200 | 200 | 200 |
| inactive cycles/step | 40 (flat) | case default (20, flat) | 20 (flat) |
| replicas | 24 | 12 | 12 |
| $k_\text{target}$ grid size | 9 | 14 | 9 |
| array size | 216 (`0-215%64`) | 168 (`0-167%16`) | 108 |

Particles, inactive cycles, and replicas were all raised for E1/E9 (not
E1B/E9B) to narrow the per-point replica statistics and shrink the
per-step $k$-estimator noise floor that `diagnose_R_variance.py`
identified as dominating the observed $R(t^*)$ variance over the
intended clock-displacement mechanism the $M^4$/$M^6$ theory rows are
about — see that script's own docstring. The $k_\text{target}$ grid was
widened only for E1B/E9B (Method B / fixed-source), at the original
replica count: Method B's per-task cost already grows with $M$ near
criticality, so more $k$-points was the requested lever there, not more
per-point statistics.

The inactive-cycle count remains a **flat, $k$-independent constant**,
deliberately — not a placeholder for a schedule. The paper's own
approach-to-critical analysis shows the required inactive-cycle count
for a given source-convergence bias is $k$-independent for this
benchmark family, so a flat count was always the theoretically-motivated
choice; the increase from 20 to 40 is purely about the noise floor
above, not source convergence. (`integrators.py`'s
`_maybe_set_n_inactive` docstring describes how a genuinely $k$- or
step-dependent schedule *could* be implemented via
`n_inactive_schedule`, for anyone who wants to explore a warm-start
approach later — this package does not use that mechanism.)

`sbatch` script changes to match: `--time` raised to 16h (E1/E9, from
8h) and 48h (E1B/E9B, from 24h) to cover the larger per-task cost;
`#SBATCH --wckey=ne_gen` added to every submit script.

## Things worth knowing before rerunning this on different hardware or parameters

- **E1B/E9B (Method B) have not been run or timed in this environment.**
  Branching fixed-source has no population control: each of
  `--n-particles` starting histories is followed through its entire
  branching descendant tree within one batch, and that tree's size
  grows with $M$ near criticality — where SCM's per-generation cost
  stays fixed regardless of $k$. Pilot a single near-critical replica
  (edit `--array` to one index) before submitting either full array,
  and adjust `--time`/the array's `%`-concurrency limit based on what
  you actually observe, not the placeholder values in the submit
  scripts. Separately, E1B's exposure target is an *approximation*, not
  an exact match to E1's own delivery: it derives a per-$k_\text{target}$
  calendar window from $M=1/(1-k_\text{target})$ rather than stepping in
  exposure directly (Method B has no native exposure coordinate — see
  `run_e1b_master_table.py`'s module docstring), so check the printed
  `tau_end` against `--tau-star` for a few replicas before trusting a
  full array's worth to be "at the same burnup" in the sense E1's own
  `tau_max` is. E9B has no such approximation — it mirrors E9 exactly.
- **`k_target=0.998` is excluded from every sweep.** Every replica goes
  supercritical partway through the exposure window at this point,
  which makes the trajectory's calendar clock non-monotonic and breaks
  the interpolation both `run_e1_master_table.py`'s downstream analysis
  and `make_fig_variance_scaling.py`'s matched-burnup panel depend on.
  `analysis_common.safe_interpolate_at_time` detects and skips any
  individual replica this happens to (a transient excursion can affect
  a single replica even at lower $k_\text{target}$), but the full
  $k=0.998$ point was dropped from every sweep entirely rather than left
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
- **E1/E9 now use a flat 40 inactive cycles out of 200 generations per
  depletion step** (`n_inactive_schedule=lambda _: 40`, doubled from the
  original 20 — see "Run parameters" above for why). This was not
  always the case: earlier versions of this pipeline never set
  `openmc.lib.settings.inactive` at all, so every generation counted
  toward the multiplication estimators from the first cycle, with no
  source-convergence burn-in. If you're comparing against results
  generated before that was wired up, they are not directly comparable
  either. No warm-start schedule is used (a `n_inactive_schedule`
  callable can implement one, shrinking the count once the source bank
  has settled between consecutive depletion steps — see
  `integrators.py`'s `_maybe_set_n_inactive` docstring — but this
  package uses a flat constant deliberately, not that mechanism).
- **The matched-burnup rows in the figure split each $k_\text{target}$'s
  replica pool in half** (a reference half used only to pick the
  calendar-time interpolation target, a measurement half actually
  contributing to the plotted variance) to avoid a self-referential
  bias that a single shared ensemble would introduce — see
  `make_fig_variance_scaling.py`'s `_series_matched_burnup` docstring.
  This roughly halves the effective replica count behind those two
  lines; with replicas now at 24 (E1/E9) this leaves 12 per side, the
  same effective count the whole pipeline used before this revision.
- **The log-log fits in the figure and in its console output are
  unweighted** — they do not use the $\sqrt{2/(R-1)}$ replica-variance
  uncertainty the error bars themselves show. Treat a fitted exponent
  as more trustworthy where the error bars are visibly small.
- **`diagnose_R_variance.py`'s `k_std` numbers are a likely-understated
  lower bound**, not an exact value: the in-run cycle-to-cycle estimator
  is known to be biased low by source-iteration correlation, especially
  under-converged (high dominance ratio / high $k$ / insufficient
  `n_inactive`) cases. If the local-noise mechanism is already reported
  as exceeding the clock-displacement mechanism using this
  likely-understated `k_std`, treat the real gap as probably wider, not
  narrower.
