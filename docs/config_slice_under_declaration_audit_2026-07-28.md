# Audit: under-declared `config_slice` in arm-reuse fingerprints

**Date:** 2026-07-28
**Scope:** `ree-v3/experiments/` (~1160 top-level `v3_exq_*.py` drivers, ~1240 under `rglob`)
**Governing design:** `REE_assembly/evidence/planning/arm_reuse_fingerprint_plan.md` section 7b
**Gate added by this audit:** `validate_experiments.py --checks config_slice_declaration`
(contract: `tests/contracts/test_config_slice_declaration_lint.py`)

---

## 1. The hazard

`experiments/_lib/arm_fingerprint.py`'s `compute_arm_fingerprint(config_slice=..., ...)` /
`arm_cell(...)` identifies a `(seed x arm)` cell so that a later `try_reuse_cell` HIT returns
an equivalent result. A `config_slice` that **UNDER-approximates** -- omits a parameter the
cell's RECORDED READOUTS depend on -- is a false-cache-HIT bug.

The governing asymmetry (plan section 2): **a false MISS only wastes compute; a false HIT
corrupts a scientific conclusion.** The fingerprint is therefore meant to be deliberately
OVER-inclusive, and the whole-config default exists for exactly that reason. Narrowing the
slice is an opt-in that transfers the correctness burden to the author.

### The interaction that makes this sharper than it looks

`include_driver_script_in_hash=False` is **MANDATORY** for a cross-driver-reusable mint
(CLAUDE.md, "Saving a baseline for reuse"): with the default `True` the driver's own content
is folded into `substrate_hash`, so a consumer's distinct driver can never match and the cell
is not reusable cross-driver at all.

But it is exactly that flag which removes the driver -- **and therefore every module-level
constant defined in it** -- from the hash. So **the more reusable a mint is made, the more
load the `config_slice` has to carry.** Scripts passing the flag are the exposed set; scripts
with the driver in the hash are structurally safe with respect to their own constants.

---

## 2. Confirmed instance

`experiments/v3_exq_798_sdmelproducer_graded_nonconverging_world.py`
(landed manifest `v3_exq_798_..._20260723T081627Z_v3.json`).

Its cells record `learn_pe_by_ssl_bin` / `learn_ssl_bin_counts` / `learn_decay_frac`, all
computed against the module-level constant:

```python
SSL_BIN_EDGES = (2, 5, 12)   # -> bins [0-2], [3-5], [6-12], [13+]
```

`SSL_BIN_EDGES` is **not** in the `cell_config` slice, and the script emits with
`include_driver_script_in_hash=False` -- so the binning scheme is invisible to the fingerprint
entirely. A consumer using different edges would HIT those cells and silently read bin means
computed under a different scheme.

The chain the lint has to follow to see this:

```
with arm_cell(..., include_driver_script_in_hash=False)
  -> run_cell()  -> _train_p0_and_probe() -> _run_step_budget() -> _ssl_bin() -> SSL_BIN_EDGES
```

The successor `v3_exq_798a_..._c4readable.py` (ree-v3 `7eeb06449d`) declares it --
`"binning"` / `"n_bins"` / `"ssl_bin_edges"` in its `cell_config` -- and is the **reference for
the correct shape**. That pair is asserted directly as a differential in the contract
(`test_confirmed_798_instance_is_the_differential`): the lint must name `SSL_BIN_EDGES` on 798
and must not name it on 798a.

---

## 3. Findings

### Population

| set | count |
|---|---|
| scripts calling `arm_cell(` / `compute_arm_fingerprint(` | 280 |
| ... passing `include_driver_script_in_hash=False` (the exposed set) | 83 |
| ... of those, using the `with arm_cell(...)` form the scan can scope | 68 |
| ... **firing: at least one undeclared readout-affecting constant** | **48** |
| total undeclared constants across those 48 | 373 |

The 15 non-`with` cross-driver scripts are **not cleared** -- they are outside what this
scan scopes. See section 5.

### Classification (task taxonomy (a) under-declared / (b) safe / (c) ambiguous)

**(a) Genuinely under-declared -- the readout-affecting parameter is absent.** This is the
bulk of the 48. Three severity bands, and the distinction matters for triage:

- **Band 1 -- SCHEME constants (highest severity).** A binning / quantisation / discretisation
  choice. A consumer that changes it gets numbers that are not merely differently-parameterised
  but differently-*meaning*, and nothing in the readout's name reveals the change.
  - `v3_exq_798` -- `SSL_BIN_EDGES` (**the confirmed instance**)
  - `v3_exq_816c` -- `PE_HIST_BINS`, `VS_HIST_BINS` (histogram binning for the decoupling readout)
  - `v3_exq_766` -- `KERNEL_BANDWIDTH`, `NUM_CENTERS` (RBF field discretisation)
  - `v3_exq_783` -- `RESIDUE_COARSE_CENTERS`, `RESIDUE_FINE_CENTERS`, `RESIDUE_BANDWIDTH`
    (the granularity crossing's own granularity)
  - `v3_exq_767` / `767a` / `768` / `768a` -- `FAMILIARITY_BANDWIDTH`, `FAMILIARITY_EMA_ALPHA`
- **Band 2 -- ENVIRONMENT / BUDGET shape.** Changes what the cell is measuring over.
  - `v3_exq_777` / `777a` / `779` / `779a` / `779b` -- `ENV_SIZE`, `ENV_HAZARDS`, `ENV_RESOURCES`,
    `ENV_DRIFT_SOURCES`, `STEPS_PER_EPISODE`, `N_EPISODES`, `MAX_ENV_STEPS_PER_CELL`
  - `v3_exq_795` -- `GRID_SIZE`, `NUM_HAZARDS`, `NUM_RESOURCES`, `STEPS_PER_EPISODE`
  - `v3_exq_784` -- `ENV_SIZE`, `ENV_HAZARDS`, `ENV_RESOURCES`, `PROBE_MAX_ENV_STEPS`
  - `v3_exq_744a` / `746a` / `746b` / `746c` -- `COLLECT_SEED_BASE`, `DECODE_SPLIT_SEED`,
    `HELDOUT_FRAC` (the seed derivation and train/heldout split are part of the cell's identity)
- **Band 3 -- TRAINING hyperparameters (lowest severity, most common).** `MAX_GRAD_NORM`,
  `CONTRASTIVE_BATCH_K`, `MIN_BUFFER_BEFORE_TRAIN`, `TRANSITION_BUFFER_MAX`, `E2_WORLD_LR`,
  `ALPHA_WORLD`, `E2W_BATCH`, `SF_TD_COEF`, ... Real under-declaration, but a consumer that
  changes these is usually running a different experiment anyway, so the false-HIT probability
  is lower.

The two largest carriers are `v3_exq_793` / `793a` (41 constants each) -- a long staged
curriculum whose entire stage ladder (`P0_BUDGET`, `HAZARD_STAGE_*`, `STAGE0B_RETENTION_GATE`,
...) sits outside the slice.

**(b) Safe -- correctly not flagged.** Two classes, both verified:
- **Driver in the hash.** The ~197 scripts using the fingerprint API without the flag. Their
  module-level constants are bound by content, so no declaration is needed. The lint's
  `test_quiet_when_driver_is_in_the_hash` pins this scoping.
- **Adjudication thresholds.** `THRESH_*` / `FLOOR_*` / criterion constants read only by
  `evaluate(rows)` **downstream** of the cells. These never enter a recorded readout, and
  scoping to the cell call graph is what excludes them -- without that scoping the check fires
  on **67 of 83** rather than 48, i.e. a whole-module constant scan over-counts by ~40%.

**(c) Ambiguous -- flagged, but plausibly bound by something already declared.** Reported
honestly rather than suppressed, because the remedy (adding the key, or the exempt marker) is
cheap in both directions:
- `v3_exq_798a` fires on `N_SSL_BINS` while declaring `ssl_bin_edges` -- the bin *count* is
  derivable from the *edges*, so this is arguably already bound. It is a known near-miss of the
  name heuristic (`ssl_bin_edges` does not substring-match `N_SSL_BINS`).
- `v3_exq_773` fires on `SPEARMAN_MONO` / `PEAK_EFFECT_FLOOR` / `SEED_PASS_N`. These are
  criterion thresholds, but they are read **lexically inside the `with` body** for a per-seed
  early-exit -- so on the current code they genuinely can change what the cell records.
- `v3_exq_760` -- `AUC_PASS`; `v3_exq_833` -- `STAGE0_ZGOAL_GATE`; same shape.

### 798a is not a clean bill of health

The task framing treats 798a as "the reference for the correct shape", and it is -- **for the
binning**. It still omits four training hyperparameters (`CONTRASTIVE_BATCH_K`, `MAX_GRAD_NORM`,
`MIN_BUFFER_BEFORE_TRAIN`, `TRANSITION_BUFFER_MAX`) plus `MIN_BIN_SAMPLES`. That is band 3 /
ambiguous rather than band 1, and it does not weaken 798a's status as the reference for how to
declare a scheme constant -- but it should not be read as a fully-declared slice.

---

## 4. What was NOT done, and why

**No landed manifest was edited. No historical `v3_exq_*.py` was retro-edited.** A completed
run's pre-registered emission is not rewritten to chase a lint, and editing historical drivers
trips the pre-existing `--strict` backlog (memory `reference-staging-old-experiments-trips-strict-gate`).

The 48-script backlog is therefore a **standing risk register, not a work queue**. The entry
that matters is the one whose baseline a *future consumer* actually tries to reuse -- a false
HIT requires a consumer to exist before it can corrupt anything. The right moment to fix any
row is when a consumer is written against it, and the right place is the successor script.

---

## 5. The gate, and its honest limits

`config_slice_under_declaration_lint` fires on a module-level UPPER_SNAKE constant bound to a
**numeric literal** (or tuple/list of numerics -- the bin-edges shape) that is read from the
**cell's own call graph** (the `with` body plus every module-level function transitively called
from it) and is not named in the resolved `config_slice`.

**WARN-only in BOTH modes** -- it never hardens under `--paths`, matching
`hardcoded_dry_run` / `dead_z_goal_stream`. Rationale: the scan is best-effort in both
directions, the landed carriers' runs are complete, and a false HIT needs a consumer first.

Known limits, stated so the fire set is not over-read:

- **Under-fires** when the value is not a module-level literal (assembled at runtime, imported
  from `_lib`, read from argv/env), when the cell calls a helper through an alias/partial the
  name-based call graph cannot follow, when the slice is built in a function the scan cannot
  resolve, and **on the 15 non-`with` cross-driver scripts**, which it does not scope at all.
  A `False` here is not a clearance.
- **Over-fires** on derived constants already bound by a declared key (798a's `N_SSL_BINS`), on
  thresholds read lexically inside the `with` body, and on name-heuristic near-misses.
- **String constants are deliberately out of scope.** In this corpus they are overwhelmingly
  labels / paths / run-ids; gating them drowns the signal.

Opt-out: `CONFIG_SLICE_DECLARATION_EXEMPT = "<reason>"`.

**Pinned corpus fire count: 48** (`_PINNED_CORPUS_FIRE_COUNT`, per the existing convention
alongside 150 / 63 / 12 / 0). A NEW script in that list should be fixed or exempted, not
re-pinned; a count that DROPS because a carrier was fixed should be re-pinned.

The lint is registered in `tests/contracts/conftest.py`'s `path_lints` per that module's
standing pattern -- it does not enumerate `experiments/` itself, so it adds no seventh
corpus walk.
