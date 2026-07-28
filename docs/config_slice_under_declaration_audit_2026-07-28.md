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
  name-based call graph cannot follow, and **on cross-module slices** (a helper imported from
  `_lib/baselines/` -- see the addendum). A `False` here is not a clearance.
  *(The "does not scope the 15 non-`with` scripts" limit recorded here originally was closed
  the same day -- see **Addendum** below. The slice-built-in-a-function limit was closed too,
  for local helpers.)*
- **Over-fires** on derived constants already bound by a declared key (798a's `N_SSL_BINS`), on
  thresholds read lexically inside the `with` body, and on name-heuristic near-misses.
- **String constants are deliberately out of scope.** In this corpus they are overwhelmingly
  labels / paths / run-ids; gating them drowns the signal.

Opt-out: `CONFIG_SLICE_DECLARATION_EXEMPT = "<reason>"`.

**Pinned corpus fire count: 56** (`_PINNED_CORPUS_FIRE_COUNT`, per the existing convention
alongside 150 / 63 / 12 / 0). Was 48 as first landed; re-pinned the same day by the addendum
below. A NEW script in that list should be fixed or exempted, not re-pinned; a count that
DROPS because a carrier was fixed should be re-pinned.

The lint is registered in `tests/contracts/conftest.py`'s `path_lints` per that module's
standing pattern -- it does not enumerate `experiments/` itself, so it adds no seventh
corpus walk.

---

## Addendum (2026-07-28, same day): the 15 non-`with` scripts, closed

As landed above, the lint located the cell body **only** via `with arm_cell(...) as cell:`.
Of the 85 scripts passing `include_driver_script_in_hash=False`, 70 use that form; the other
15 call `compute_arm_fingerprint(...)` directly, so the lint returned early and scanned
nothing. **Those 15 were unexamined, not cleared** -- and that state was invisible from
outside, because "returned None because I could not find a cell" looks exactly like "clean".

Investigating them turned up **two** defects, not one.

### Defect 1 -- scope. The chosen scope is the nearest enclosing LOOP, not the enclosing function.

The direct-call form has no lexical cell body: the call computes a hash and the arm is run by
a sibling statement. The obvious candidate scope -- the enclosing function plus transitive
callees -- **degenerates**, and it was measured rather than assumed. The enclosing function is
`run_experiment` in 13 of the 15, and `run_experiment` also calls `evaluate(rows)`, so the
scope swallows the whole adjudication block. Calibrated against the 60 with-form scripts,
where the `with` body **is** ground truth:

| scope | fires (of 60) | reproduces with-scope set | spurious constants added | byte-identical to whole-module scan |
|---|---|---|---|---|
| `with` body (ground truth) | 48 | -- | -- | -- |
| **nearest loop** (shipped) | 52 | **40 / 60** | **+79** | 7 / 60 |
| enclosing function (rejected) | 52 | 28 / 60 | +271 | 18 / 60 |
| whole module (the original over-firing baseline) | 55 | -- | +588 | -- |

On the direct-call scripts the difference is exactly the adjudication band: loop scope excludes
`MIN_SEEDS_FOR_PASS`, `DIVERGENT_PASS_FRACTION`, `ABLATION_MARGIN`, `CONVERSION_MARGIN`, all of
which enclosing-function scope pulls in. Attribution on V3-EXQ-700c confirmed the loop-scope
names are read by `_make_agent` (27), `_run_seed_arm` (15) and `_lpfc_reinforce_loss` (3) --
the arm-execution path -- and by **no** criteria function. *(Worth recording: those names carry
`_FLOOR` / `_THRESHOLD` suffixes and read at a glance like criteria. Judging them by suffix
rather than by which function reads them is what nearly got loop scope wrongly rejected here.)*

**Loop scope changes the result of zero with-form scripts**, so the extension is strictly
additive; the with-form half of the pin moved only for defect 2.

### Defect 2 -- declaration resolution. All 15 build the slice through a helper.

None of the 15 passes a literal dict except V3-EXQ-714. The rest call a helper the resolver did
not follow, so the slice resolved as **empty** and the fire was dominated by constants that are
in fact declared one call away:

- **10 scripts** (704, 704b, 707, 707a, 707b, 707c, 708, 708a, 708b, 710) use a **local in-file**
  `_arm_config_slice()`. `_absorb` now absorbs a local function's `return` values. This also
  cleared **pre-existing false positives on 13 with-form scripts** (793/793a 41 names -> 15,
  794/794a 11 -> 2, 766 8 -> 1, 751 3 -> 0, 735 1 -> 0), which is why the with-form half of the
  pin moved **down**.
- **4 scripts** (685, 700c, 700c_mint, 700d) import the helper from
  `experiments/_lib/baselines/`. That is genuinely **cross-module** and a single-file scan cannot
  see one key of it, so these are now an explicit **documented skip** rather than a fire. This
  matters because firing there would be a near-total false positive landing on the *best-behaved*
  population -- the canonical-baseline pattern CLAUDE.md mandates for a reusable mint. Measured
  on 700c: **21 of the 27** constants a fire would have named are declared in that module's
  `MATCHED_ENVELOPE`. The skip is keyed on the `config_slice` argument itself being a call to the
  imported helper, not on the mere presence of a `_lib/baselines` import -- a script that imports
  `REUSABLE_ARM_IDS` but builds its slice inline stays fully assessed
  (`test_cross_module_skip_needs_the_slice_to_come_from_that_helper`).

### Outcome

Fire count **48 -> 56**: 45 with-form (was 48; -2 resolver, -1 cross-module skip) + **11
direct-call newly covered**. The four remaining direct-call scripts are accounted for
individually -- 685 has nothing undeclared, 700c_mint has no module-level numeric constants,
700c/700d are the cross-module skip. `test_no_cross_driver_script_is_silently_unscanned` is the
standing check that no cross-driver script sits in the old invisible fourth state.

The 11 new carriers are **one driver family** (the ARC-110 / MECH-440 / MECH-451 lineage on a
shared template), so their 38-45 names each are the same defect replicated rather than 11
independent problems: their `_arm_config_slice` declares the envelope's **booleans**
(`use_f_eligibility_demotion`) but not its **values** (`f_eligibility_envelope_floor`, which
goes straight into the agent build). Spot-verified on 704 at line 440 against a slice that
declares only the boolean. Per the standing rule these landed runs are **not** retro-fixed --
the backlog is a risk register, and the entry that matters is the one whose baseline a future
consumer tries to reuse.

**Still open (out of scope here):** the cross-module band. Closing it needs the lint to parse
the imported `_lib/baselines/` module, which is a different kind of change -- the lint is
currently a strictly single-file `path -> Optional[str]` function riding the shared corpus
parse cache, and cross-module resolution would need its own calibration.

---

## Addendum 2 (2026-07-28, same day): the cross-module band, closed

The "still open" item above is now closed. `config_slice_under_declaration_lint` resolves a
`config_slice` built by a helper imported from `experiments/_lib/baselines/` by parsing that
module: it absorbs the helper's returned dict, a sibling helper it tail-calls
(`off_path_config_slice` -> `arm_config_slice`), and the module-level dicts it splices in
(`dict(MATCHED_ENVELOPE)`, `dict(ENV_KWARGS)`) -- whose **keys** are what actually bind the
driver's constants. The blanket skip is retained **only** as the fallback: if the module cannot
be located, parsed, or the named function found with a returned value, the check says nothing.
All 5 affected drivers resolve, so that fallback is currently unreachable in the corpus.

### The two spellings that decide whether this is real or nominal

Parsing the module bought **nothing** on the corpus -- 700c stayed at 55 names -- until both of
these were handled. The canonical baseline modules write the envelope as an **annotated**
assignment bound to a `dict(...)` **call**:

```python
MATCHED_ENVELOPE: Dict[str, Any] = dict(settling_rounds=3, ...)
```

so a mapping test of `isinstance(node, ast.Dict)` declines to enter it, and a name->value map
built from `ast.Assign` alone never finds it at all. Each half is a silent no-op on its own: the
module parses, the helper resolves, and no result changes. Both are now handled
(`_is_mapping_expr`, `_module_assigns_and_funcs`) and both are pinned directly
(`test_spliced_module_dict_declares_its_keys_through_dict_call_and_annotation`) rather than left
to the corpus count -- which, see below, could not have caught them.

### Calibration (done before shipping, per the standing convention)

| driver | before resolution | after | note |
|---|---|---|---|
| 700c | 55 | **31** | 0 of the 31 appear as a key or kwarg anywhere in `exq700_arc108_settling_baseline.py` |
| 700d | 56 | **32** | same module, same profile |
| 833 | 1 | **1** | `STAGE0_ZGOAL_GATE` -- genuine, see below |
| 685 | -- | **clean** | one module constant (`DEMO_SEED`), declared |
| 700c_mint | -- | **clean** | no module-level numeric constants at all |

That the surviving 31 are **absent from the baseline module entirely** is the check that the
resolution is complete with respect to it rather than partial -- a partial resolution would leave
names the module does declare. Hand-checked sample of the 31: `CRF_MAINTENANCE_DECAY`,
`CONTRASTIVE_BATCH_K`, `MAX_GRAD_NORM`, `POLICY_TEMPERATURE` are genuine, and are the **same
booleans-not-values defect the 704 family carries** -- `MATCHED_ENVELOPE` declares
`use_candidate_rule_field=True` but none of the CRF values, while `CRF_MAINTENANCE_DECAY = 0.0`
goes straight into the agent build. The four `FIELD_NOISE_*` / `NOISE_FLOOR_*` names are the
known over-fire class: read lexically in the loop but bound off for every reusable arm
(`noise_floor_alpha=(NOISE_FLOOR_ALPHA if noise_on else 0.1)`, all four reusable arms
`noise_on=False`), which is exactly what the baseline module's own docstring says it excludes on
purpose. 833's single name is a true positive of the **scheme** band -- the severest one --
`STAGE0_ZGOAL_GATE = 0.4` decides the recorded `stage0_zgoal_formed` readout and no baseline key
binds it.

### Outcome: the count did not move, and that is the finding to carry forward

Fire count **56 -> 56**. The two effects cancelled exactly, and reading that as "nothing
happened" would be wrong -- the **set** turned over by six scripts:

- **+3** 700c, 700d, 833 left the unassessable band and now fire; 685 and 700c_mint left it and
  are now verifiably clean rather than merely silent.
- **-3** 114a, 120a, 266b cleared outright as **pre-existing false positives** -- all three
  declare their constants in an annotated `full_config: Dict[str, Any] = {...}`. Eight more
  shrank (777 12 -> 7, 777a 11 -> 8, 779 14 -> 8, 779a 18 -> 8, 779b 18 -> 9, 800/801 2 -> 1,
  and `REINFORCE_BATCH_SIZE` off all 11 direct-call carriers).

A pinned **count** is structurally blind to a net-zero turnover, so the coverage claim is carried
by a named set instead -- `_CROSS_MODULE_CARRIERS` + `test_cross_module_carriers_stay_covered`,
mirroring what `_DIRECT_CALL_CARRIERS` does for the loop scope. Reverting the resolver would take
the count 56 -> 53, which reads as remediation; the named set fails as a loss of coverage.
`test_no_cross_driver_script_is_silently_unscanned` was tightened in the same pass: its
accounted-for reason is no longer `"_lib.baselines" in src` (which, once the band is resolvable,
would let any script go silent just by importing from a baselines module) but an actual
resolution attempt.

Cost: the resolution is a second parse for ~5 drivers, cached by (path, mtime, size). Corpus
scan measured at **7.90s before / 7.84s after** -- no measurable change. Per the standing rule
the landed carriers are **not** retro-fixed; the backlog remains a risk register. The gate stays
**WARN-only in both modes**.

---

## Addendum 3 (2026-07-28T21:44Z): 833's `STAGE0_ZGOAL_GATE` adjudicated -- DECLARE, deferred to run completion

Addendum 2 left `STAGE0_ZGOAL_GATE` named as "a true positive of the scheme band" without saying
what to do about it. This addendum decides that, and records why the code edit is **deliberately
not landed in the same pass**.

### Verdict: declare it. Exemption is not available on the merits.

The tension is real and worth stating, because the naive reading of the baseline module points
the other way. `off_path_config_slice()`'s own docstring says the slice

> "must NOT carry acceptance thresholds or ON-arm gains -- those do not change the computation,
> and folding them in would refuse a legitimate reuse on every threshold tweak."

`STAGE0_ZGOAL_GATE = 0.4` **is** an acceptance threshold, so read literally that rule licenses its
omission and the lint is over-firing. It does not, and the reason is the rule's rationale rather
than its wording: a threshold is excluded because **each consumer recomputes it from raw rows**, so
addressing it buys nothing. That rationale holds for every other threshold in the driver --
`HARM_DISC_RANGE_FLOOR`, `SURVIVAL_FLOOR_STEPS`, `MIN_FRACTION`, `K_SE`, `ABS_FLOOR_STEPS`,
`NULL_PRECISION_CEILING_STEPS` -- all of which are read in the post-hoc analysis, **outside**
`arm_cell`. It fails for `STAGE0_ZGOAL_GATE` alone, which is read **inside the cell body** and
whose verdict is stamped into the cached row as `stage0_zgoal_formed` (driver lines 471, 617).
A consumer does not recompute an inherited field; it reads it. That is the constant crossing from
*threshold* into *scheme*, and it is exactly the line the lint's call-graph scope implements. It is
the only constant in this driver that crosses it.

So declaring the gate does not violate the module's rule -- it honours the rule's actual reason,
and the rule's wording should be amended so the next author does not read it as licensing the
omission. `CONFIG_SLICE_DECLARATION_EXEMPT` would be a false statement here: nothing already in the
slice binds the gate.

### Severity is genuinely lower than 798's, and that is what makes deferral affordable

Both record paths stamp the raw `stage0_z_goal_norm_peak` **beside** the derived boolean
(lines 470/471 and 616/617). A consumer at a different gate can therefore recompute formation
exactly; the false HIT is **recoverable**. Contrast the confirmed 798 instance, where
`learn_pe_by_ssl_bin` is a per-bin aggregate under `SSL_BIN_EDGES` -- lossy, and the original
binning is not recoverable from the stamped row. Recoverability does not change the verdict (it
depends on a future consumer *noticing*, which is the assumption the fingerprint mechanism exists
to remove) but it is what makes waiting a sound trade rather than a compromise.

### Fork taken: WAIT for V3-EXQ-833 to finish, then declare. The edit is blocked, not forgotten.

Landing the one-line declaration now would **certainly** destroy the run's reuse value, and this
was verified rather than assumed:

- `_SUBSTRATE_GLOBS` includes `experiments/_lib/**/*.py` (`arm_fingerprint.py:68-73`), so any byte
  change to the baseline module changes the substrate hash.
- `experiment_runner.py:2958-2962` starts `_background_sync` as a daemon thread that calls
  `_sync_pull_tick` -> `_pull_ree_v3` **every 60s while the experiment subprocess is running**. The
  hub's checkout therefore adopts `origin/main` within ~a minute of a push, mid-run.
- `substrate_stability_report()` re-hashes from disk at stamp time, so the manifest would carry
  `substrate_stable_across_run: false`, and `arm_reuse.source_run_substrate_unstable()` refuses to
  serve **any** cell from it.

Confirmed live at 21:43Z: 833 running on the hub as PID 2515777, claimed `2026-07-27T23:39:56Z`,
`estimated_minutes=1800`, zero manifests emitted -- roughly 8h remaining of a 30h run over
2 arms x 20 seeds.

That trade is one-sided. The cost of landing now is the whole ARM_LEGACY cell bank of a 30-hour
mint, refused for **every** consumer including the overwhelmingly likely one using the same
family-standard 0.4. The cost of waiting is that 833's own cells carry the under-declared address
for a hazard that is recoverable in-row. The run's scientific result is unaffected either way
(`CLAIM_IDS = []`, `EXPERIMENT_PURPOSE = "diagnostic"`).

Not taken, and why: the `_ree_v3_pull_blocked` guard (`experiment_runner.py:1142`) would skip the
hub's pull if a TASK_CLAIMS claim named the substrate path, which could be used to slip the edit
past. Defeating a safety mechanism to beat a deadline that does not exist is not a fix.

### The deferred patch, so the follow-on is mechanical

In `experiments/_lib/baselines/stageh_strict_goal_isolation.py` -- move the constant to the
lineage module (it is a lineage-level pre-registered value, not a driver detail; the driver already
imports six constants from here), declare it, and correct the docstring rule:

```python
# --- Stage-0 formation gate (scaffold family standard) ----------------------
# IN THE SLICE, unlike the driver's other thresholds: this one is read INSIDE
# the cell and its verdict is stamped as `stage0_zgoal_formed`, so a consumer
# INHERITS it rather than recomputing it. See audit Addendum 3.
STAGE0_ZGOAL_GATE = 0.4
```

then in `off_path_config_slice()`'s returned dict:

```python
        "stage0_zgoal_gate": float(STAGE0_ZGOAL_GATE),
```

and in the driver, drop the local `STAGE0_ZGOAL_GATE = 0.4` (line 278) in favour of adding
`STAGE0_ZGOAL_GATE` to the existing `from experiments._lib.baselines.stageh_strict_goal_isolation
import (...)` block at line 249.

Amend the `off_path_config_slice` docstring's threshold rule to read: acceptance thresholds
applied in **analysis** are excluded; a threshold read inside the cell whose verdict is stamped
into the row is **scheme** and must be declared.

Then re-run the lint on 833 (expect silent), re-pin `_PINNED_CORPUS_FIRE_COUNT` 56 -> 55 in
`tests/contracts/test_config_slice_declaration_lint.py`, and drop 833 from
`_CROSS_MODULE_CARRIERS` in the same file.

**833's already-minted cells keep the old address.** That is correct and should not be papered
over: they were computed under a slice that does not name the gate. Re-minting them is not
warranted (recoverable in-row, gate at the family standard), but a successor that varies the gate
must not consume them, and the REUSE paragraph of the module docstring is where that belongs.
