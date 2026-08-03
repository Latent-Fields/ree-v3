# Corpus Lint Index

One row per `tests/contracts/test_*_lint.py` file. Each was added reactively,
after one specific confirmed incident — this is the incident-driven ratchet
described in `REE_assembly/evidence/planning/experiment_verification_harness_plan.md`
(Gap 2). Check this table before recommending a new lint: the bug class you
found may already be covered.

**Count as of 2026-08-04: 19 files.** (A prior plan-doc draft said 47 — that
figure counted `.pyc` cache variants and nested `.claude/worktrees/` copies
alongside the source files; corrected here. Recount with:
`find . -iname "test_*_lint.py" -not -path "*__pycache__*" -not -path "*/.claude/worktrees/*" | wc -l`
from `ree-v3/`.)

**Hard vs Warn-only:** "Warn" means the underlying `validate_experiments.py
--checks <name>` finding never hardens the exit code, in `--strict` or full-glob
mode — it is diagnostic, printed for a human to read. "Hard" means a real
finding fails a pytest assertion (or, for `arm_fingerprint`/`manifest_writer`,
fails `validate_experiments.py --strict --paths <script>` and therefore blocks
`precommit_contracts.sh` Block 1 / Block 1b / Block 1c).

| Lint file | `--checks` name | Bug class | Hard/Warn | Motivating incident |
|---|---|---|---|---|
| `test_agent_construction_seed_order_lint.py` | `agent_construction_before_seed` | Agent constructed (torch weight init) before torch RNG is seeded in the same execution flow — initial weights are not a function of `seed`. | Warn | Found 2026-08-01 in `q081_pair_reach_check.py`; corpus audit found the same shape in 18 driver scripts (Q-081, INV-091 families). |
| `test_anchor_reachability_lint.py` | `anchor_reachability` | A readiness-anchor precondition's hand-written predicate is narrower than the state it anchors to — unmeetable by construction, so it reports `met=false` forever. | Warn | V3-EXQ-778d; `failure_autopsy_SD-068-rem-fanout-cluster_2026-07-18.md` sec 2. |
| `test_arm_fingerprint_lint.py` | n/a (own selector) | Multi-arm script (writes `arm_results`) omits the per-cell RNG reset + fingerprint emission required for arm-reuse determinism. | **Hard** under `--strict --paths` (advisory in full-glob backlog mode) | Design gate ratified 2026-06-07, `arm_reuse_fingerprint_plan.md`. |
| `test_config_slice_declaration_lint.py` | `config_slice_declaration` | A cross-driver-reusable arm cell's `config_slice` omits a numeric constant its own call graph reads — a false-cache-HIT risk (worse than a false miss). | Warn | `arm_reuse_fingerprint_plan.md` sec 7b. |
| `test_dacc_last_bundle_lint.py` | `dacc_last_bundle` | Driver reads the dACC bundle via the wrong attribute path (`dacc._last_bundle` instead of `agent._dacc_last_bundle`); the wrong `getattr` default silently returns `None`, pinning dACC metrics at 0.0. | Warn | Found 2026-07-29; V3-EXQ-687 self-routed `substrate_not_ready_requeue` on the resulting fake zero. 15 landed carriers are record-frozen but still counted by the corpus pin. |
| `test_dead_z_goal_stream_lint.py` | `dead_z_goal_stream` | Driver enables a `z_goal`-dependent config but never calls the sole writer (`REEAgent.update_z_goal`) — every goal-gated branch silently no-ops for the whole run. | Warn | Confirmed twice, opposite orders: V3-EXQ-626 (2026-06-01) and a later recurrence. |
| `test_dose_saturation_lint.py` | n/a (manifest-local, `dose_saturation.stamp_dose_saturation`) | Two declared dose levels produce values identical beyond float noise — the manipulation never moved the readout. | Never raises (manifest-stamped WARN) | V3-EXQ-794 (SD-076); `failure_autopsy_V3-EXQ-794_2026-07-22.md` sec 6 item 2. Both SD-076 and MECH-204 went untested while appearing tested. |
| `test_dry_run_unreachable_criterion_lint.py` | `dry_run_unreachable_criterion` | `--dry-run`'s reduced episode count falls below the absolute episode index a reported detector gates on — its `false` is arithmetically unreachable, not measured. | Warn | Fourth sibling of the dry-run family; distinct from the other three (asks whether the smoke's *criteria* are evaluable, not whether the *output* is marked). |
| `test_e3_diagnostics_staleness_lint.py` | `e3_diagnostics_staleness` | Driver reads a latched E3 `last_*` diagnostic every env-step without clearing it — one `select()` call gets pseudo-replicated into many "independent" rows. | Warn | V3-EXQ-785 (2026-07-19): 600 recorded rows behind 67 genuine `select()` calls (~9x). |
| `test_e3_hold_weighted_readout_lint.py` | `e3_hold_weighted_readout` | Pseudo-replication FORM 2 — accumulating from the E3 `select_action` return value or cached candidates during a hold, weighting by hold duration rather than per-selection; structurally blind to the staleness-lint's latch check. | Warn | V3-EXQ-699 re-adjudication (`REE_assembly ac2fb64028`) forced withdrawal of the `levers_compound` finding. |
| `test_emit_outcome_dry_run_lint.py` | `emit_outcome_dry_run` | Driver reduces work under `--dry-run` but doesn't thread `dry_run=` into `emit_outcome` — a smoke manifest isn't relocated out of the scoring path. | Warn | Sibling of `hardcoded_dry_run`; V3-EXQ-650 threaded the writer flag but not the emitter flag. |
| `test_hardcoded_dry_run_lint.py` | `hardcoded_dry_run` | Driver hardcodes a literal `dry_run=False` into `write_flat_manifest` even under `--dry-run` — a toy-episode smoke manifest looks real and lands straight in `evidence/experiments/`. | Warn | V3-EXQ-696 relocation gap (defence-in-depth; `emit_outcome` and `generate_pending_review.py` are the other layers). |
| `test_inert_arm_knob_lint.py` | n/a (manifest-local, `inert_arm_knob.stamp_inert_arm_knob`) | Two declared-distinct arms are bit-identical on every recorded per-cell field at matched seed — the fingerprint says "distinct", the readouts say otherwise. | Never raises (manifest-stamped WARN) | V3-EXQ-689d: 26/27 fields identical across two arms; a conjunctive criterion silently degraded to testing only one conjunct. |
| `test_inert_salience_dacc_bias_lint.py` | `inert_salience_dacc_bias` | `salience_apply_to_dacc_bias=True` set with no positive `dacc_weight` — the dACC->E3 channel (MECH-244) multiplies a zero vector; no error, a guaranteed null that looks measured. | Warn | V3-EXQ-799 near-miss, 2026-07 — caught by a P0 probe before landing. |
| `test_invalid_escape_sequence_lint.py` | n/a (corpus-wide pytest assertion) | An invalid `\x` escape sequence in a non-raw string literal — `DeprecationWarning` today, `SyntaxError` in a future CPython, which would take out parsing (and every lint) for that file. | **Hard** (`test_corpus_has_no_invalid_escape_sequences`, pinned at zero) | 2 files carried real instances until the 2026-07-28 fix (`37673f280b`); previously unattributed because `ast.parse` was called with no `filename=`. |
| `test_manifest_writer_lint.py` | `manifest_writer` | A script carrying manifest-identity tokens (`run_id` + `evidence_direction`) does a raw `json.dump` instead of routing through `pack_writer.write_flat_manifest`. | **Hard** under `--paths` (this is the `precommit_contracts.sh` Block 1b commit gate) | `pack_writer_single_writer_migration_plan.md` sec 7 item 3. |
| `test_precondition_recomputability_lint.py` | `precondition_recomputability` | A precondition's `met` is not recomputable from its reported measured/threshold/direction — missing `direction`, or `met` computed from a different statistic than `measured`. | Warn | V3-EXQ-648a/649 (2026-06-07 directionality bug); V3-EXQ-726 (median-across-seeds `measured` vs seed-count `met`). |
| `test_spearman_guard_shape_lint.py` | `spearman_guard_shape` | Hand-rolled rank correlation guards degeneracy on the variance of the RANK vector (double-argsort) rather than the input — the guard never fires on a genuinely constant input. | Warn | 18 pre-SD-081 experiment scripts; `|rho|` up to 0.74 measured on constant vectors. `failure_autopsy_sd081-spearman-degenerate-dv_2026-07-27.md`. |
| `test_write_pack_dry_run_lint.py` | `write_pack_dry_run` | Driver threads `dry_run` into the flat manifest / `emit_outcome` but not into `pack_writer.write_pack` — the RUN PACK, which is what the indexer actually scores, doesn't carry the flag. | Warn | Third sibling of the dry-run family; the pack (not the flat manifest) is on `build_experiment_indexes._scan_runs`'s scoring path (MECH-245). |

## Not in this table

- **`test_precommit_contracts_experiment_lint_scope.py`** and **`test_corpus_scan_sharing.py`** — test the lint *infrastructure* (the commit-gate trigger, the shared corpus-scan fixture), not a bug class in experiment scripts themselves.
- **`test_ceiling_route_anchor_floor.py`** — has "lint"-shaped assertions but is not named `test_*_lint.py`; not audited for this index.
- Semantic checks that live in `validate_queue.py` rather than a `tests/contracts/test_*_lint.py` file (e.g. `prereg_share_feasibility_lint`) — see `queue-experiment/SKILL.md` instead, which documents those inline.

## Maintenance

When you add a new `test_*_lint.py` file, add a row here in the same commit.
When you're about to recommend a NEW lint (from an autopsy, a chip, or your own
review), check this table first — the bug class may already have a WARN-only
gate that just needs a driver fixed, not a new check written.
