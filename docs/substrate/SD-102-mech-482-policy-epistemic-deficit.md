## SD-102 / MECH-482: policy.epistemic_deficit_accumulator -- IMPLEMENTED (2026-08-29)
- SD-102: policy.epistemic_deficit_accumulator -- IMPLEMENTED 2026-08-29. Full design:
  `REE_assembly/docs/architecture/sd_102_epistemic_deficit_accumulator.md`.
  `ree_core/policy/epistemic_deficit.py` (`EpistemicDeficitAccumulator` / `EpistemicDeficitConfig`).
  Fills the `per_candidate_learning_progress` slot MECH-314c's Phase-2 per-candidate extension
  reserved (ree-v3 `c0e0ce8`, 2026-08-08) but left unfilled -- the honest per-candidate 314c source
  is MECH-482 itself, per that landing's own design doc.
  Config: `REEConfig.curiosity_learning_progress_source` {broadcast, epistemic_deficit} -- EXISTING
  reserved enum value, this landing fills the "epistemic_deficit" branch (default stays
  "broadcast", bit-identical -- `self.epistemic_deficit` stays `None`, same None-when-off pattern
  as `self.e2_world_uncertainty` / `self.curiosity`). New: `epistemic_deficit_max_targets` (16),
  `epistemic_deficit_match_radius` (1.0, matches ResidueField's default RBF bandwidth),
  `epistemic_deficit_ema_alpha` (0.1), `epistemic_deficit_uncertainty_weight` /
  `_disagreement_weight` / `_persistent_pe_weight` (1.0 each).
  Data flow: two-phase, mirroring ResidueField's persistent spatially-indexed accumulator (the
  existing precedent for this shape). UPDATE (post-hoc, `REEAgent._update_epistemic_deficit`,
  called from `sense()` right after `_train_e2_world_uncertainty`, same realized-transition cache
  cadence): `(z_world_prev, action_taken, z_world_now)` -> `e2.world_forward` point prediction +
  `E2WorldUncertaintyHead.forward` median quantile prediction + `predictive_variance` ->
  `deficit_input = w_u*uncertainty + w_d*disagreement + w_pe*persistent_pe` -> nearest existing
  target within `match_radius` EMA-updated, or a new target allocated (lowest-deficit target
  evicted at `max_targets` capacity). READOUT (pre-hoc, `REEAgent._curiosity_per_candidate_
  learning_progress`, called from the same `select_action()` site as MECH-314b): this tick's K
  candidate `e2.world_forward` first-step predictions -> nearest-target lookup (read-only) -> `[K]`
  persistent deficit vector -> `StructuredCuriosity.compute_score_bias(per_candidate_learning_
  progress=...)`.
  Candidate inputs (conservative subset per claims.yaml MECH-482 notes): candidate-specific
  predictive uncertainty (SD-063 head, same source 314b's Phase-2 path reads), persistent
  prediction error (REALIZED `||z_world_now - e2.world_forward(...)||`, distinct from 314c's
  `_lp_ema` rate-of-change EMA), predictive-system disagreement (`||e2.world_forward(...) -
  head.forward(...)[median]||` -- two independently-parameterized predictors, MSE-trained vs
  pinball-trained, sharing no parameters). MECH-441's `ModelDisagreementEnsemble` was considered
  and REJECTED as the disagreement source (separate not-built-by-default claim, undeclared
  cross-claim dependency). NOT implemented: failed-replay-resolution / competence-blocking-
  uncertainty inputs (no memory/replay visibility at this integration point) and the full
  `importance x uncertainty x expected_resolvability x persistence` multiplicative formula from
  the claim's title (no `importance` / `expected_resolvability` signal exists in the substrate;
  manufacturing one would be the vacuous-channel risk the 314bc design doc warns against) -- v1
  scope is a persistence-weighted ADDITIVE combination of the three available proxies.
  Readiness gate (binding, corrected ARC-065 gate per `mech314bc_percandidate_extension_staged_
  2026-08-08.md` section 5): READOUT refuses (returns `None` -> Phase-1 broadcast fallback) unless
  the K-candidate batch `predictive_variance` read yields `e2_world_uncertainty_last_pvar_
  relative_spread > 0` this tick -- absolute range alone is necessary but NOT sufficient. A
  refusal calls `accumulator.mark_vacuous_readout()` (self-report, observable via `get_state()`)
  rather than silently reading as "no deficit anywhere." UPDATE is not gated on this (accumulates
  unconditionally every waking tick, mirrors `update_prediction_error`'s always-on cadence).
  MECH-094: `update()` takes `simulation_mode` (no-op, mirrors `update_prediction_error`); no
  memory/replay write surface (waking online read/accumulate, same posture as the SD-063 head).
  Phased training: N/A -- pure arithmetic, no learned parameters, no `nn.Module` (same posture as
  `StructuredCuriosity`).
  Episode lifecycle: `reset()` clears all persistent targets, called alongside
  `StructuredCuriosity.reset()`'s own per-episode 314c LP-EMA clear (MECH-482 is architecturally
  314c's genuine source, inherits the same episode-scoping convention).
  Backward compatible: disabled by default (`curiosity_learning_progress_source="broadcast"`).
  Contracts: `tests/contracts/test_mech_482_epistemic_deficit_accumulator.py` (18 tests, all
  green) -- accumulator unit tests (config validation, target match/create/evict, EMA persistence,
  MECH-094 no-op, readout matching semantics, reset, diagnostics) + agent wiring (OFF path never
  instantiates the accumulator; ON path accumulates across a real rollout; readiness gate refuses
  and self-reports on a single-candidate tick; per-episode reset clears both the accumulator and
  its one-tick-lag prev-z_world cache; simulation_mode does not update). Full existing MECH-314 /
  SD-063 contract suites re-run green (75 tests) confirming no regression.
  Validation experiment: V3-EXQ-964 queued (`/queue-experiment`, `EXPERIMENT_PURPOSE=diagnostic`
  -- substrate readiness, not MECH-482's own claim hypothesis). 2-arm yoked pair (314c source
  broadcast vs epistemic_deficit, 314a/314b OFF on both arms to isolate the factor), SD-063 head
  trained identically on both arms. C1 (load-bearing): accumulator becomes live (`n_targets>0` AND
  `n_updates>0`). C2 (load-bearing): downstream consumer can diverge (yoked divergence vs the
  broadcast reference `> 0` on at least one seed). `--dry-run` smoke: all three checks OK
  (accumulator live, updates fire, self-yoked instrument control `== 0`).
  Governance posture: MECH-482's own `claims.yaml` entry gets an `implementation_note` only (this
  landing does not resolve its non-degeneracy precondition's confirming/falsifying signature,
  which needs the validation experiment's result). Downstream claims naming MECH-482 in
  `depends_on` (MECH-483/Q-089/ARC-121/MECH-485/MECH-487/MECH-493) each still have OTHER unmet
  dependencies, so none had `v3_pending` flipped by this landing. ORNT-2's `status` flip
  (open -> in_progress) is left to `/governance` per this chip's brief, not applied here.
  See MECH-314/314a/314b/314c (parent + siblings), ARC-065 (parent architectural slot), SD-063 (the
  uncertainty-head keystone this reads from), MECH-483/Q-089 (downstream, ORNT-3/ORNT-4).
