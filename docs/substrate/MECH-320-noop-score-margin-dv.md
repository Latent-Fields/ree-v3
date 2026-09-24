## MECH-320 No-op Score-Margin DV Hook: pre- and post-bias action-vs-no-op margin from E3 select (2026-09-24)

- MECH-320 (ARC-066 child): e3.select.noop_score_margin_dv -- IMPLEMENTED 2026-09-24.
  `ree_core/predictors/e3_selector.py` `E3TrajectorySelector.select(noop_class=...)` +
  `_record_noop_margin()`. `ree_core/agent.py` passes the kwarg next to the
  `_e3_select_kwargs` guards, just before `self.e3.select(...)` in `select_action`, and
  under the same guard in `act_with_log_prob` (the REINFORCE path). Chip:
  `chip-20260902-mech320-implement-noop-margin-dv` (authorised by
  `chip-20260901-mech320-noop-margin-dv-substrate`, user-approved through the decision lane
  2026-09-02T16:31Z). Campaign `science-20260924-mech320-margin-hook`.
  Config: `REEConfig.tonic_vigor_record_noop_margin` (default False; set True to enable).
  It is plumbed through `from_dims` (field + signature + re-apply; contract N2). The no-op
  class is `REEConfig.tonic_vigor_noop_class`.
  **On `CausalGridWorldV2` set `tonic_vigor_noop_class=4` (stay).** The default 0 is a
  move class there and never shows up as a no-op candidate class. Measured: 0 no-op
  candidates per tick at 0, against ~25 of 32 at 4, so at the default the DV reads None.
  The flag is independent of `use_tonic_vigor`, so a vigor-OFF control arm still records.
  Data flow: agent (flag on) -> `select(noop_class=k)` -> after `last_selected_idx` is set,
  `_record_noop_margin(candidates, raw_scores, scores, selected_idx, k)` -> the
  `E3.last_noop_margin` dict plus `last_noop_candidate_present` /
  `last_noop_candidate_margin` (POST) / `last_noop_candidate_margin_pre`. All are None when
  the kwarg is absent. A select() without it resets them.
  **Per-step readers: dedupe on `select_seq`.** select() does not run on every agent step:
  committed-hold ticks return early from `select_action`. On those ticks the fields still
  hold the previous select's value, and `select_seq` (a monotone per-E3 count) does not
  advance. Measured: 7 selects in a 30-step episode.
  Candidate class = first-step action argmax, the same classing the MECH-320 bias uses.
  REE scores are lower-is-better.
  - `candidate_margin_{pre,post}` = min score over no-op candidates OTHER than the
    committed one, minus the committed candidate's score, i.e. how far no-op was from
    winning. SIGNED: selection can be multinomial or stratified, so the committed candidate
    need not be the argmin and the value can be < 0.
  - `action_gap_{pre,post}` = min no-op score minus min action score. Independent of
    selection; > 0 means the best action beats the best no-op.
  - Also recorded: `select_seq`, `post_is_selection_basis`, `n_noop`, `n_candidates`,
    `selected_is_noop`, `noop_candidate_present` (for the experiment's
    `noop_candidate_present_rate`).
  PRE = `raw_scores`: after SD-081 dual-system arbitration, before the score_bias chain.
  POST = `scores` (= `last_scores`), after every modulatory term. POST is the ordering the
  selection was drawn from only when `post_is_selection_basis` is True. When a modulatory
  shortlist or loop arbitration (`shortlist_idx`) picked the winner, POST describes an
  ordering selection did not use, so filter those ticks.
  **TAUTOLOGY GUARD.** POST - PRE is the WHOLE modulatory contribution as selection saw
  it. That covers every score_bias channel (dACC, OFC, lPFC, curiosity, MECH-320 vigor,
  ...), the MECH-341 bonus, routed terms and the explore term, not the MECH-320 term alone.
  - With `use_modulatory_selection_authority` OFF and vigor the sole channel, POST - PRE is
    exactly (w_action + w_passive) * v_t (contract N6). That is the injected bias read back,
    not a finding.
  - With authority ON, which the V3-EXQ-951 lineage runs, POST - PRE is rescaled to
    gain * raw_score_range. It is INVARIANT to v_t magnitude: contract N7 pins identical
    shifts at two v_t floors. v_t then acts only as an on/off switch on the POST ordering.
    **This is a design constraint for V3-EXQ-951b:** under authority ON no POST-bias
    margin can carry a v_t-graded MECH-320 effect.
  - Also note: w * v_t saturates at `tonic_vigor_bias_scale` (0.1). Measured: floor 0.5
    gives a 0.1 shift and floor 2.0 gives 0.2, so the bias is capped.
  A MECH-320 criterion must never be a between-arm POST-bias offset. Under the current
  MECH-320 what_would_answer this margin is a SECONDARY DV, and committed action density
  stays load-bearing.
  Backward compatible: diagnostic only, no control effect. Selection is bit-identical ON vs
  OFF, with the tonic-vigor bias off and on (contract N5). The default path never sends
  the kwarg (version-layering guard).
  Contract: `tests/contracts/test_mech_320_noop_margin.py` (N1-N9). On pre-build code every
  test except the bit-identity N5 fails.
  Phased training: no (no learning). MECH-094: not applicable (waking diagnostic only).
  Validation experiment: none queued by this build. The V3-EXQ-951b design is owed via
  /governance + /queue-experiment under MECH-320's current what_would_answer:
  baseline_mode='ewma' arms, a dead-scalar 'none' control, and a DV-headroom bed where
  ARM_0 action density < 1.
  See MECH-320, `docs/substrate/MECH-320-arc-066-child-tonic-vigor-coupling.md`, and the v_raw
  baseline lever (ree-v3 12588f5).
