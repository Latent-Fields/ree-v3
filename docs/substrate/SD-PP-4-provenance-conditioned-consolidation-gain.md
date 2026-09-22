## SD-PP-4 Provenance-Conditioned Consolidation Gain (2026-09-22)
- SD-PP-4: sleep.provenance_conditioned_consolidation_gain — IMPLEMENTED 2026-09-22.
  The CONSUMER of the precision-provenance chain (SD-PP-1 evidence precision,
  SD-PP-2 model precision, SD-PP-3 the replay packet). Turns a per-replay-row
  provenance packet into a bounded per-row consolidation GAIN, and spends that
  gain in the two places the sleep weight-consolidation pass can actually be
  moved. Contract:
  REE_assembly/docs/architecture/precision_provenance_substrate_spec.md section 5.
  Modules and paths:
    ree_core/sleep/provenance_gain.py — NEW. GAIN_MODES, ProvenanceGainConfig,
      compute_provenance_gains(packets, pi_cur, per_row_loss, config) ->
      (gains [K] float32 no-grad, diag dict), weighted_row_loss(per_row_loss,
      gains) = sum(g*l)/sum(g) with g detached. Pure float/tensor arithmetic;
      no RNG; packets are DUCK-TYPED (only .evidence_precision_z,
      .evidence_variance_z, .pe, .surprise, .has_prev are read), so this module
      never imports ree_core/hippocampal/replay_provenance.py.
    ree_core/sleep/cross_module_consolidation.py — CrossModuleConsolidator.
      consolidate() gains two optional kwargs, module_step_scale
      (Dict[str, Callable[[], float]]) and record_trace (bool), plus the
      last_step_trace property. In _step_module the scale callable is read
      AFTER loss_fn() and AFTER backward(), BEFORE opt.step(), and sets every
      param_group["lr"] = rate * scale for that one step.
      step_scale_mean_/min_/max_<name> are emitted ONLY for modules that
      received a scale.
    ree_core/predictors/e2_fast.py — E2FastPredictor.world_forward_contrastive_loss
      gains a final reduction: str = "mean" kwarg ("mean" | "none"); "none"
      returns the per-row CE [K]. The three degenerate early-returns keep
      returning a 0-d zero under either value.
  Config (REEConfig, wired by the integration session, every default a no-op):
    use_provenance_conditioned_consolidation_gain (bool, default False),
    provenance_gain_mode (str, default "provenance"),
    provenance_gain_min (float, 0.02), provenance_gain_max (float, 2.0),
    provenance_gain_surprise_beta (float, 0.5),
    provenance_gain_reopen_max (float, 3.0),
    provenance_gain_v_ref (float, 1e-2),
    provenance_gain_global_scale (float, 1.0),
    cross_module_consolidation_record_trace (bool, default False).
    The gain requires use_replay_precision_provenance AND
    use_sleep_world_forward_consolidation.
  Data flow:
    compute_e2_world_loss draws idx -> packets[i+1] (the SD-PP-3 +1 alignment:
    the training triple at replay index i is (world[i], action[i+1],
    world[i+1])) -> compute_provenance_gains(pi_cur = SD-PP-2
    current_read().pi_epi at sleep entry) -> weighted_row_loss over
    world_forward_contrastive_loss(..., reduction="none") -> the module's
    consolidate(module_step_scale={"e2_world": ...}) lr scaling -> per-step
    parameter DISPLACEMENT. Replay content, order and count are untouched: the
    same randperm draw, the same K rows, the same 8 steps.
  Why the gain acts in TWO places (fresh Adam normalises magnitude, MECH-572):
    consolidate() builds a FRESH torch.optim.Adam per module per call, so its
    bias-corrected step is ~lr * sign(g) REGARDLESS of |g|. A per-transition
    loss weight is therefore normalised AWAY in magnitude -- it changes only
    the DIRECTION of the step, which is exactly the phenotype MECH-572
    measured (displacement pinned at the Adam bound in 6/6 cells). So (a) the
    per-row weights sum_i g_i l_i / sum_i g_i set the direction, and (b) the
    module's lr for that step is scaled by mean_i g_i, which is what moves the
    displacement. Both legs are pre-registered. Measured liveness (pinned seed,
    3 steps, lr 1e-3, contract test T10a): max|delta| = 3.000e-4 / 1.500e-3 /
    3.001e-3 / 6.004e-3 at scales 0.1 / 0.5 / 1.0 / 2.0 -- linear in the scale,
    and scale 1.0 is bitwise identical to passing no kwarg at all.
  Rule (constants provisional until the preregistration's freeze record). For
  row i with packet p and current global epistemic precision pi_cur:
    K_i    = p.evidence_precision_z / (p.evidence_precision_z + pi_cur)
             Kalman-form write authority in (0,1): reliable evidence against
             the belief currently held.
    m_i    = sqrt( max(p.pe - noise_gain * p.evidence_variance_z, 0) / v_ref )
             the epistemic innovation MAGNITUDE -- precisely the information a
             fresh Adam discards.
    r_i    = min( reopen_max, 1 + surprise_beta * max(0, ln(p.surprise)) )
             the reopen factor. HISTORICAL model precision enters ONLY here,
             and only as an interpreter of the surprise. Surprise <= 0 or
             non-finite is read as r = 1.
    gain_i = clip( gain_max * K_i * m_i * r_i, gain_min, gain_max )
    ANTI-SELF-SEALING is structural: r_i >= 1 always, so a confidently-held
    prediction that is reliably falsified is never PROTECTED by having been
    confident. Historical precision can never appear as a multiplier below 1.
  Four modes:
    "provenance"        — the rule above (ARM C).
    "provenance_nohist" — the rule with r_i = 1 (ARM C-nohist; intake F1 asks
                          whether historical precision is load-bearing).
    "residual_only"     — g_i = global_scale * l_i / mean_j(l_j) from the
                          detached CURRENT per-row loss, unclipped (ARM
                          D-residual: current residual only, matched budget).
    "global"            — g_i = global_scale (ARM D-global: matched budget
                          carrying no per-row information).
    A row whose packet is missing (None, has_prev False, or a nan pe /
    evidence field) gets g_i = 1.0 and is counted in n_missing, in every mode
    except "global", which never inspects a packet at all.
  Diagnostics: gain_mean, gain_min, gain_max, gain_sd (POPULATION sd, so K=1
    reports 0.0 not nan), n_missing, k_mean, m_mean, r_mean, r_max,
    surprise_max, pi_cur, mode (float index into GAIN_MODES). The K/m/r
    readouts are over the NON-MISSING rows and are nan when there are none.
  Backward compatible: default OFF is bit-identical by STRUCTURAL ABSENCE — no
    ProvenanceGainConfig is built, no gain is computed, consolidate() receives
    neither new kwarg (so no callable is consulted, no trace entry is built and
    no metric key is added), and world_forward_contrastive_loss is called with
    its default reduction. Pinned by contract test T10b: the no-kwargs metrics
    dict has exactly the pre-existing keys and scale 1.0 is a numerical no-op.
    No RNG is drawn anywhere in this module, and no checkpoint or serialisation
    format changes (packets are runtime-only buffers, never saved).
  MECH-094: the gain consumer is a WEIGHT-update pass inside the SAME explicit
    exception CrossModuleConsolidator already relies on. It writes no residue,
    anchor or memory content, produces no hypothesis-tagged content, and the
    optimisers stay scoped to the named modules' parameters.
  Biological basis: precision-weighted plasticity gain — the Kalman-form K_i is
    the confidence-weighted update of Meyniel & Dehaene 2017 (human inference
    weights new evidence by its reliability relative to the precision of the
    belief already held), and the reopen factor is the corresponding refusal to
    let a high-confidence prior immunise itself against reliable contradiction.
    Sleep is treated as a SEPARATE gain regime rather than a replay of waking
    plasticity (Swift 2018 — neuromodulatory state during sleep sets a distinct
    plasticity gain; Feher 2026). The two-place application is forced by the
    optimiser, not by the biology: see the MECH-572 paragraph above.
  Phased training: none. SD-PP-4 trains nothing — it only weights an existing
    loss and scales an existing learning rate.
  Validation experiment: V3-EXQ-1073 (reserved). Preregistration
    REE_assembly/evidence/planning/precision_provenance_consolidation_gain_design_20260922.md;
    arms C / C-nohist / D-residual / D-global map onto the four modes above.
  Tests: tests/contracts/test_sdpp4_provenance_gain.py (15 tests covering the
    spec's eleven contracts: bounds; monotone in evidence precision;
    anti-self-sealing; noisy contradiction floored; control-mode budgets;
    nohist <= provenance rowwise; missing-packet handling; config validation;
    weighted_row_loss gradient path; consolidator liveness + OFF-path
    invariance + trace + the after-the-closure ordering contract;
    reduction="none"/"mean" agreement).
  MEASURED CAVEAT (darwin-arm64, torch 2.10.0): reduction="none" meaned by
    torch.mean is NOT bitwise equal to reduction="mean" — F.cross_entropy's
    fused mean sums in a different order. Agreement is ~7e-8 RELATIVE (2.4e-7
    absolute on a CE of ~3.64). The contract is stated as a relative tolerance.
    Per the cross-machine class-contract rule, do not tighten this to a bitwise
    assertion on any machine class.
  See MECH-572, MECH-574, MECH-016, ARC-055, MECH-043, MECH-368/431, ARC-137.


### Amendment 2026-09-22 (pre-freeze, V3-EXQ-1073 freeze record item 6): `residual_only` redesigned
- `residual_only` now reads `per_row_residual` (the CURRENT per-row mean-squared residual of
  `e2.world_forward` on the replayed triple, computed under `no_grad` inside
  `agent.compute_e2_world_loss` and passed as a new keyword to `compute_provenance_gains`) and
  schedules `g_i = clip(gain_max * sqrt(pe_cur_i / v_ref), gain_min, gain_max)` -- C's magnitude
  factor with K = 1, r = 1 and the current rather than the stored innovation. No packet, no
  precision term, `global_scale` ignored. The earlier budget-matched form (`global_scale * l_i /
  mean(l)`) inherited a pooled global budget and could not reallocate across epistemic regimes, so
  it was not the current-residual rival the design needs. Test T5 pins the new form; the `global`
  mode is unchanged and remains the matched-total-budget rival (ARM D-global).
