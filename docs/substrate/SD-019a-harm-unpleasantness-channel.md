## SD-019a: harm_unpleasantness_channel (2026-05-04)
- SD-019a: harm_stream.immediate_affective_valence -- IMPLEMENTED 2026-05-04.
  Module: ree_core/agent.py (REEAgent.__init__, reset(), sense(), select_action()),
    ree_core/latent/stack.py (LatentState.z_harm_un), ree_core/utils/config.py
    (LatentStackConfig.use_harm_un, harm_un_ema_alpha).
  Config: LatentStackConfig.use_harm_un (bool, default False; set True to enable).
    LatentStackConfig.harm_un_ema_alpha (float, default 0.2; ~5-step rise to
    z_harm_s=1.0 at alpha=0.2).
  State: REEAgent._harm_un_ema (Optional[Tensor], same dim as z_harm_s); reset
    per episode; seeded at z_harm_s on first tick; None when feature is OFF.
  Data flow: sense() -> LatentStack.encode() -> new_latent.z_harm (z_harm_s) ->
    [if use_harm_un and not hypothesis_tag] EMA update: _harm_un_ema <-
    (1-alpha)*_harm_un_ema + alpha*z_harm -> new_latent.z_harm_un <-
    _harm_un_ema.clone() -> AIC urgency (aic_z_norm reads z_harm_un.norm() when
    use_harm_un; falls back to z_harm_a.norm() otherwise) + select_action() E3
    short-horizon urgency_weight and MECH-091 interrupt (redirect z_harm_a
    variable to z_harm_un when use_harm_un).
  Three-tier harm hierarchy: z_harm_s (fast, instantaneous) ->
    [EMA alpha=0.2] -> z_harm_un (medium, ~5-step rise) ->
    [MECH-219/SD-019b, not yet implemented] -> z_harm_a (slow, suffering).
  Controllability parity (Loffler 2018 key constraint): SD-021 descending
    modulation only multiplies new_latent.z_harm (z_harm_s). z_harm_un evolves
    exclusively via its own EMA rule and is NOT attenuated during commitment.
    Verified: UC3 in V3-EXQ-518 (norm_un unchanged during commitment, norm_s
    attenuated by 50%).
  AIC redirect (SD-032c): when use_harm_un=True, aic_z_norm reads
    z_harm_un.norm() so the AIC urgency computation tracks the medium-timescale
    unpleasantness channel, not the slow suffering accumulator.
  E3 redirect: z_harm_a variable in select_action() is shadowed by z_harm_un
    when use_harm_un=True and z_harm_un is not None. Affects urgency_weight
    computation and MECH-091 interrupt threshold. dACC (SD-032b), pACC
    (SD-032e) retain independent z_harm_a reads (NOT redirected -- they
    correctly consume the slow suffering accumulator).
  MECH-094 compliance: EMA update gated on hypothesis_tag=False; replay/
    simulation paths do not advance _harm_un_ema.
  Backward compatible: use_harm_un=False by default; z_harm_un=None on
    every LatentState; all integration sites no-op. 184/184 contracts PASS
    with flag OFF.
  Biological basis: Loffler et al. 2018 (Pain Reports) three-way dissociation
    (intensity / unpleasantness / suffering): controllability selectively reduces
    suffering (z_harm_a) without touching unpleasantness (z_harm_un). ACC /
    anterior insula medial-pathway affective-motivational component of pain
    (Price 2000 dual-pathway; Rainville 1997 ACC/S1 double dissociation).
  Phased training: not applicable (non-trainable EMA buffer; no gradient flow).
  Validation experiment: V3-EXQ-518 queued (4-arm diagnostic, 9 acceptance
    criteria UC0a-b/UC1a-d/UC2a-b/UC3; dry-run PASS 2026-05-04).
  See SD-019a, SD-019 (parent nonredundancy), SD-011 (z_harm_s source),
    SD-019b (SD-019a is the input to the MECH-219 hysteretic integrator),
    SD-021 (descending mod; controllability parity), SD-032c (AIC redirect
    consumer), MECH-091 (urgency interrupt; redirect consumer), MECH-094
    (EMA gate).
