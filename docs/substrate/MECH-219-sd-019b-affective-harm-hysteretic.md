## MECH-219 (SD-019b): affective-harm hysteretic integrator (z_harm_suffering) (2026-06-10)
- MECH-219: affect.affective_harm_hysteretic_integration -- IMPLEMENTED 2026-06-10
  (substrate; SD-019b stays v3_pending until the validation EXQ PASSes). The
  tier-2 -> tier-3 step of the harm-affect hierarchy: SD-019a builds z_harm_un
  (symmetric EMA of z_harm_s, "make it stop" unpleasantness); MECH-219 turns it
  into a slow, persistent, controllability-gated SUFFERING load state
  (z_harm_suffering) -- the mechanism SD-019b (harm_stream.suffering_accumulator)
  names and is blocked on. Built from the landed design-first plan-of-record
  REE_assembly/evidence/planning/mech_219_hysteretic_integrator_design.md (Section 10
  checklist).
  Module: ree_core/affect/harm_suffering_accumulator.py (HarmSufferingAccumulator +
  HarmSufferingAccumulatorConfig + HarmSufferingAccumulatorOutput). Pure-arithmetic
  regulator (no nn.Module, no learned parameters, no gradient flow); sibling to the
  MECH-353 BlockedAgency / MECH-313 / MECH-320 / MECH-342 regulator pattern. Owns
  only the scalar suffering state s_t; the agent builds the z_harm_suffering vector.
  Dynamics (memo Section 4): u_t = ||z_harm_un|| (+ body_damage_weight*||z_harm_a||);
  g_t = 1 - escapability; drive_t = g_t*u_t (+ pe_gain*unsigned_PE); asymmetric
  accumulation s_t = s_{t-1} + alpha*(drive_t - s_{t-1}) with alpha=alpha_rise when
  building else alpha_fall (alpha_rise >> alpha_fall = the hysteresis); optional
  Schmitt bistable latch (theta_on/theta_off). Output z_harm_suffering = the
  z_harm_un direction scaled to magnitude s_t (same dim as z_harm_un).
  Escapability source PLUGGABLE (harm_suffering_escapability_mode): constant
  (default 1.0 -> g=0 -> inert/bit-identical) / avoidance_efficacy (reads SD-058
  InstrumentalAvoidanceGate.effective_efficacy() -- the literal escapability
  construct; soft dependency on the v3_pending SD-058 substrate) / external
  (REEAgent.set_harm_suffering_escapability() seam for the validation experiment).
  NEVER sourced from MECH-353 capacity_belief (= 1 - w*||z_harm_a||) to avoid the
  z_harm_a -> capacity_belief -> z_harm_a loop; capacity_belief is a validation
  cross-check only (memo Section 3 / R1).
  Module: ree_core/latent/stack.py -- LatentState.z_harm_suffering [batch, harm_dim]
  + detach() handling. ree_core/utils/config.py -- REEConfig fields +
  REEConfig.from_dims passthrough (use_harm_suffering_accumulator + alpha_rise/fall +
  escapability_mode/constant/external + s_cap + body_damage_weight + pe_gain +
  use_bistable_latch + theta_on/off + 5 per-consumer redirect flags).
  Agent wiring (ree_core/agent.py): self.harm_suffering_accumulator built when the
  master flag is on (PRECONDITION: requires use_harm_un=True -- loud ValueError
  otherwise, since z_harm_un is the drive input); ticked in sense() right after the
  SD-019a z_harm_un EMA and BEFORE the SD-032 consumers (AIC in sense; PAG/pACC in
  select_action) so a redirect reads the suffering output same-tick; z_harm_suffering
  vector built from the regulator's s_t; reset() clears it per episode;
  _resolve_harm_suffering_escapability() resolves the escapability scalar per mode;
  set_harm_suffering_escapability() drives the external mode.
  z_harm_a re-source migration (memo Section 6): use_harm_suffering_accumulator is the
  master flag; PER-CONSUMER redirect flags (harm_suffering_redirect_{aic,pag,mech091,
  dacc,pacc}) let the migration be staged + individually ablated. Redirects are
  MAGNITUDE-based (||z_harm_suffering||): v1 WIRES the urgency/PAG/interrupt consumers
  (AIC urgency aic_z_norm; PAG MECH-279 freeze drive pag_z_norm; MECH-091 urgency-
  interrupt _urgency_signal) per memo R3. The dACC/pACC flags are DEFINED but left
  UNWIRED (default off, currently no-ops): their E2_harm_a forward models are keyed on
  the current z_harm_a dim (z_harm_a_dim != harm_dim), so they migrate last after
  measuring R^2 -- v1 keeps them on legacy z_harm_a (memo R3). Body-damage fold-in
  (memo Section 6 fork b): body_damage_weight (default 0.0) folds ||z_harm_a|| into the
  drive so SD-022 / EXQ-319 / EXQ-323a non-redundancy evidence is preserved, not
  orphaned.
  Backward compatible: use_harm_suffering_accumulator=False by default ->
  agent.harm_suffering_accumulator is None, LatentState.z_harm_suffering stays None,
  no consumer redirect fires -> bit-identical (default == explicit-False action stream
  verified). Also inert under the default escapability_mode=constant=1.0 (g=0 -> s->0)
  even when explicitly enabled. 7/7 preflight + full contract suite green (the one
  scaffolded_sd054 C11 reef-spawn failure is a pre-existing env-OS-entropy flake --
  _build_env passes no seed to CausalGridWorldV2(np.random.default_rng(None)) -- ~10%
  failure rate on clean HEAD too; unrelated to MECH-219). 11/11 new contracts in
  tests/contracts/test_mech_219_harm_suffering_accumulator.py.
  Activation smoke (2026-06-10): constant esc=1.0 -> s=0 even at u=1.2 (inert OFF);
  external esc=0.0 (inescapable) -> suffering rises fast (alpha_rise=0.2) to plateau;
  then esc=1.0 (relief) -> SLOW fall (alpha_fall=0.01, after 40 steps still > half-peak
  = hysteresis); body-damage fold-in u=2.0 at body_damage_weight=0.5; bistable latch
  flips at theta_on/off; avoidance_efficacy reads SD-058 effective_efficacy; AIC
  redirect tracks the suffering channel.
  Phased training: N/A (pure-arithmetic regulator; no learned parameters). MECH-094:
  update() no-op under simulation_mode (hypothesis_tag), so replay/DMN ticks do not
  accumulate suffering on imagined outcomes.
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default flag; every existing
  experiment uses the default (accumulator off), so no dependent claim's measured
  mechanism changed. KEEP all evidence.
  Validation experiment: NOT queued in this build session (separate /queue-experiment
  session per the memo Section 7 sketch -- the controllability dissociation falsifier
  C1/C2/C3/C4 under escapable vs inescapable matched-nociception, driving escapability
  from SD-058 effective_efficacy() or an external scripted schedule). PASS clears the
  SD-019b / Q-036 gate.
  Design doc: REE_assembly/docs/architecture/mech_219_hysteretic_integrator.md
  (status flipped plan-of-record -> IMPLEMENTED 2026-06-10). Plan-of-record:
  REE_assembly/evidence/planning/mech_219_hysteretic_integrator_design.md.
  See MECH-219, SD-019b (the suffering_accumulator claim this builds), SD-019a
  (z_harm_un, the tier-2 input + the redirect precedent mirrored), SD-019 (parent
  nonredundancy), SD-011 (z_harm_s / z_harm_a streams), SD-021 (descending modulation;
  controllability parity preserved -- escapability is NOT the attenuation factor),
  SD-022 (body-damage; folded into the drive), SD-058 / MECH-357 (escapability source;
  soft depends_on), MECH-353 (blocked-agency; opposite controllability pole, anti-
  correlation cross-check), SD-032b dACC / SD-032e pACC (z_harm_a forward-model
  consumers; migrate last), MECH-279 PAG / SD-032c AIC / MECH-091 (v1 redirect
  consumers), Q-036 (variable adjudication; resolved-by-design pending validation),
  MECH-094 (simulation gate).
