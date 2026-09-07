## MECH-269 Base Substrate -- Phase 1 (2026-04-22)
- MECH-269 base: hippocampal.per_stream_verisimilitude -- IMPLEMENTED 2026-04-22.
  Module: ree_core/hippocampal/module.py (HippocampalModule.update_per_stream_vs,
  HippocampalModule.reset_per_stream_vs, HippocampalModule._stream_value).
  Phase 1 of the V_s invalidation runtime (architecture doc:
  REE_assembly/docs/architecture/v_s_invalidation_runtime.md). Adds the
  observable per-stream verisimilitude foundation that Phase 2 (MECH-287
  broadcast invalidation trigger) and Phase 3 (MECH-284 staleness
  accumulator + MECH-269 anchor-reset hysteresis) will consume.
  Computation (Phase 1, identity-prediction proxy):
    For each registered stream s in config.per_stream_vs_streams:
      z_curr = LatentState[s] (or GoalState.z_goal for s=='z_goal',
                               LatentState.z_harm for s=='z_harm_s')
      err = ||z_curr - z_prev|| / (||z_curr|| + 1e-6)
      score = clip_[0,1](1 - err)
      V_s[s] = (1-tau)*V_s_prev[s] + tau*score   # EMA
    First observation seeds V_s[s] = 1.0 (perfect verisimilitude assumed)
    and caches z_curr; subsequent ticks compute the proxy.
  Forward-predictor routing (z_world -> ReafferencePredictor SD-007;
  z_harm_s -> HarmForwardModel SD-011) is RESERVED for Phase 2. Phase 1
  uses the identity proxy uniformly to keep HippocampalModule decoupled
  from per-stream predictor wiring; the dict is populated as an
  OBSERVABLE that downstream phases can consume.
  Config: HippocampalConfig.use_per_stream_vs (bool, default False),
  HippocampalConfig.per_stream_vs_tau (float, default 0.1),
  HippocampalConfig.per_stream_vs_streams (tuple, default
  ("z_world", "z_self", "z_harm_s", "z_harm_a", "z_goal", "z_beta")).
  Streams absent from the current LatentState / GoalState are silently
  skipped (no entry written to per_stream_vs).
  Agent wiring:
    REEAgent.sense() -- after new_latent.detach(), before return:
      if hippocampal.config.use_per_stream_vs:
        hippocampal.update_per_stream_vs(new_latent, goal_state=self.goal_state)
    REEAgent.reset() -- after MECH-279 PAG reset:
      if hippocampal.config.use_per_stream_vs:
        hippocampal.reset_per_stream_vs()
  Backward compatible: use_per_stream_vs=False by default; HippocampalModule
  exposes per_stream_vs={} and update_per_stream_vs() returns immediately.
  All 58 contract tests + 7 preflight tests pass with flag OFF
  (bit-identical to legacy).
  Activation smoke (2026-04-22, default agent + flag ON):
    Tick 1 (zero baseline obs): per_stream_vs = {z_world: 1.0,
      z_self: 1.0, z_beta: 1.0}. Streams z_harm_s/z_harm_a/z_goal absent
      because default REEConfig leaves harm streams and goal seeding off.
    Tick 2 (perturbed obs): per_stream_vs = {z_world: 0.958,
      z_self: 0.959, z_beta: 0.959} -- identity proxy responds to the
      change as designed.
    Reset: per_stream_vs = {} (cache cleared).
  No trainable parameters; pure arithmetic over latent norms. No
  phased training needed.
  MECH-094: hypothesis_tag is NOT yet checked in update_per_stream_vs()
  -- Phase 1 is invoked only from REEAgent.sense() (waking observation
  stream, never replay/simulation). Phase 2 will add the gate when
  replay paths begin to consume the V_s signal directly.
  Validation experiment: deferred to Phase 2/3 -- Phase 1 is substrate
  scaffolding for the MECH-287 trigger and MECH-284 staleness layers
  that follow. End-to-end EXQ-476 (re-run of EXQ-475 with full V_s
  invalidation circuit on) is the validation experiment for the
  combined cluster.
  Contract tests: tests/contracts/test_mech_269_per_stream_vs.py
    C1: default config backward-compat.
    C2: master switch OFF -> per_stream_vs stays empty.
    C3: master switch ON -> seeds at 1.0, drops on perturbation.
    C4: per-stream isolation -- a perturbation in one stream does not
        move other streams' V_s.
    C5: EMA correctness under repeated identical observations.
  Design doc: REE_assembly/docs/architecture/v_s_invalidation_runtime.md
  See MECH-269, MECH-272 (state-gated routing -- Phase 3), MECH-284
  (staleness accumulator -- Phase 3), MECH-287 (broadcast trigger --
  Phase 2), SD-007/MECH-101 (ReafferencePredictor -- Phase 2 z_world
  routing), SD-011 (HarmForwardModel -- Phase 2 z_harm_s routing),
  MECH-094 (hypothesis_tag gate -- Phase 2 when replay consumes V_s).
