## ARC-062 Phase 1: Gated-Policy Heads + Context Discriminator (2026-05-09)
- ARC-062 (Phase 1, weak reading): rule_apprehension.gated_policy_heads --
  IMPLEMENTED 2026-05-09. Phase 1 of arc_062_rule_apprehension_plan.md
  (GAP-A). V3-tractable instantiation of the rule-apprehension architectural
  slot identified by MECH-309 (logical-necessity claim: trainers weight
  rules they do not invent; without a non-Bayesian rule-creator at the
  policy layer, gradient descent on a parametric policy collapses to the
  smoothest single regime good-enough across the whole state space).
  Module: ree_core/policy/gated_policy.py (GatedPolicy + GatedPolicyConfig
  + GatedPolicyOutput). New ree_core/policy/__init__.py package.
  Architecture (Phase 1, weak reading):
    Two scoring heads (head_0, head_1) sharing E3 candidate features [K,
      world_dim]. Each head: Linear(world_dim, head_hidden) -> ReLU ->
      Linear(head_hidden, 1) -> [K] scalar bias. Symmetry-broken init
      on the heads' last-Linear bias term (head_0 +offset, head_1
      -offset; default offset=0.05) so heads can differentiate from
      step 0 under any training pressure -- avoids the all-zero
      degenerate equilibrium where w*head_0 + (1-w)*head_1 collapses to
      a single value.
    Context discriminator: 3-stream input (z_world, z_self, z_harm_a)
      per Pull A SYNTHESIS verdict 1 (Miller & Cohen 2001 + Rigotti 2013
      + Mitchell 2016 macaque MD with insular cluster). Linear(world_dim
      + self_dim + harm_a_dim, disc_hidden=24) -> ReLU -> Linear(24, 1)
      -> sigmoid -> scalar w in [0, 1]. Discriminator weights scaled by
      disc_init_scale=0.1 at init so sigmoid output sits near 0.5 --
      avoids early head over-commitment before either head has
      differentiated.
    Gated bias: gated_score_bias = w * head_0(features) + (1 - w) *
      head_1(features), clamped to [-bias_scale, +bias_scale]
      (bias_scale=0.1; mirrors lateral_pfc_bias_scale so Phase 1
      magnitudes are comparable to existing PFC-side contributions).
  Pull A SYNTHESIS verdicts (resolved defaults; do NOT re-litigate):
    R1 multi-stream input (z_world, z_self, z_harm_a) -- single-stream
      z_world-only is the impoverished case, reserved for Phase 2
      ARM_1a/b/c input-ablation sub-arms.
    R2 N=2 heads at Phase 1 -- substrate-constrained by SD-054 reef-vs-
      forage two-mode partition. GatedPolicyConfig.n_heads != 2 raises
      ValueError on construction; multi-head extension is Phase 4 / GAP-E
      multi-strategy scaling probe.
    R3 score_bias level (option iii) -- engineering reasons dominate
      (SD-033a substrate is wired, gradient path through E3 score-
      aggregation is clean). FAIL chain at score_bias level routes
      discriminator to BG-side first then trajectory-proposal level
      then ARC-063 V4 strong reading.
  Config: REEConfig.use_gated_policy (bool, default False; bit-identical
  OFF). Sub-knobs (REEConfig + REEConfig.from_dims): gated_policy_n_heads
  (2), gated_policy_disc_hidden (24), gated_policy_disc_init_scale (0.1),
  gated_policy_head_hidden (32), gated_policy_bias_scale (0.1),
  gated_policy_head_init_bias_offset (0.05),
  gated_policy_use_first_action_onehot (False; see GAP-B below).
  Agent wiring (REEAgent.__init__): when use_gated_policy=True, instantiate
    GatedPolicy with (world_dim, self_dim, z_harm_a_dim) from
    config.latent. Phase 1 has NO connection to SD-033a LateralPFCAnalog
    -- that wiring is Phase 3 (closes commitment_closure GAP-1) per
    arc_062_rule_apprehension_plan.md. Per-episode reset() clears
    diagnostic counters (no persistent state to clear -- module is
    stateless across ticks).
  Data flow (REEAgent.select_action): immediately before the MECH-295
    block, compose gated_policy_score_bias additively into dacc_score_bias
    (parallel to the dACC / lateral_pfc / ofc / mech295 composition
    pattern). Per-candidate features = first-step z_world summary [K,
    world_dim] (reuses cand_world_summaries from lateral_pfc / ofc when
    they ran earlier this tick; builds fresh otherwise). simulation_mode
    parameter is False at this call site (waking action selection); the
    module's MECH-094 simulation_mode=True path (used by replay/DMN
    consumers if they ever call this module) returns (0.5, zeros[K])
    without advancing diagnostics.
  Backward compatible: use_gated_policy=False by default; agent.gated_policy
    is None and the entire select_action GatedPolicy block is skipped.
    249/249 preflight + contracts PASS (244 prior + 5 new) with master
    OFF. Bit-identical to baseline.
  Biological basis: Pull A SYNTHESIS (lit-pull A: lateral PFC rule-context
    modulation, 8 entries) -- Miller & Cohen 2001 PFC rule-as-bias
    foundational; Rigotti et al. 2013 mixed selectivity; Mitchell et al.
    2016 macaque MD network with insular cluster; Bongard & Nieder 2010
    PFC rule-coding units; Erez & Duncan 2015 MD adaptive coding;
    Capkova/Mansouri 2025 frontal lesion rule-value-learning dissociation.
    Pull A SYNTHESIS verdicts captured in
    REE_assembly/evidence/literature/targeted_review_arc_062_rule_apprehension/
    SYNTHESIS.md.
  MECH-094: simulation_mode argument on forward(); when True, returns
    (gating_weight=0.5, zeros[K], zeros[K], zeros[K]) and increments
    only the simulation-skip counter. Match SD-035 amygdala / MECH-279
    PAG simulation_mode pattern. Module has no internal state buffer
    (no EMA, no rule_state) -- stateless across ticks; reset() only
    clears diagnostic counters.
  Phased training: not required for Phase 1 substrate-readiness. The
    head + discriminator parameters develop training pressure naturally
    via E3 score-aggregation gradient when the validation experiment
    queues an end-to-end task. Phase 2 monomodal-collapse falsifier on
    SD-054 is the first behavioural training environment; Phase 1 only
    validates substrate wiring + architectural prerequisites for the
    Phase 2 falsifier. V3-EXQ-543f (GAP-B falsifier) requires phased
    training (P0 encoder warmup -> P1 frozen-encoder head training ->
    P2 eval).
  Validation experiment: V3-EXQ-542 5/5 PASS 2026-05-09T20:22:11Z (Mac
    runner; v3_exq_542_arc062_gated_policy_substrate_readiness_v3_*.json
    in REE_assembly/evidence/experiments/). Five sub-tests UC1-UC5:
    UC1 forward-pass instantiation; UC2 master-OFF no-op vs baseline E3;
    UC3 discriminator output varies with z_world (input sensitivity,
    threshold 0.001 in the Phase-1 init regime; substantial discriminator
    variation is a Phase-2 training signal); UC4 head differentiation
    under training pressure (>5x output divergence on held-out batch
    after 200 SGD steps); UC5 MECH-094 simulation_mode gate. Phase 2
    monomodal-collapse falsifier on SD-054 reef + hazard_food_attraction
    substrate is the next behavioural validation (queued as a separate
    session per the plan-of-record's six-phase sequencing -- do NOT
    bundle Phase 2 EXQ in the Phase 1 landing session).
  Contract tests: tests/contracts/test_gated_policy.py 6/6 PASS (C1
    default-off no-op; C2 backward-compat flag-on does not raise during
    construction or first sense() tick; C3 discriminator output in
    [0, 1] across 64 diverse latent states with bias_scale clamp
    respected; C4 heads' OUTPUTS diverge >5x on held-out batch after
    200 SGD steps under anti-symmetric loss; C5 simulation_mode=True
    returns zeros + increments skip counter only, subsequent waking
    call does not retroactively re-increment skip counter; C6
    use_first_action_onehot: correct head_in_dim, output shape, differs
    from base, sim-mode still zeros, _last_onehot_was_none diagnostic).
  Plan-of-record: REE_assembly/evidence/planning/arc_062_rule_apprehension_plan.md
    GAP-A status open -> done 2026-05-09; owner_exq=V3-EXQ-542; Phase 2
    GAP-B in-progress pending V3-EXQ-543f.
  GAP-B: ARC-062 head-input first-action one-hot augmentation (option 2)
    IMPLEMENTED 2026-05-17. Root cause from EXQ-543e autopsy: SP-CEM
    delivers ~5 distinct first-action classes but E2 world-forward
    compresses them to 0.22% of z_world magnitude before reaching the
    z_world-only GatedPolicy heads -- the heads are under-fed. Fix:
    bypass E2 compression by concatenating the first-action one-hot
    directly onto the head's candidate_features input.
    Config: REEConfig.gated_policy_use_first_action_onehot (bool, default
      False; bit-identical OFF when False). GatedPolicyConfig gains
      use_first_action_onehot (bool, False) and first_action_dim (int, 0;
      set to config.e2.action_dim by REEAgent.__init__ -- single source
      of truth, not a separate REEConfig knob).
    Data flow: c.actions[:, 0, :][0] -> [action_dim] one-hot per
      candidate; stacked to [K, action_dim]; cat with gp_summaries
      [K, world_dim] -> augmented head input [K, world_dim+action_dim].
      Discriminator input (z_world, z_self, z_harm_a) UNCHANGED.
    Backward compatible: gated_policy_use_first_action_onehot=False
      (default); head Linear input stays [K, world_dim]; select_action
      sets first_action_onehots=None; forward() skips the cat. 484/484
      contracts+preflight PASS with defaults unchanged.
    Phased training required for V3-EXQ-543f (P0 -> P1 -> P2).
    Validation: V3-EXQ-543f to be queued via /queue-experiment (supersedes
      V3-EXQ-543e; same 2x2 SP-CEM/dACC factorial; also requires
      dacc_weight>0 + pre-flight non-degeneracy assertion per 543e
      autopsy addendum).
  Cross-plan link: commitment_closure_plan.md GAP-1 (SD-033a bias-head
    training) unblocked at substrate level by GAP-C + GAP-D (below).
    Validation EXQ for GAP-1 deferred until V3-EXQ-543f returns a
    contributory result (GAP-B scientific gate).
  See ARC-062 (this claim, weak reading), MECH-309 (logical-necessity
    diagnostic the substrate addresses), ARC-063 (V4 strong reading,
    distributed CandidateRule field; deferred via Phase 4 / GAP-E
    multi-strategy scaling probe), SD-033a (Phase 3 downstream consumer
    -- not wired in Phase 1), MECH-262 (rule-selective persistence;
    Phase 3 work), SD-029 (monomodal-collapse measurement gate; Phase 2
    falsifier dependent variable), SD-054 (reef + hazard_food_attraction
    substrate; Phase 2 falsifier environment), MECH-094 (simulation_mode
    argument), Pull A SYNTHESIS verdicts R1 / R2 / R3, Pull B SYNTHESIS
    R4 verdict (Phase 2 acceptance criteria).
  GAP-C: ARC-062 discriminator output -> SD-033a rule_state source vector --
    IMPLEMENTED 2026-05-17.
    Changes: LateralPFCConfig gains use_discriminator_source (bool, default
      False), discriminator_pool_weight (float, default 0.3);
      LateralPFCAnalog gains discriminator_proj (nn.Linear(1, rule_dim))
      and accepts optional disc_output arg in update(). REEConfig gains
      lateral_pfc_use_discriminator_source and lateral_pfc_discriminator_pool_weight.
    agent.py reorder: gated_policy block now runs BEFORE lateral_pfc block
      so gp_output.gating_weight is available as disc_output. Score-bias
      composition is additive -- reorder is value-identical to prior ordering.
      gated_policy block sets cand_world_summaries = gp_summaries so
      lateral_pfc / ofc / mech295 blocks reuse rather than rebuild.
    Data flow: GatedPolicy.gating_weight scalar -> tensor([[w]]) [1,1]
      -> discriminator_proj -> [1, rule_dim] added to source with weight
      discriminator_pool_weight. Gated by use_discriminator_source (default
      False = no-op, bit-identical backward compat).
    Backward compatible: use_discriminator_source=False by default; update()
      disc_output=None arg is no-op. 484/484 contracts PASS; 543f dry-run
      exit 0. MECH-094: disc_output is detached (gated_policy under no_grad);
      existing MECH-319 lateral_pfc skip gate unchanged.
  GAP-D: SD-033a rule_bias_head trainable -- IMPLEMENTED 2026-05-17.
    Changes: LateralPFCConfig gains train_rule_bias_head (bool, default False);
      when True, last Linear is NOT zeroed at init (random init preserved).
      LateralPFCAnalog gains bias_head_parameters() method returning
      self.rule_bias_head.parameters() for optimizer inclusion. REEConfig
      gains lateral_pfc_train_rule_bias_head (bool, default False).
    Gradient path: E3 loss -> score_bias -> compute_bias() -> rule_bias_head
      weights. No separate loss term needed. Starting from zero-init (default
      False) OR from random-init (True), gradient flows from tick 1 of P1.
    Experiment use: in P1 optimizer, add:
        list(agent.lateral_pfc.bias_head_parameters())
    Backward compatible: train_rule_bias_head=False by default; last Linear
      remains zeroed (bias output stays 0.0); behavior bit-identical.
    Phased training: bias head can join P1 optimizer alongside gated_policy
      heads (same E3 gradient path). No separate warmup protocol needed.
    Validation EXQ (commitment_closure:GAP-1): 2-arm ablation (head frozen
      vs trainable) deferred until V3-EXQ-543f contributory result.
