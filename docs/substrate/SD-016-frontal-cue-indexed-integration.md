## SD-016: Frontal Cue-Indexed Integration (2026-04-16)
- SD-016: e1.frontal_cue_indexed_integration -- IMPLEMENTED 2026-04-16.
  Module: ree_core/predictors/e1_deep.py (E1DeepPredictor).
  Three new projections gated by sd016_enabled=True:
    world_query_proj: Linear(world_dim=32, hidden_dim=128) -- z_world-only ContextMemory query
    cue_action_proj:  Linear(latent_dim=64, action_object_dim=16) -- affordance bias for E2
    cue_terrain_proj: Linear(latent_dim=64, 2) -- (w_harm, w_goal) terrain precision weights for E3
  Entry point: E1DeepPredictor.extract_cue_context(z_world) -> (action_bias, terrain_weight).
  Config: E1Config.sd016_enabled (default False; backward compatible).
  Data flow: z_world -> world_query_proj -> ContextMemory attention -> cue_action_proj (affordance)
             and cue_terrain_proj (terrain precision). terrain_weight passed to E3; action_bias to E2.
  Training for cue_terrain_proj: supervised terrain_loss using hazard_field_view proxy (lambda=0.1).
    terrain_loss must be included in experiment E1 training loops to train this projection.
    Pattern: see EXQ-182, EXQ-187a, EXQ-194. Omitting terrain_loss leaves cue_terrain_proj random.
  Training for cue_action_proj: the original claim "implicit via E3 trajectory selection
    gradient (no new loss)" is DEMONSTRABLY FALSE. V3-EXQ-449 (diagnostic probe,
    2026-04-20) confirmed cue_action_proj.weight receives exactly 0.0 gradient under this
    path (C1 PASS, 2 seeds, ~1.7k steps) because the CEM argmax in HippocampalModule is
    non-differentiable and agent.py:694 detaches action_bias before rollouts. EXQ-449 C2
    arm added a supervised MSE loss against E2.action_object(z_world, a_executed).detach():
    weights trained (grad ~0.013, delta ~0.21) but action_bias_divergence stayed at exactly
    0.0 in both seeds.
  EXP-0155: RESOLVED 2026-08-18 -- ITS PREMISE WAS FALSE, and this is the load-bearing
    correction (chip-20260818-exp0155-action-bias-scoring-disconnect; source-traced +
    measured; contract tests/contracts/test_exp0155_action_bias_no_scoring_authority.py,
    ree-v3 a1d49706).
    EXP-0155 was queued to find "the specific blocker DOWNSTREAM of cue_action_proj that
    zeroes the signal before it reaches E3.select". There is no such blocker, because
    `action_bias_divergence` NEVER LOOKS DOWNSTREAM: `_action_bias_divergence` (see
    experiments/v3_exq_449_sd016_cue_action_proj_wiring_probe.py:180) is computed directly
    on `e1.extract_cue_context(z_world)[0]` -- the E1 OUTPUT -- with zero calls into
    HippocampalModule, E2 rollout or E3.select. Nothing downstream can move it.
    The exact 0.0 is an UPSTREAM degeneracy: under the uniform-softmax ContextMemory
    attention saddle `cue_context` is constant in z_world, so the pre-EXQ-449a
    `cue_action_proj(cue_context)` returned one vector for every context -> cosine
    similarity exactly 1.0 -> divergence exactly 0.0. Measured at fresh init 2026-08-18:
    slot-selection entropy 2.7726 == ln(16) (exactly uniform), divergence of cue_context
    itself 0.000e+00. That is the subject of
    chip-20260816-implsub-contextmemory-writepath-degeneracy, NOT a forward-path defect.
    The `torch.cat([cue_context, z_world])` line in extract_cue_context is the EXQ-449a
    partial fix for exactly this ("cue_context is constant under uniform attention").
  SEPARATE, GENUINE finding from the same adjudication -- action_bias has NO
    trajectory-RANKING authority. Nothing on the ranking path reads an action-object:
    `o_t` lands only in `Trajectory.action_objects`; `world_forward`/`predict_next_self`
    never read it; `_score_trajectory` scores `get_world_state_sequence()` (terrain +
    optional wanting/curiosity/MECH-267 mode terms) and never touches action_objects;
    `_trajectory_first_action_class` (and thus the SP-CEM elite path) and the MECH-294
    theta packet both read `.actions`; `e3_selector.py` has ZERO occurrences of
    action_object/action_bias. Across ree_core, `action_objects` appears in exactly two
    files (e2_fast.py producer, hippocampal/module.py refit + storage copies).
    Its ONLY channel is the CEM refit, where -- scores and hence softmax weights and elite
    membership being bias-blind, and the bias being the same constant on every candidate
    and horizon step -- it is an exact additive translation: ao_mean += b, ao_std unchanged,
    in BOTH the legacy elite-mean and differentiable-softmax branches (so it never
    compounds -- always exactly one b). Measured: candidate scores BIT-IDENTICAL at b=None
    vs b=1e3 (incl. all scoring extensions on), elite ordering unchanged, and in a
    matched-RNG pass the first differing scoring call is exactly the iteration-1 boundary.
    At num_cem_iterations=1 the bias is bit-identically inert end-to-end.
    DO NOT over-read this: "no ranking authority" is NOT "no effect". The rigid translation
    passes through the nonlinear action_object_decoder and genuinely moves the pool -- a
    large bias flips the whole final candidate set's first-action class. It steers WHERE
    proposals are drawn, never WHICH proposal wins. This also explains why SD-055's
    differentiable CEM restored an action-sequence gradient (V3-EXQ-568 grad_max=372)
    without behavioural divergence.
    MECH-151's registered notes ("action-objects consistent with the cue are ELEVATED,
    contextually inappropriate ones SUPPRESSED ... E1 biases which action-objects RANK
    HIGHLY, but E3 still selects") are therefore NOT satisfied by the substrate as built --
    governance flag raised on MECH-151.
    REPAIR SITE IS E3, NOT `_score_trajectory`. Q-020/ARC-007 STRICT (module header):
    HippocampalModule generates VALUE-FLAT proposals, no value head, "E3 introduces ALL
    weighting". Teaching `_score_trajectory` to read action_objects would put E1-supplied
    weighting inside the hippocampal scorer -- forbidden. MECH-151's own wording is
    satisfiable only where E3 selects, and E3 has NO action-object input channel at all.
    That is the substrate gap; it is a design decision for /implement-substrate under
    ARC-007, not a local patch.
  GAP CLOSED (BUILD, NOT VALIDATION) 2026-09-01, GFLAG-0051 / ARC-007 option A
    (user-authorised: "build the channel, even if it is just to test the idea";
    chip-20260901-gflag0051-e3-action-object-channel, substrate_queue
    mech151-action-bias-has-no-e3-ranking-channel flipped off DO_NOT_BUILD_YET).
    E3 now has an action-object input channel:
    `ree_core/predictors/e3_selector.py::compute_action_object_alignment_bias(candidates,
    action_bias, weight)` computes, per candidate, the cosine similarity between that
    candidate's OWN `Trajectory.get_action_object_sequence()` (genuinely per-candidate,
    unlike the CEM proposal-mean shift above) and the SD-016 `action_bias` direction, and
    returns `-weight * alignment` (lower-is-better convention, so an ELEVATED/aligned
    candidate is favoured). REEAgent.select_action (agent.py, immediately before the
    orienting-decision `_do_bias` block folds into `dacc_score_bias`) composes this
    additively into the SAME score_bias chain every other E3 modulatory head uses, so the
    computation of weight from action-objects happens in E3's own module (ARC-007 STRICT
    satisfied) while composition follows the established agent.py convention. Gated by
    `config.e3.use_action_object_bias_channel` (default False) and
    `config.e3.action_object_bias_weight` (default 1.0) -- bit-identical OFF, so every
    existing experiment and the EXP-0155 finding above is unaffected until a driver opts in.
    Contract tests: `tests/contracts/test_gflag0051_action_object_bias_channel.py`.
    `tests/contracts/test_exp0155_action_bias_no_scoring_authority.py`'s
    `test_e3_selector_has_no_action_object_channel` was REWRITTEN (not deleted, per that
    file's own instructions) to pin the new channel's default-off/opt-in shape instead of
    the zero-occurrence finding, and its corpus-level consumer pin
    (`test_action_objects_have_no_consumer_outside_producer_and_hippocampus`) now includes
    `predictors/e3_selector.py` as an intentional third consumer.
    STILL OPEN, NOT DONE HERE: this is a BUILD, not a validated behavioural claim -- no
    experiment was queued (out of scope for this chip; report only). MECH-151's
    "RANK HIGHLY"/"ELEVATED/SUPPRESSED" wording is now IMPLEMENTABLE end-to-end, but
    whether it actually moves selection in a real rollout (and survives the still-live
    ContextMemory write-path degeneracy and the action_bias_divergence~=0.0 upstream
    issue below) is untested. Do not read this entry as resolving MECH-151's candidate
    status or the GFLAG-0051 governance flag -- flag resolution is /governance's call.
  Standing guidance for sd016_enabled=True experiments (UNCHANGED in force, now with a
    cause): expect action_bias_divergence ~= 0.0 until the ContextMemory write-path
    degeneracy is fixed, and do not rely on cue_action_proj for candidate-RANKING effects
    at all -- not even with use_differentiable_cem=True (SD-055 substrate-ready 2026-05-15;
    V3-EXQ-568 PASS grad_max=372 -- gradient barrier only, not behavioural divergence).
    (cue_terrain_proj path remains valid -- trained via terrain_loss.)
  Backward compatible: sd016_enabled=False by default; existing experiments unaffected.
  MECH-094: not applicable (waking encoder query, not replay content).
  Validation experiment: V3-EXQ-418a queued (SD-016+SD-017 combined retest with terrain_loss).
    V3-EXQ-418/418a/418b have all FAILed with action_bias_divergence=0.0. The EXQ-418b
    successor was GATED on EXP-0155; EXP-0155 resolved 2026-08-18 (premise refuted -- see
    above), so the gate now transfers to the UPSTREAM cause: the ContextMemory write-path
    degeneracy (chip-20260816-implsub-contextmemory-writepath-degeneracy). A successor
    written before that lands will reproduce action_bias_divergence=0.0 for the same
    reason, and -- independently -- must not be scored on candidate-RANKING divergence,
    which the substrate cannot produce at all (see the EXP-0155 block above).
  See MECH-150, MECH-151, MECH-152, ARC-041, INV-040, EXP-0155 (cue_action_proj diagnostic).
  Design doc: REE_assembly/docs/architecture/sd_016_frontal_cue_integration.md
