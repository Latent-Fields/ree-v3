## MECH-423 readiness substrate: R2 iterative-inference convergence + R3 interleaved cross-module consolidation + R1 shared-latent grad probe (2026-06-12)
- MECH-423 super-additivity readiness instrumentation -- IMPLEMENTED 2026-06-12
  (substrate readouts; MECH-423 stays candidate / v3_pending -- this PROMOTES
  NOTHING, it only unblocks EXP-0380 from blocked_substrate). Routed by the
  2026-06-12 /queue-experiment Step-2.5 substrate gate that found the R1/R2/R3
  readiness readouts the EXP-0380 acceptance_checks require ABSENT from the live
  ree-v3 eval path (so the super-additivity ablation would self-route
  substrate_not_ready_requeue every run = vacuous probe). Three capabilities, all
  no-op-default + bit-identical OFF + contract-tested; MECH-094 simulation-gating
  on the replay/consolidation path; pure-arithmetic readouts. Amends the ARC-004
  inference machinery (R2) + MECH-121 consolidation cluster (R3) -- implementation_note
  only, NO new claim minted.
  R2 (iterative-inference convergence; ARC-004): ree_core/latent/stack.py
    (LatentStack.encode) + ree_core/utils/config.py (LatentStackConfig). The legacy
    encode() is a FIXED two-pass amortized recognition (bottom-up init -> ONE
    top-down round): no settling loop, so the per-inference-step ||delta z_shared||
    the EXP-0380 R2 check needs had no source. When use_iterative_inference=True the
    single top-down round is generalised into a predictive-coding settling loop over
    the shared z_beta -> z_theta -> z_delta stack -- iterate the recurrent top-down
    map with the bottom-up data terms (combined_init/z_beta_init/body_obs/world_obs)
    held fixed -- run BEFORE the SD-007 reafference correction + EMA, tracking per-round
    ||delta z_shared|| / (||z_shared|| + eps) with early-stop at rel_tol. Readout:
    LatentState.inference_convergence (plain-float dict per_step_rel_delta[]/converged/
    n_iters/final_rel_delta), cached agent.last_inference_convergence; detach()
    passes it through. Config (LatentStackConfig, no-op): use_iterative_inference
    (False), inference_settle_iters (1), inference_convergence_rel_tol (0.05),
    surfaced via REEConfig.from_dims kwargs.pop (signature-stable). Grounds Gershman
    & Goodman 2014 (amortization gap). EXP-0380 R2 reads final_rel_delta < 0.05.
    Smoke: OFF bit-identical; ON settles 3 rounds 0.062 -> 0.0074 converged=True.
  R3 (module-tagged interleaved cross-module consolidation; MECH-121):
    NEW ree_core/sleep/cross_module_consolidation.py (CrossModuleConsolidator +
    CrossModuleConsolidatorConfig). The legacy MECH-121 offline pass trains
    e2_harm_s ALONE (sleep/phase_manager.py) over region-keyed traces with no module
    identity, so cross_module_replay_share is unmeasurable and the integrated E1<->E2
    representation cannot be acquired under interleaving. The consolidator takes named
    modules + loss closures + param lists, runs a configurable schedule, tags each
    replayed trace by which modules it ACTUALLY updated (a loss returning an
    exactly-zero sentinel == no replay content -> module not touched, so the share
    reflects genuine integration not the schedule label), and returns a flat readout
    {n_updates, n_traces, n_cross_module_traces, cross_module_replay_share, interleaved,
    updates_<name>}. "interleaved" runs one step per module per trace (a trace can
    touch >1 module -> share 1.0); "blocked" trains modules sequentially (each trace
    one module -> share 0.0, the catastrophic-interference control). Wiring: built on
    the agent (agent.cross_module_consolidator) when use_cross_module_consolidation=True
    (standalone-usable by the experiment) AND passed to SleepLoopManager, where a
    flag-gated hook in _run_cycle (AFTER the existing writeback, additive) runs the
    default E1 (compute_prediction_loss over _world_experience_buffer) + E2
    (compute_e2_loss over _e2_transition_buffer) loss set and merges
    cross_module_consolidation_* into the sleep-cycle metrics -- a readout of the LIVE
    MECH-121 pipeline. Config (REEConfig, no-op): use_cross_module_consolidation
    (False), cross_module_consolidation_schedule ("interleaved"), _steps (0 == none),
    _lr (1e-3), _batch (16), surfaced via from_dims kwargs.pop. MECH-094: the SAME
    explicit exception the e2_harm_s writeback uses -- per-module optimisers
    constructed LOCALLY over only the named modules' params; NO residue/anchor/memory
    write; simulation_mode returns the zeroed no-op (the offline call site passes
    False). Grounds McClelland 1995 + Kumaran 2016 CLS (interleaving necessary for
    shared-rep integration; blocked schedule -> catastrophic interference ->
    sub-additive ARTEFACT, so "blocked" is the pre-registered control not a bug).
    EXP-0380 R3 reads n_updates>0 AND cross_module_replay_share>0 AND interleaved==True.
    Smoke: OFF consolidator None; interleaved share 1.0 / 8 updates; blocked share 0.0;
    sim no-op; live force_cycle merges the keys share 1.0.
  R1 (shared-latent gradient probe; reusable utility): NEW
    ree_core/utils/shared_latent_probe.py (shared_latent_gradient_probe). Pure
    function (no substrate state, no hot-path touch): given z_shared + a
    {module: loss_fn(z_shared)} map, computes d(loss)/d(z_shared) per module via
    torch.autograd.grad(retain_graph=True) and returns {per_module_grad_norm,
    min_grad_norm, mean_pairwise_cosine, n_modules, coupled}. EXP-0380's integrated
    arm constructs z_shared (latent fed jointly to E1+E2) and reads the R1 verdict
    (min_grad_norm>0 AND mean_pairwise_cosine>=0). Grounds Yu/PCGrad 2020 (conflicting
    gradients = negative transfer) + Caruana 1997 (shared-rep MTL helps only when
    tasks related). Built per user decision 2026-06-12 (reusable hook in substrate).
  Backward compatible: all flags default no-op; full ree-v3 contract suite green with
    everything OFF (1013 prior PASS + 12 new MECH-423 contracts; the 1 control_vector
    C4 failure is the documented pre-existing baseline flake -- CONFIRMED still failing
    on a clean stash of these changes, PASSES with them in isolation). New contracts:
    tests/contracts/test_mech423_inference_convergence.py (C1 OFF bit-identical / C2 ON
    readout / C3 settling reduces delta / C4 settle_iters=1 == OFF / C5 detach passthrough)
    + tests/contracts/test_mech423_cross_module_consolidation.py (C1 OFF None / C2 config
    validation / C3 interleaved share 1.0 / C4 blocked share 0.0 / C5 single-module-data
    share 0.0 / C6 sim no-op / C7 SleepLoopManager hook merges the readout).
  Phased training: N/A (R2 is a settling-loop readout over the existing encoder, no new
    learned head; R3 reuses the already-trained E1/E2 module losses; R1 is a pure
    autograd probe). Evidence-staleness (Step 8.5): NOT triggered -- no-op-default
    flags; every existing experiment uses the defaults (no iterative inference, no
    cross-module consolidation), so no dependent claim's measured mechanism changed.
    KEEP all evidence.
  Validation experiment: V3-EXQ substrate-readiness diagnostic (claim_ids=[]; asserts
    R2 final_rel_delta < 0.05 + R3 interleaved share>0 vs blocked share=0 + R1 coupling)
    queued via /queue-experiment. PASS confirms the readiness readouts are non-vacuous
    on a trained substrate. EXP-0380 (the super-additivity ablation) is the SEPARATE
    /queue-experiment session this unblocks (flipped blocked_substrate -> proposed).
  Design doc: REE_assembly/docs/architecture/mech_423_superadditivity_readiness_substrate.md
  Lit-pull: REE_assembly/evidence/literature/targeted_review_mech_423_integration_prerequisites.
  See MECH-423 (the claim this unblocks; PROMOTES NOTHING), ARC-004 (shared L-space
    latent / inference machinery -- R2 amend), MECH-121 (NREM consolidation cluster --
    R3 amend), MECH-273 / SelfModelAggregator (the single-module e2_harm_s offline pass
    R3 generalises to interleaved E1<->E2), ARC-001/002 (E1/E2 streams), MECH-081/082/033
    (pairwise transfer paths super-additivity generalises), ARC-080 (object spine; triple
    arm), EXP-0380 (the super-additivity ablation), MECH-094 (call-site scoping).
