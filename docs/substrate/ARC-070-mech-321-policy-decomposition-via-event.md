## ARC-070 / MECH-321: policy.decomposition_via_event_segmenter -- IMPLEMENTED (2026-07-24)
- ARC-070 / MECH-321: policy.decomposition_via_event_segmenter -- IMPLEMENTED 2026-07-24.
  ree_core/policy/policy_decomposition.py (new: DecompositionDecision, PolicyDecomposition,
  PolicyDecompositionConfig; pure decision logic, no learned parameters, mirrors
  ChunkAccumulator/ChunkLibrary in shape), ree_core/hippocampal/module.py
  (set_decomposition_source, _region_vs, _evaluate_decomposition_ticks,
  _recursive_leaf_tiles, _rollout_tile, _apply_policy_decomposition -- wired into
  propose_trajectories right after the ARC-071 chunk splice), ree_core/agent.py
  (instantiation with a loud precondition on hippocampal.use_event_segmenter,
  set_decomposition_source call, a mid-execution abort block beside the rung-6
  natural-commit-urgency release, get_policy_decomposition_state API).
  Config: REEConfig.use_policy_decomposition (default False; set True to enable),
  decomposition_vs_threshold 0.4 (matches this codebase's existing vs_gate_e1/e2_threshold
  convention), decomposition_depth_cap 3 (mirrors ARC-071's chunk_max_depth default so the
  inverse operations stay symmetric). NO hippocampal sub-config mirror -- unlike
  use_chunk_proposal_injection, HippocampalModule never reads use_policy_decomposition
  from its own config; it acts purely on whether set_decomposition_source() was called
  (same external-source-injection pattern as use_policy_chunking / _chunk_source), so the
  three-site REEConfig.from_dims hazard reduces to three sites here, not four.
  Data flow (pre-commit, R4 first phase): HippocampalModule.propose_trajectories builds
  ARC-071 chunk candidates as usual -> _apply_policy_decomposition sweeps up to 8 ticks of
  each candidate's own rolled-out states through PolicyDecomposition.evaluate(), which
  calls MECH-288 EventSegmenter.boundary_on(stream="rollout", ...) at each tick -> R1 OR
  trigger (region V_s -- mean of HippocampalModule.per_stream_vs, the SAME aggregate
  already used for anchor-write last_vs -- below decomposition_vs_threshold, OR the
  boundary firing) -> if depth < depth_cap and PolicyDecomposition.decompose_sequence()
  produces tiles (library-aware longest-match against shallower registered chunks,
  falling back to raw single actions), the candidate is REPLACED by its leaf-tile
  Trajectories (bounded recursive descent up to depth_cap levels); if depth >= depth_cap
  or no tiles exist, the candidate is EXCLUDED from the pool this tick rather than
  offered blind. Chunk injection is additive (the flat-grain CEM pool is untouched
  either way), so an excluded/replaced chunk never leaves E3 with nothing to select.
  Data flow (mid-execution, R4 second phase): while beta_gate.is_elevated and
  e3._committed_trajectory originated from chunk injection / MECH-321 decomposition,
  agent.select_action re-evaluates the REMAINING (unexecuted) tail of the committed
  sequence against the CURRENT observed latent on every tick, still via
  boundary_on(stream="rollout", ...) (the remaining content is, by definition, still a
  prediction, not an observation) -- a triggering remainder releases the commit latch
  (beta_gate.release() + the same five-field clear as the rung-6 release) so the NEXT
  tick's _e3_tick replans, rather than blindly finishing the remainder. This is a FOURTH
  principled release, alongside MECH-091 (safety, never overridden), the rung-6 duration
  release, and SD-034 closure de-commit.
  Backward compatible: disabled by default; agent.policy_decomposition stays None, every
  call site is None-guarded, HippocampalModule never registers a decomposition source and
  never calls boundary_on(stream="rollout", ...).
  Biological basis: Zacks et al. 2007 event segmentation theory (PE is the canonical
  boundary trigger, framework substrate-agnostic about observed-vs-imagined streams, R1
  conf 0.78); Schacter/Addis/Buckner 2008 constructive episodic simulation (the SAME core
  network supports remembering past and imagining future events -- R2 LOAD-BEARING
  empirical anchor for the bidirectional-substrate design, conf 0.74); Badre & D'Esposito
  2009 rostro-caudal prefrontal hierarchy + Koechlin & Summerfield 2007 cascaded cognitive
  control (R3 multi-level decomposition, depth cap 3-4, conf 0.78); McGovern & Barto 2001
  bottleneck-state subgoal discovery is an explicit FOIL, not the primary trigger (R5).
  Phased training required: no (pure arithmetic, no learned parameters, no gradients).
  THE ASYMMETRY WITH ARC-071 IS THE MOST IMPORTANT THING TO GET RIGHT HERE. ARC-071 is
  SLOW, repetition-driven, execution-side, and its default write path REFUSES
  hypothesis_tag=True outright (a hallucinated chunk would install a macro the agent never
  executed). ARC-070 is FAST, V_s-driven, simulation-side, and LEGITIMATELY FIRES under
  hypothesis_tag=True during rollout deliberation -- that is its PRIMARY phase, not an
  exception to be gated against. PolicyDecomposition.evaluate() has NO MECH-094 refusal
  branch at all: it never writes residue, never updates MECH-269 anchor sets, never
  touches MECH-287 broadcast, so there is nothing to refuse. The R4 hypothesis_tag
  argument is a pure diagnostics label distinguishing pre-commit (True) from
  mid-execution (False) call counts -- do not "fix" this by adding a write-gate that
  mirrors ChunkAccumulator.record_step; that would be importing the wrong claim's
  constraint. See policy_decomposition.py's "asymmetry with ARC-071" module docstring.
  A DEPTH-1 CHUNK MUST STILL BE ABLE TO DECOMPOSE -- CAUGHT BY THIS SESSION'S OWN
  ACTIVATION SMOKE TEST. decompose_sequence()'s first draft short-circuited to () whenever
  depth<=1 (reasoning: "nothing shallower than depth 1 to tile against"), which is correct
  about the library lookup but wrong about the OUTCOME: it made the single most common
  chunk shape (depth 1, composed directly of raw actions -- most ARC-071 formations never
  reach depth 2) permanently un-decomposable, landing every triggering depth-1 chunk as
  marked_unreliable/dropped instead of genuinely re-segmented. Fixed to skip only the
  library-lookup phase at depth<=1 while still falling through to the raw-action tiling
  loop (contract C3 pins this as a regression guard).
  Contracts: tests/contracts/test_arc070_policy_decomposition.py (21 tests: C1 defaults +
  from_dims forwarding + OFF inertness (no sub-config mirror needed), C2 loud precondition
  on use_event_segmenter, C3 depth-1 regression pin, C4 library-aware tiling incl.
  DISSOLVED chunks excluded, C5 atomic sequence has no tiles, C6 R1 OR-trigger (V_s alone
  / boundary alone / neither), C7 depth_cap marks unreliable instead of decomposing, C8
  live pre-commit withhold-and-replace, C9 live mid-execution latch release, C10 additive
  passthrough when nothing triggers incl. the no-decomposition-source bit-identical path;
  C11-C16 = the R5 bottleneck trigger mode below).
  Registered in tests/test_flag_inertness.py PROBED.
  R5 BOTTLENECK TRIGGER MODE -- ADDED 2026-07-25 (unblocks ARM_2 of the discriminative
  validation). The R1 trigger above is now selectable against an R5 alternative via
  REEConfig.decomposition_trigger_mode: "vs_boundary" (DEFAULT = the R1 OR trigger, unchanged
  and bit-identical) | "bottleneck" (R5: decompose ONLY at bottleneck states, REGARDLESS OF
  V_s). ARM_1 uses "vs_boundary"; ARM_2 uses "bottleneck". Bottleneck detection is an ONLINE
  INCREMENTAL diverse-density accumulator (PolicyDecomposition._update_and_test_bottleneck):
  a region is a bottleneck once it has BOTH recurred >= decomposition_bottleneck_min_visits
  times (default 3 -- the "repeated traversals" gate, McGovern & Barto 2001, and the source
  of ARM_2's predicted one-shot-rare signature) AND been entered-from/exited-to >=
  decomposition_bottleneck_min_distinct_neighbors distinct regions (default 2 -- funnel
  topology). Region key = a coarse FIXED quantisation of z_world content (round(z_world[:dims]
  / quant); decomposition_bottleneck_region_quant default 1.0, decomposition_bottleneck_region_dims
  default 8) -- NOT MECH-288's segment_id, which is monotonic and never recurs so could never
  register a bottleneck (documented in the module docstring "REGION KEY"). boundary_on(stream=
  "rollout", ...) is still called every evaluate() (R2 shared-substrate consumer + advances the
  rollout stream) but in bottleneck mode feeds only the audit fields, not the decision; V_s /
  boundary counters keep incrementing for audit (so a manifest can show "V_s WAS low yet the
  bottleneck trigger did not fire on it" -- the ARM_2 discriminative evidence).
  FAITHFULNESS CHOICE (documented deliberately): MECH-321's functional_restatement frames R5
  as a candidate OFFLINE/CONSOLIDATION-PHASE mechanism (batch analysis of the ARC-071
  ChunkLibrary for bottleneck topology). This landing operationalises the SAME statistical
  signal ONLINE/incrementally for the ARM_2 primary-trigger requirement; it does NOT claim to
  be the offline consolidation mechanism, which stays DEFERRED per R5 (see
  policy_primitive_granularity.md and the module docstring "R5 BOTTLENECK TRIGGER MODE").
  Default OFF / bit-identical: the accumulator is never allocated in "vs_boundary" mode (C15
  pins regions_tracked==0). New diagnostics in get_policy_decomposition_state():
  decomp_trigger_mode, decomp_n_bottleneck_fires, decomp_n_bottleneck_regions_tracked. Three-
  site REEConfig.from_dims wiring for all five new params (dataclass field, from_dims signature,
  from_dims body) + the agent.py PolicyDecompositionConfig construction site.
  Contracts for the mode: C11 default+validate (rejects unknown mode / non-positive gates) +
  from_dims forwarding; C12 bottleneck mode ignores V_s (low V_s alone does not fire, audit
  counter still increments); C13 one-shot-rare vs repeated-fires; C14 diagnostics surfaced;
  C15 vs_boundary leaves the accumulator untouched (bit-identity); C16 depth_cap still respected.
  Validation experiment: V3-EXQ-TBD queued via /queue-experiment (substrate-readiness
  diagnostic -- does decomposition fire and correctly withhold/replace a chunk candidate
  under an artificially induced low-V_s / boundary-firing rollout region). The full
  ARM_0/ARM_1/ARM_2 discriminative-pair design from MECH-321's claims.yaml
  functional_restatement (V_s-drop primary vs bottleneck-state primary vs OFF baseline,
  measuring execution-time prediction-failure rate) is a LATER experiment, mirroring how
  ARC-071's behavioural-latency measurement was deferred past its own substrate-readiness
  diagnostic.
  See ARC-070, ARC-069 (parent), ARC-071 (the inverse composition operator, BUILT
  2026-07-22 -- see the entry above), MECH-288 (substrate consumed), MECH-269 (V_s trigger
  source), MECH-094, REE_assembly/docs/architecture/policy_primitive_granularity.md.
