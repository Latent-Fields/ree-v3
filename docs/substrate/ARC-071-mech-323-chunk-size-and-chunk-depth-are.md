## ARC-071 / MECH-323: chunk SIZE and chunk DEPTH are GROWABLE CEILINGS DERIVED FROM THE DELIBERATION BUDGET, not fiat constants -- IMPLEMENTED (2026-07-27)
- ARC-071 / MECH-323: chunk SIZE and chunk DEPTH are GROWABLE CEILINGS DERIVED FROM THE
  DELIBERATION BUDGET, not fiat constants -- IMPLEMENTED 2026-07-27 (size ree-v3 c74434f,
  pre-rebase sha 95f9376932; depth ree-v3 7c201f7). Two landings on ONE compute-versus-
  efficiency trade-off, documented together because they share a budget and are coupled
  structurally. ree_core/policy/policy_chunking.py (ChunkAccumulator.effective_max_chunk_size
  / consider_ceiling_growth, PolicyChunking.effective_max_depth / structural_max_depth /
  consider_depth_growth, ChunkAccumulatorConfig.derived_chunk_ceiling),
  ree_core/policy/policy_decomposition.py (depth_cap_config_issues takes the derived bound),
  ree_core/utils/config.py, ree_core/agent.py (_resolve_chunk_deliberation_horizon + the
  ChunkAccumulator/PolicyChunking build sites).
  Config: REEConfig.use_growable_chunk_ceiling (default False = chunk_max_size is a hard
  lifetime cap, as-first-built, bit-identical; set True to enable) + chunk_ceiling_budget_fraction
  0.1667 + chunk_ceiling_returns_threshold 0.10 + chunk_ceiling_hard_max 12; and
  REEConfig.use_growable_chunk_depth (default False = chunk_max_depth is a hard lifetime cap;
  set True to enable) + chunk_depth_budget_fraction 0.1 + chunk_depth_returns_threshold 0.10 +
  chunk_depth_hard_max 6. BOTH read ONE shared budget knob, REEConfig.chunk_deliberation_horizon
  (default 0). All nine plumbed through the three from_dims sites (dataclass field, signature,
  body) plus the agent build site, and registered PROBED in tests/test_flag_inertness.py.
  NESTING: both are sub-switches of use_policy_chunking (the accumulator/library are never
  constructed without it), so either flag ON with chunking OFF is silently inert rather than an
  error -- unlike use_chunk_dissolution_retention, which RAISES under use_chunk_maintenance=False.
  THE SHARED BUDGET KNOB IS SENTINEL-DEFAULTED, and that is a live-knob guarantee rather than a
  convenience: chunk_deliberation_horizon = 0 MIRRORS HippocampalConfig.horizon (the real rollout
  budget; falls back to 10 with no hippocampal block), >= 1 is an explicit experimental override
  honoured verbatim (ree_core/agent.py:_resolve_chunk_deliberation_horizon). Unconditionally
  mirroring the hippocampal horizon would have shipped a from_dims knob that accepts a value and
  silently discards it -- the same dead-knob class as from_dims swallowing an unknown kwarg, one
  layer further down and harder to see.
  Data flow: HippocampalConfig.horizon (or the override) -> derived_chunk_ceiling =
  floor(horizon * chunk_ceiling_budget_fraction) and derived_max_depth = floor(horizon *
  chunk_depth_budget_fraction) -> effective ceilings START at chunk_max_size / chunk_max_depth
  and grow ONE element / ONE level at a time toward those bounds -> each step licensed by a
  realised marginal outcome gain >= the returns threshold -> the raised ceiling takes effect from
  the NEXT note_outcome. Growth is evaluated in PolicyChunking.note_outcome AFTER the tally,
  reacquisition and maintenance passes, size BEFORE depth (the structural bound reads the live
  size ceiling, so evaluating depth first would judge it against a stale size budget and refuse a
  growth the same trial had just licensed).
  Backward compatible: both default False; the effective ceilings are pinned at the inherited
  constants and no growth path runs, so a default agent is bit-identical to the 2026-07-22
  landing. VERIFIED, and this is the point of the anchoring below: at REE's ACTUAL rollout horizon
  of 30 the derivations return exactly 5 and exactly 3, so even flag-ON at today's budget starts
  where the fiat constants sat.
  NEITHER NUMBER IS REPLACED. Neither source licenses a replacement constant, so none is asserted
  -- the fractions are ANCHORED (0.1667 = 5/30, 0.1 = 3/30) to reproduce the inherited Sakai
  budget and the inherited R4 recursion cap at the budget the agent actually has. Only a LARGER
  deliberation budget derives a larger ceiling. What changed is the parameter's SHAPE, from
  constant to function. Read the anchors as calibration: the papers fix the shape of the
  relationship and nothing about its scale, so the scale is pinned to the one quantity that does
  carry warrant rather than to a number invented here. The depth anchor is pinned by a contract
  that builds a REAL REEAgent, because HippocampalConfig.horizon has a dataclass default of 10
  that no built agent uses -- anchoring against 10 would have been a large raise disguised as a
  derivation.
  THE BRAKE IS THE GROWTH RULE, not a second parameter. What bounds chunk growth empirically is
  DIMINISHING RETURNS (Ramkumar's monkeys never collapse the whole sequence into one chunk), so a
  merge must have actually PAID: size growth needs a realised gain over the best shorter context,
  depth growth a realised gain from composing AT the ceiling over the best chunk one level
  shallower that it contains. An accumulator that grew monotonically with practice would
  eventually collapse everything into one unit whatever number it stopped at -- the fixed-constant
  failure in a slower disguise. Means come from the LIVE tally, never the frozen value_tag, so a
  chunk whose returns have since collapsed cannot keep licensing growth; and "no evidence" is not
  a gain of zero (both contract-pinned).
  DECOUPLED FROM R_min BY CONSTRUCTION, on both ceilings. Bo 2009 found the capacity-to-chunk-
  LENGTH correlation in both age groups but capacity-to-learning-RATE only in the young, so size
  and formation rate are SEPARABLE quantities. The obvious implementation -- grow the ceiling as
  repetitions accumulate -- would silently re-couple them and re-introduce exactly the confound
  that dissociation rules out. Growth reads realised marginal outcome gain and the deliberation
  budget; chunk_min_repetitions enters only as a judge-ability filter (is this sequence attested
  enough to be MEASURED), never as the thing being measured. Do not "simplify" it into a practice
  counter.
  THE STRUCTURAL COUPLING IS THE SHARP PART, and it is mechanical rather than by analogy. A chunk
  composes another only by CONTAINING it, so a depth-D chain needs D distinct sequence lengths
  drawn from [min_chunk_size, effective ceiling] and the deepest hierarchy this substrate can
  physically mint is structural_max_depth = effective_max_chunk_size - min_chunk_size + 1 (4 at
  the 2-5 budget). consider_depth_growth() refuses to grow past that bound and reads the LIVE
  (possibly grown) size ceiling, so raising the size budget is what makes deeper hierarchies
  reachable at all. Refusing an inert raise is how this avoids repeating the defect the 2026-07-27
  MECH-321 scoping spike found in decomposition_depth_cap: a depth knob with no degree of freedom
  over which a literature argument was nonetheless conducted. Note max_depth=3 is genuinely
  BINDING today -- at the 2-5 budget the substrate would otherwise mint a depth-4 chunk.
  DEPTH IS LINEAR IN THE BUDGET, NOT LOGARITHMIC. The log intuition comes from hierarchies whose
  span MULTIPLIES per level (b**D primitives at branching factor b). REE's descent is not that:
  each level is one more sequential re-tiling pass, and HippocampalModule._recursive_leaf_tiles is
  bounded by `iterations < depth_cap`, an ITERATION count -- so depth D costs linearly in D and the
  affordable depth is linear in the budget. The fraction absorbs the unknown per-level cost
  constant.
  MECH-321 COUPLING: decomposition_depth_cap is a DERIVED MIRROR of chunk_max_depth, so
  depth_cap_config_issues() now takes derived_max_depth as an optional third argument and stops
  warning INERT about a cap the growing ceiling will in fact reach. Omitted, or not higher than
  the static bound (the default, and every shipped MECH-321 run), it is byte-identical.
  TRACTABILITY IS PRESERVED, worth stating because the original 2-5 bound was justified by it.
  note_outcome's enumeration is O(L * ceiling), not combinatorial, L is capped at
  max(effective_max_chunk_size * 4, 32) by record_step, max_tracked_sequences (FIFO) is the hard
  memory bound and is unchanged, and chunk_ceiling_hard_max / chunk_depth_hard_max are absolute
  backstops.
  NOT IMPLEMENTED -- SHRINKAGE, on both ceilings, recorded as a gap rather than guessed at.
  Nothing lowers a ceiling once raised. Bo 2009's cross-agent half is covered by the derivation;
  the within-lifetime declining-capacity half needs a signal REE does not have.
  Biological basis: Ramkumar, Acuna, Berniker, Grafton, Turner & Kording 2016 (Nat Commun,
  10.1038/ncomms12176) -- chunking is the OUTPUT of an efficiency/computation trade-off, not a
  fixed capacity; two macaques learning a ten-element reaching sequence over months MERGE chunks
  as practice lowers computation cost, optimising "over increasingly longer horizons". Bo, Borza &
  Seidler 2009 (J Neurophysiol, 10.1152/jn.00393.2009) -- chunk length tracks visuospatial
  working-memory capacity and DECLINES with age, constraining from the other side. Solway, Diuk,
  Cordova, Yee, Barto, Niv & Botvinick 2014 (PLoS Comput Biol, 10.1371/journal.pcbi.1003779) --
  filed in the corpus as a deliberate NULL: the normative account of what makes one action
  hierarchy better than another is the paper that WOULD carry a principled policy-grain depth
  limit, and it declines to give one, capping its own analysis at one level for stated
  tractability reasons and stating the framework generalises to deeper hierarchies unaltered. The
  R3 sources supplying 3-4 (Badre & D'Esposito 2009, Koechlin & Summerfield 2007) are ANATOMICAL
  grain, not policy grain -- how deep a brain's control hierarchy runs, not how deep THIS agent
  can afford to search.
  New readouts (so an arm is separable in a manifest without reading config):
  chunk_acc_effective_ceiling / chunk_acc_ceiling_derived_max / chunk_acc_n_ceiling_growths /
  chunk_acc_last_ceiling_gain on the accumulator; chunk_effective_max_depth /
  chunk_derived_max_depth / chunk_structural_max_depth / chunk_n_depth_growths on the facade.
  Phased training required: no (pure arithmetic, no learned parameters, no gradients).
  MECH-094: not applicable directly -- growth is evaluated inside note_outcome, which is reached
  only from the waking outcome path, and the accumulator's existing hypothesis_tag refusal on
  record_step is untouched.
  Contracts: tests/contracts/test_arc071_policy_chunking.py -- the C11 block (14 contracts, 21
  cases: off-by-default and pinned when off, the derivation reproducing 5 at horizon 30, scaling
  with the budget, growth requiring a realised return, no-growth when a shorter context already
  predicts, no-evidence-is-not-a-gain-of-zero, the brake plateauing below the derived maximum,
  never exceeding the derived bound, decoupling from the repetition tally, an end-to-end
  not-inert check, reset returning the initial budget, incoherent-config refusal, from_dims
  forwarding, sentinel horizon mirroring the hippocampal budget) and the C12 block (20 contracts,
  28 cases: the same battery for depth, plus depth-and-size reading the SAME budget, the inherited
  cap being binding today, the structural bound being set by the size ceiling AND rising with a
  grown one, returns reading the live tally not the frozen value_tag, and get_state reporting both
  bounds separately); plus 4 C17b contracts in tests/contracts/test_arc070_policy_decomposition.py
  for the MECH-321 derived-bound handoff.
  Validation experiment: NOT YET QUEUED. Both ceilings are prerequisites for the coupled-parameter
  experiment (do decomposition depth and composition parameters move together as compute changes),
  which needed an independent variable that could move; chunk_deliberation_horizon is now that
  single variable. NOTE the standing V3-EXQ-810 bound: at 24-step episodes the buffer holds ~3
  symbols, so sizes 4-5 and any depth above the resulting structural bound are unreachable
  whatever these flags are set to -- a successor must run LONGER EPISODES, not merely flip a flag.
  MECH-323 / ARC-071 stay candidate/v3_pending; this build promotes and demotes nothing.
  See ARC-071, MECH-323, MECH-324, MECH-321/ARC-070 (the derived-mirror depth_cap), MECH-094, and
  the "CHUNK SIZE IS A GROWABLE CEILING" / "CHUNK DEPTH IS ALSO BUDGET-DERIVED" sections of
  policy_chunking.py's module docstring.
