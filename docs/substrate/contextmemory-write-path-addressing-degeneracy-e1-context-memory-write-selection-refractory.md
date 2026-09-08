## contextmemory-write-path-addressing-degeneracy: e1.context_memory.write_selection="refractory" -- IMPLEMENTED (2026-08-19)
- contextmemory-write-path-addressing-degeneracy: e1.context_memory.write_selection="refractory" --
  IMPLEMENTED 2026-08-19 (chip-20260819-contextmemory-add-refractory-mode; USER-AUTHORISED build,
  option (c) of chip-20260819-contextmemory-writesel-disposition-evidenced).
  ree_core/predictors/e1_deep.py (ContextMemory.__init__ / .write() / ._select_write_slot() /
  ._record_write()), ree_core/utils/config.py (E1Config fields + REEConfig.from_dims threading).
  SECOND mechanism for the same substrate_queue entry, landed ALONGSIDE the conscience bias
  above. That one is NOT reverted, replaced, or re-scaled -- both are available and both are
  default-off.
  The two are orthogonal by construction: `contextmemory_write_usage_balancing` adjusts the
  SCORE, `contextmemory_write_selection="refractory"` restricts the ELIGIBLE SET (the k
  most-recently-written slots are ineligible). They compose, and all four combinations are
  legal. Deliberate rather than a missing mutual exclusion -- mutual exclusion would need a
  raise or a silent precedence rule, whereas the composition is well defined and measurable.
  Config: E1Config.contextmemory_write_selection (str, default "argmin" = every slot eligible =
  bit-identical legacy path; only "argmin" and "refractory" accepted, anything else raises at
  construction), .contextmemory_write_refractory_k (int, default 2, inert unless the mode is
  "refractory", capped internally at num_slots - 1 so a large k degrades instead of deadlocking).
  Both threaded through REEConfig.from_dims() at all three sites (from_dims silently swallows
  unknown kwargs, so a knob wired at two of three fails open and silently).
  Data flow: query_proj(state) . memory.T -> mean_scores -> + usage-EMA bias (when enabled) ->
  _select_write_slot() masks the last-k window with +inf -> argmin over the remainder ->
  _record_write() updates slot_write_counts / last_write_index.
  Backward compatible: default "argmin"; verified bit-identical against the verbatim legacy
  write() expression over 5 seeds x 200 writes (identical slot sequence AND identical final
  memory tensor). Consumes NO RNG in any mode -- refractory is fully deterministic -- so every
  existing seeded trajectory is unchanged. state_dict() unchanged (slot_write_counts is
  persistent=False, so existing checkpoints load untouched); named_buffers() does gain
  slot_write_counts, which the landed sibling contract's negative control tolerates by scoping
  itself to "write_usage_ema" names (pinned, not assumed).
  WHY BOTH MECHANISMS EXIST, AND WHAT DOES NOT JUSTIFY EITHER. Both clear the registered
  acceptance floor (>= 2 occupied slots on >= 3/5 seeds), so occupancy CANNOT choose between
  them. The independently pre-registered probe (REE_assembly/evidence/planning/
  contextmemory_write_selection_comparison_20260819.md; pre-registration REE_assembly
  fcfb311e4b, results b7e072ddf0) found the occupied-slot COSINE column cannot discriminate the
  arms at 5 seeds -- every contrast |dz| <= 0.47, |t(4)| <= 1.04, sign-inconsistent across
  seeds, INCLUDING the +0.6060 -> +0.5919 refractory-over-legacy gap (dz = -0.06). Do NOT cite
  that column for either mode. The case for refractory is STRUCTURAL and rests on the probe's
  deterministic columns, reproduced exactly by this landing: (1) occupancy >= k+1 holds BY
  CONSTRUCTION for any stream/seed/init, where the conscience bias reaches full occupancy only
  empirically; (2) with the bias on, the sqrt(memory_dim) = 11.31 scaling puts the usage term
  2-3 orders of magnitude above the ~0.026 across-slot spread of mean_scores, so the address
  becomes a function of the write COUNTER rather than the query -- a strict LRU cycle on ~99%
  of writes (round-robin agreement 0.991, entropy exactly 4.00 bits, HHI exactly 1/16, all 16
  slots on every seed regardless of content). Real occupancy, and not globally content-blind
  since the cycle ORDER is content-determined once, but occupancy without addressing.
  Refractory scores 0.000 on that same index and its slot count tracks legacy's on the
  non-locking seeds. (3) An absolute refractory period on a recently-active unit is a
  first-order property of real neurons; a global usage-EMA conscience bias is not.
  NOT ADOPTED: the salvaged branch's "usage" and "gumbel" modes (ree-v3 tag
  stash-archive/20260819-dd4b0a4, LOCAL-ONLY to DLAPTOP). Measured content-blind -- cluster
  Jaccard 0.600 and exactly 1.000 on 5/5 seeds respectively, i.e. both contexts write the same
  slot set. The salvaged "usage" mode is also a DIFFERENT algorithm from the conscience bias
  above (argmax(-z(sim) - w*z(usage)) vs argmin(sim + w*usage*sqrt(d))); they must not be
  conflated. Both names are rejected explicitly by the config validator so a config carrying
  them over errors rather than silently falling back to the defect.
  Phased training required: no (write() is entirely under torch.no_grad(); no encoder head and
  no gradient path is added or altered). MECH-094 hypothesis_tag: not applicable (no
  simulation, replay, or non-waking-state content).
  Instrumentation (maintained in EVERY mode, including the legacy default): ContextMemory
  .last_write_index, .slot_write_counts, .occupied_slots(). V3-EXQ-436f's occupancy tracker
  learned the written slot by DUPLICATING write()'s own argmin expression, which reports the
  wrong slot the moment the selection rule changes -- and it has now changed twice. Any new
  driver must read these instead of re-deriving; pinned by
  test_stale_reimplementation_of_the_old_rule_disagrees.
  Contracts: ree-v3/tests/contracts/test_contextmemory_write_address_selection.py (51 with the
  landed sibling file; roughly half negative controls -- the defect pin, bit-identity, RNG
  non-consumption, unchanged state_dict, the landed knobs untouched, and the load-bearing
  test_landed_usage_balancing_is_a_fixed_cycle_and_refractory_is_not). Every numeric assertion
  is on a deterministic quantity; there is deliberately NO assertion on occupied-slot cosine.
  Validation experiment: still PENDING for BOTH modes -- substrate_queue status stays
  implemented_pending_validation. Chipped as
  chip-20260819-queueexp-contextmemory-writesel-validation (covering both arms; the
  previously-recorded chip for the conscience-bias arm was never actually written to
  TASK_CHIPS.json, so this repairs that gap rather than adding a second unowned one).
  Does NOT unblock chip-20260818-sd017-ceiling-retest-gated or
  chip-20260818-mech152-redesign-queue-gated: a default-off knob changes no driver, no driver
  sets either flag, and the occ_cos DV still cannot discriminate at those experiments' powers.
  Plan of record: REE_assembly/evidence/planning/contextmemory_refractory_mode_dataflow_plan_20260819.md.
  See substrate_queue contextmemory-write-path-addressing-degeneracy, SD-017, ARC-045, MECH-166, SD-016.
