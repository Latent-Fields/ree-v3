## contextmemory-write-path-addressing-degeneracy: e1.context_memory.write_selection="gumbel_learned" -- IMPLEMENTED (2026-08-27)
- contextmemory-write-path-addressing-degeneracy: e1.context_memory.write_selection="gumbel_learned" --
  IMPLEMENTED 2026-08-27 (chip-20260826-contextmemory-gumbel-writeselect-build; HUMAN-DIRECTED build,
  substrate_queue HUMAN DECISION 2026-08-26 via /metaworker-orchestrate).
  ree_core/predictors/e1_deep.py (ContextMemory.__init__ / .write() / ._select_write_slot() /
  ._select_write_slot_gumbel() / .compute_write_addressing_loss(), E1DeepPredictor.__init__),
  ree_core/utils/config.py (E1Config fields + top-level REEConfig field + REEConfig.from_dims
  threading), ree_core/agent.py (REEAgent.compute_prediction_loss() wiring).
  THIRD mechanism for the same substrate_queue entry. Bias and refractory (above) are NOT
  reverted, replaced or re-scaled -- all three are available, all three default-off.
  2026-08-26 HUMAN DECISION: neither bias nor refractory closes the corrupting defect as a
  matter of addressing POLICY (both are mechanical occupancy workarounds, not a learned
  write-selection mechanism); build the entry's own implementation_hint -- annealed
  Gumbel-softmax matching V3-EXQ-908's confirmed READ-path mechanism.
  THIS IS NOT A RESURRECTION OF THE REJECTED "gumbel" MODE FROM THE 2026-08-19 ENTRY ABOVE --
  read that entry's "NOT ADOPTED" paragraph first. The rejected mode perturbed write()'s own
  UNTRAINED query.memory score with Gumbel noise and nothing ever shaped it (write() is
  entirely no_grad), reproducing exactly the same failure V3-EXQ-418i independently diagnosed
  on the read path's legacy q.k attention (uniform softmax saddle at self.memory's 0.01 init
  scale). This build's write_addr_tagger is a DEDICATED feedforward MLP (matching
  cue_slot_tagger's actual shape, not the q.k attention path), scoring state -> num_slots
  logits directly (never touching self.memory's scale), and it receives a REAL, VERIFIED
  gradient via the new compute_write_addressing_loss(), called explicitly by
  REEAgent.compute_prediction_loss() -- the piece compute_diversification_loss() never
  provided (it only ever trains self.memory content, never write-ADDRESS selection). Deliberately
  spelled "gumbel_learned", not "gumbel" -- that string stays rejected, pointing here, so no
  stray reference to the old name silently resurrects it.
  MEASURED, HONESTLY, AND THE RESULT IS MIXED. A first-attempt loss design (standard MoE
  load-balancing / "importance" loss, Shazeer et al. 2017) was built, wired, and then MEASURED
  (5 seeds x 300 SGD steps on the 2-cluster content-conditioning stream) to reproduce the
  IDENTICAL failure signature as the rejected mode -- 2-cluster Jaccard EXACTLY 1.000 on 5/5
  seeds -- because batch-mean uniformity has a degenerate minimum satisfied equally by
  content-blind near-uniform-everywhere and by genuine content-conditioning. REPLACED with a
  pairwise-diversity loss mirroring compute_diversification_loss()'s own structure (mean
  squared off-diagonal cosine similarity) applied to PER-EXAMPLE selection distributions
  instead of memory rows -- this has no such degenerate minimum (near-uniform distributions
  are mutually HIGH-similarity, so the gradient pushes away from them) but its
  content-discrimination effect is NOT YET DEMONSTRATED either: a toy 300-step training loop
  did not move it off its own near-uniform starting point in the time available, which is
  consistent with (not distinguishable from) the SAME symmetry-breaking difficulty
  compute_diversification_loss() itself needed a full experiment (V3-EXQ-907) rather than a
  toy script to overcome.
  Config: E1Config.contextmemory_write_selection accepts "gumbel_learned" (third value;
  "gumbel" bare stays rejected), .contextmemory_write_gumbel_tau_init (1.0),
  .contextmemory_write_gumbel_tau_min (0.1), .contextmemory_write_gumbel_anneal_steps (2000),
  .contextmemory_write_gumbel_tagger_hidden (32). Top-level REEConfig.contextmemory_write_addressing_loss_weight
  (float, default 0.0 -- mirrors sd016_diversification_weight's own top-level placement and
  the reason for it). All threaded through REEConfig.from_dims() at all three sites.
  Data flow (selection): write_addr_tagger(state) -> mean over batch -> + usage-EMA bias (when
  composed) -> annealed straight-through Gumbel-max (train) / plain argmin (eval) ->
  _record_write() updates slot_write_counts / last_write_index -- write() itself stays
  ENTIRELY torch.no_grad() in this mode too, exactly like argmin/refractory.
  Data flow (training signal, SEPARATE call, not reused from write()):
  compute_write_addressing_loss(states) -- a fresh forward pass over a caller-supplied batch
  of already-detached states -- -> write_addr_tagger(states) -> per-example softmax(-scores)
  -> pairwise cosine similarity, minimized -> REEAgent.compute_prediction_loss() adds
  weight * this loss, gated on weight > 0 AND write_selection == "gumbel_learned". Recomputed
  fresh (not retained from write()) because write()'s .data content update bypasses autograd's
  version-counter check -- a retained graph consumed after a later write() would silently
  differentiate through the WRONG (since-mutated) memory values.
  Backward compatible: default "argmin"; write_addr_tagger is None (not constructed) and no
  state_dict keys are added unless "gumbel_learned" is selected. RNG: eval mode is
  deterministic (bit-identical to plain argmin on the tagger's scores, verified exactly, not
  approximately) and consumes no RNG; TRAIN mode DOES consume RNG -- a real, documented
  difference from argmin/refractory, which stay fully deterministic in every mode.
  Occupancy: >= 2 occupied on >= 3/5 seeds cleared decisively (16/16 on all 5 measured seeds),
  delivered by Gumbel noise alone, independent of whether write_addr_tagger has been trained
  at all.
  Contracts: ree-v3/tests/contracts/test_contextmemory_write_gumbel_learned.py (26,
  time-independent). Assertion policy: no assertion claims content-discrimination is achieved
  (that is explicitly not yet proven); assertions cover gradient reachability into
  write_addr_tagger (the concrete difference from the rejected mode), write()'s continued
  full no_grad status, RNG consumption exactly where documented, config validation, and the
  occupancy floor.
  Phased training required: the write_addr_tagger / compute_write_addressing_loss() pairing
  IS a learning module in the gradient sense (unlike bias/refractory) -- MECH-094 N/A (write()
  remains a waking-stream operation; the new loss trains on already-detached buffer states,
  not simulation/replay content).
  Validation experiment: PENDING, and needs MORE than a flag flip -- unlike bias/refractory, a
  gumbel_learned driver must actually run contextmemory_write_addressing_loss_weight > 0
  through real training steps, or it only exercises the (already-decisive) occupancy effect
  and says nothing about the still-open content-discrimination question. Does NOT unblock
  chip-20260818-sd017-ceiling-retest-gated or chip-20260818-mech152-redesign-queue-gated: a
  default-off knob changes no driver, and even with the flag on, content-discrimination is
  unproven.
  Architecture doc: REE_assembly/docs/architecture/contextmemory_write_address_selection.md
  ("THIRD mechanism" section, 2026-08-27) has the full measurement record for both loss-design
  attempts.
  See substrate_queue contextmemory-write-path-addressing-degeneracy, SD-017, ARC-045, MECH-166, SD-016.
