## contextmemory-write-path-addressing-degeneracy: e1.context_memory.write_usage_balancing -- IMPLEMENTED (2026-08-19)
- contextmemory-write-path-addressing-degeneracy: e1.context_memory.write_usage_balancing --
  IMPLEMENTED 2026-08-19 (chip-20260816-implsub-contextmemory-writepath-degeneracy).
  ree_core/predictors/e1_deep.py (ContextMemory.__init__ / .write()), ree_core/utils/config.py
  (E1Config fields + REEConfig.from_dims threading).
  `ContextMemory.write()` addresses by a hard `scores.mean(0).argmin()`, which under a
  near-constant query stream is a deterministic single-slot fixed point --
  failure_autopsy_V3-EXQ-436e_2026-08-13.md established a closed-form sign discriminator
  `q . (write_signal - memory[argmin])` that predicted lock-vs-rotate 5/5, and V3-EXQ-436f
  confirmed it live (n_occupied_slots = 1 of 16 in BOTH arms on 3/5 seeds despite 2,837-4,903
  write() calls per arm, with the full SD-016 production combination armed and engaged). This
  is the WRITE-side sibling of SD-016 (cue-indexed RETRIEVAL, already implemented) -- the
  READ-path fix does not touch this defect: write() runs entirely under torch.no_grad() and
  compute_diversification_loss() only ever trains self.memory, never the write-address
  selection itself.
  Fix: a frequency-sensitive competitive-learning "conscience" bias (DeSieno 1988) on the
  write-address selection score -- a slot's eligibility for re-selection is penalized in
  proportion to an EMA of how recently/often it has already won, scaled to
  sqrt(memory_dim) (the expected order of magnitude of the raw dot-product scores by CLT), so
  a self-reinforcing lock cannot persist. No gradient is required (write() is already
  no_grad), so this is a pure selection-score adjustment, not a differentiable relaxation like
  the SD-016 read-path Gumbel-softmax (V3-EXQ-908) it is the write-side sibling of; the
  annealed-Gumbel alternative named in the substrate_queue implementation_hint was
  investigated and not used, since an annealing schedule and a straight-through estimator both
  exist to preserve gradient flow that write() never has.
  Config: E1Config.contextmemory_write_usage_balancing (bool, default False -- bit-identical
  to the legacy argmin path, no extra buffer constructed), .contextmemory_write_usage_bias_weight
  (float, default 1.0), .contextmemory_write_usage_decay (float, default 0.99, ~100-write EMA
  time constant). All three threaded through REEConfig.from_dims().
  Data flow: query_proj(state) . memory.T -> mean_scores -> + usage-EMA bias (when enabled) ->
  argmin -> write_usage_ema EMA-updated toward the winning slot.
  Backward compatible: default False; with the flag off, `selection_scores` is exactly
  `scores.mean(0)` and `write_usage_ema` is None -- verified bit-identical against the legacy
  write() expression.
  Validated (contract-test level, ree-v3/tests/contracts/test_contextmemory_write_usage_balancing.py):
  replaying a perfectly constant query stream (the failure_autopsy's own H-degenerate-query
  regime) against 3 seeds independently confirmed to lock the legacy path to n_occupied_slots=1
  over 3000 write() calls -- all 3 reach n_occupied_slots >= 2 under the fix (measured 3, 3, 16
  in a broader 20-seed sweep: 11/20 seeds locked under the legacy path, 20/20 reached >= 2
  occupied slots under the fix).
  Not a learning module in the gradient sense -- write() is no_grad throughout; no phased
  training. MECH-094 N/A (write() is a waking-stream operation, not simulation/replay content).
  Validation experiment: diagnostic-purpose ablation queued via /queue-experiment (see queue
  entry for EXQ id) -- feature ON vs OFF, same seeds, measuring n_occupied_slots against the
  substrate_queue entry's own registered acceptance target (>= 2 occupied slots in both arms
  on >= 3/5 seeds).
  See SD-017, ARC-045, MECH-166 (unblocked for their own validation now that the write-path
  precondition is repaired), SD-016, V3-EXQ-436e, V3-EXQ-436f.
