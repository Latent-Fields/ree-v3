## SD-033b GAP-8-affordability: reusable trained-OFC-head cache (chip-20260730-ofc-deval-affordable) (2026-07-31)
- SD-033b affordability infra: experiments/_lib/ofc_head_cache.py +
  ree_core/pfc/ofc_analog.py OFCAnalog.head_state_dict()/load_head_state_dict() --
  IMPLEMENTED 2026-07-31 (infra only; SD-033b / MECH-263 / Q-085 / MECH-163 status
  UNCHANGED, this PROMOTES NOTHING and claims.yaml is untouched).
  PROBLEM (measured, not assumed): OFCAnalog.state_bias_head /
  devaluation_bias_head emit exactly zero until deliberately trained
  (train_state_bias_head / train_devaluation_head, last Linear zeroed at init).
  Training requires gradient tracking through the live agent loop --
  experiments/_lib/baselines/arc071_chunking.py run_cell measured this at ~2 s/step
  against ~0.9 s/step no-grad, which is why V3-EXQ-841's driver substituted a
  direct env-side devaluation for the claims.yaml-named "SD-033b OFC-analog
  devaluation; existing driver lineage V3-EXQ-485g/i/k/m" route (a 6-arm x 8-seed
  grid at ~28 h trained vs ~7 h no-grad, per that driver's own docstring). Routed
  by chip-20260730-ofc-deval-affordable (spawned 2026-07-30 from
  failure_autopsy context around V3-EXQ-841/810a/834).
  THE FIX (additive, no-op for every existing caller -- nothing calls the new
  methods/module unless a script opts in):
    Module: ree_core/pfc/ofc_analog.py -- OFCAnalog.head_state_dict() (returns
      {"state_bias_head": <state_dict>} always, plus "devaluation_bias_head" when
      that head was built) and load_head_state_dict(blob) (strict PyTorch
      load_state_dict per head; a shape mismatch RAISES rather than silently
      loading a wrong-architecture head; loading a blob without a
      devaluation_bias_head entry, or onto an instance that built none, is a
      valid partial load). Neither touches state_code, use_ofc_analog, or the
      train_* flags -- pure weight I/O on the already-constructed heads.
    New module: experiments/_lib/ofc_head_cache.py -- a content-addressed,
      machine-local cache (torch.save blobs under ~/.ree_ofc_head_cache, override
      REE_OFC_HEAD_CACHE_DIR; REE_OFC_HEAD_CACHE_DISABLE=1 escape hatch), keyed via
      the SAME content-hash discipline arm_fingerprint.py uses for per-cell
      metrics: resolve_substrate_identity() [frozen once per process, the
      executed- not on-disk-substrate], machine_class(), a caller-declared
      config_slice, and seed. Mirrors experiments/_lib/baselines/
      maturation_curriculum.py's frozen-prefix cache mechanics (same governing
      asymmetry: a false HIT corrupts a conclusion, a false MISS only wastes
      compute; same atomic write + key-mismatch-on-load = MISS handling).
      compute_head_key() / load_blob() / store_blob() are the primitives;
      load_into() / store_from() are OFCAnalog-facing (duck-typed, no hard
      ree_core import); get_or_train(ofc, config_slice=..., seed=..., train_fn=...)
      is the one-call entry point most scripts want (HIT -> load, MISS -> call the
      CALLER-SUPPLIED train_fn then store).
  THE AFFORDABILITY LEVER (why this is more than a generic cache): OFCAnalog.
  compute_bias/compute_devaluation_bias read ONLY (state_code,
  candidate_world_summaries); state_code is an EMA of world_proj(z_world) [+
  outcome_proj(z_harm)] and never reads policy_chunking / chunk-proposal-injection
  / chunk_max_size. So a caller whose config_slice deliberately OMITS the dose
  axis (config_slice_declared=True, exactly mirroring arm_fingerprint's own
  narrowing contract) gets the SAME key across a grid's dose arms for one seed --
  O(arms x seeds) training collapses to O(seeds). This module does NOT verify
  that narrowing is sound for a given config_slice; it is the CALLER's obligation,
  same as arm_fingerprint.compute_arm_fingerprint's config_slice_declared. The
  safe default (whole config_slice, no narrowing) gets zero cross-arm reuse but
  zero false-hit risk.
  CROSS-DRIVER REUSE default flipped from arm_fingerprint's own default:
  include_driver_script_in_hash is NOT a parameter here at all (the key never
  folds in a script path) -- deliberately, so a LATER differently-authored grid
  experiment can reuse a head an EARLIER script trained, matching the existing
  "mint-as-you-go" / include_driver_script_in_hash=False convention for arm-reuse
  baselines (see CLAUDE.md "Saving a baseline for reuse").
  WHAT THIS DOES NOT DO, stated plainly: it does not train a head, pick a
  training protocol, or know anything about CausalGridWorldV2 vs SD-054
  reef/forage (the env the existing 485-lineage heads were actually trained on --
  those weights are NOT bundled or assumed portable here; a CausalGridWorldV2
  consumer must train its own). It does not attempt dependency-scoped substrate
  hashing (maturation_curriculum's narrower, provably-safe scope) because no
  training driver exists yet to derive a provable scope from -- the safe
  whole-tree default is used, matching arm_fingerprint's own posture.
  SCOPE NOTE (investigated, not fixed here): the immediate practical blocker on
  re-running Q-085/MECH-163 on CausalGridWorldV2 is NOT devaluation-route cost --
  V3-EXQ-841's own (draft, unconfirmed) autopsy found the committed-chunk
  commit-latch pins at realised length 1 regardless of configured dose in 100% of
  32 chunking cells (Finding B), independent of and upstream of which devaluation
  route is used; both claims carry pending_retest_after_substrate until that is
  resolved. This build is the general affordability capability the chip asked
  for; it does not itself unblock a re-queue.
  Tests: ree-v3/tests/contracts/test_ofc_head_cache.py (7 groups: key stability/
  discrimination C1-C2, the dose-inclusion-vs-exclusion reuse lever C3, a real
  get_or_train miss-then-hit round trip with a fake train_fn C4, OFCAnalog
  head_state_dict/load_head_state_dict round-trip + shape-mismatch-raises C5,
  corrupt/mismatched-blob-is-a-miss C6, the disable knob C7).
  GOVERNANCE: PROMOTES NOTHING, DEMOTES NOTHING. claims.yaml untouched (no
  dependency resolved, no evidence added or reweighted -- this is caching
  infrastructure for a training cost, not a result).
  See SD-033b (parent), SD-033b GAP-8 / GAP-8 DECOUPLE (the heads this caches),
  arm_fingerprint.py (the key discipline this mirrors), maturation_curriculum.py
  (the cache-mechanics precedent), failure_autopsy_V3-EXQ-841_2026-07-31.md
  (draft; the current actual blocker on Q-085/MECH-163, orthogonal to this build),
  failure_autopsy_V3-EXQ-810a_2026-07-30.md / failure_autopsy_V3-EXQ-822b-834_
  2026-07-29.md (the s/step and formation-schedule measurements this cites).
