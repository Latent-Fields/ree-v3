## MECH-027 Build 2: force_sleep_cycle_at_eval_boundary -- sleep-cycle interleave reachable inside an eval window (2026-09-02)
- MECH-027's hypervigilance falsifier also needed the replay/hippocampal-injection-
  suppression channel reachable DURING an eval/measurement window. A red-team review of the
  drafted V3-EXQ-979 falsifier found this structurally BLOCKED for the normal driver pattern:
  SleepLoopManager.notify_episode_end() (the K-episode-cadence entry) is reachable only across
  an agent.reset() episode boundary, and every experiment driver surveyed called
  agent.sleep_loop.force_cycle() (the existing "run a cycle immediately, bypassing the
  K-episode counter" hook, phase_manager.py) ONLY during warmup, before eval begins -- a
  documented WARMUP-only convention, not a substrate limitation.
  INVESTIGATION FINDING: force_cycle() already works correctly at an eval boundary -- ONE
  experiment, V3-EXQ-909 (a continuity-trajectory eval driver: a single continuous life across
  eval segments, agent.reset() called only before the FIRST segment so residue/goal
  state/episode counters persist across segments -- see its own docstring), already calls it
  there, via a hand-rolled two-step sequence at every segment boundary:
  `agent._flush_exploration_episode()` (so the just-completed segment's waking exploration
  trajectory reaches the hippocampal replay pool -- agent.reset() would do this automatically,
  but a continuity driver deliberately never calls reset() at a boundary) then
  `agent.sleep_loop.force_cycle(agent)` (fires immediately, bypassing the K-cadence; resets
  episodes_since_sleep / steps_since_sleep on completion so the cadence and the GAP-9
  within-life trigger both stay consistent afterward). This sequence was NOT documented as a
  sanctioned pattern anywhere outside that one script -- the draft V3-EXQ-979 script's own
  author, unaware of the 909 precedent, wrote "driven manually via force_cycle() during warmup
  only" directly in its docstring.
  FIX: `REEAgent.force_sleep_cycle_at_eval_boundary()` (ree_core/agent.py, sibling to
  run_sleep_cycle()) formalizes the exact 909 sequence as one documented, tested method: flush
  then force_cycle, in that order. It does NOT call agent.reset() -- residue, goal state,
  latent state, and episode counters (_step_count, _harm_this_episode, etc.) are all left
  untouched, the defining difference from resetting at a boundary. Returns None (no-op
  downstream) when use_sleep_loop is False, mirroring notify_episode_end's None-safe
  consumption -- a driver need not guard the call. A driver that wants sleep-cycle interleaving
  reachable at chosen eval boundaries (every boundary, or a subset) should call this method
  instead of hand-rolling the sequence.
  SCOPE: this covers BETWEEN-episode / between-segment interleaving (the pattern 909
  demonstrates and MECH-027's falsifier needs). TRUE mid-episode (within a single continuous
  step sequence, no boundary at all) interleaving is a DIFFERENT, already-solved problem via a
  DIFFERENT mechanism -- sleep_substrate:GAP-9's automatic within-life trigger
  (SleepLoopManager.notify_waking_step(), use_within_life_sleep_trigger, landed 2026-08-14) --
  which fires on its own need/ceiling predicate rather than at a driver-chosen point; this
  method does not replace or interact with it.
  Module: ree_core/agent.py (REEAgent.force_sleep_cycle_at_eval_boundary, ~55-line method with
  its own docstring covering the ordering rationale; no new config flags, no new class).
  Backward compatible: purely additive method; no existing call site touched;
  agent.sleep_loop.force_cycle / agent.reset() / notify_episode_end are byte-identical.
  Contracts: tests/contracts/test_mech027_force_sleep_cycle_at_eval_boundary.py (5: bypasses
  the K-episode cadence and fires regardless; flushes the exploration buffer BEFORE firing and
  clears the per-episode buffers; does NOT reset step counter / episode harm accumulator /
  current-latent identity, unlike agent.reset(); returns None without raising when
  use_sleep_loop is off, and still flushes; resets the cadence counters after firing, same as
  force_cycle() itself).
  MECH-094: N/A (this is a boundary-timing convenience wrapper around two existing,
  already-classified surfaces -- _flush_exploration_episode is a waking-episode-end operation,
  force_cycle's own sleep-pass content keeps its existing hypothesis_tag=True tagging
  unchanged).
  GOVERNANCE: PROMOTES NOTHING. MECH-027 stays provisional; substrate-only (no claims.yaml
  disposition change). Unblocks the sleep-interleave-in-eval half of MECH-027's non-degeneracy
  precondition for a future V3-EXQ-979-style falsifier (the graded-precision half is Build 1,
  the preceding CLAUDE.md section).
  See MECH-027 (claims.yaml; the claim this unblocks), sleep_substrate:GAP-9 (the orthogonal
  automatic within-life trigger; distinct mechanism, distinct scope), sleep_substrate_plan.md
  (plan-of-record), V3-EXQ-909 (the prior-art precedent this formalizes), MECH-165 (the
  exploration-buffer/_flush_exploration_episode substrate this reuses unmodified).
