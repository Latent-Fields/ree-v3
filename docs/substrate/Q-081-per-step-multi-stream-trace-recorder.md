## Q-081 per-step multi-stream trace recorder (`experiments/_lib/`, IMPLEMENTED 2026-07-22)

Shared recording harness for cross-stream organisation questions. **No `ree_core` change
-- zero substrate delta.** Landed `a0289b6` (recorder + profile + contracts), fixes
`d673790` / `4592e41`, `.gitignore` `925ceeb`. Foundational module under the Q-081 /
INV-091 program now documented below (surrogate null, reach-check, landmark-removal); it
already backs the run corpus V3-EXQ-824/824a/827/827a/828/838 and the 849/828a queue.
Scoped by `REE_assembly/evidence/planning/q081_cross_stream_telemetry_audit.md`, whose
verdict was that the signals are runtime-reachable but nothing recorded them and four were
default-OFF: the gap was a recorder + a config, not substrate.

- `experiments/_lib/stream_recorder.py` -- `StreamTraceRecorder`. One `on_step()` per env
  step, sampling `agent.get_state()` (ungated, broad) plus the E1/E3/salience/hippocampal
  caches. Replaced the episode-log block copy-pasted across 15 scripts, which stored
  **norms** and so was unusable for Q-081.
- `experiments/_lib/trace_store.py` -- content-addressed `.npz` sink; digest is over
  CONTENT, not the `.npz` file bytes (numpy embeds wallclock in zip headers). Manifests
  carry a lean pointer, never the blob. Root `$REE_TRACE_ROOT`, else `traces/` (gitignored).
- `experiments/_lib/q081_profile.py` -- the 8-flag config profile + non-default-substrate
  declaration block (`use_harm_stream`, `use_affective_harm_stream`,
  `use_salience_coordinator`, `use_event_segmenter`, `use_invalidation_trigger`,
  `use_tpj_comparator`, `use_sleep_loop`, `z_goal_enabled`).
- `tests/contracts/test_stream_trace_recorder.py` -- 18 contracts (C1..C6).

**Three load-bearing properties** (drop any and a Q-081 run is invalid): **(1) vectors,
not norms** -- a norm collapses 32-D and destroys the configuration structure the question
is about. **(2) per-signal freshness flags** -- E1 every step, E2 every 3, E3 every 10, and
`select_action` short-circuits between E3 ticks, so a naive recorder writes 9 held
duplicates + 1 fresh value per E3 stream and cross-stream analysis over that finds
structure that is a **sampler artefact** (Outcome B). Each stream declares its freshness
source; E3 freshness reads `clock._e3_phase_step == 0`, exact under MECH-091/MECH-093 where
`% 10` is wrong. **(3) store by reference** -- the blob never enters the git coordination
plane four phase3 writers share. **Boundary events are read NON-DESTRUCTIVELY** (the agent
drains its own queue at the next `sense()`), and `on_step()` MUST be called after `act()`
and before the next `act()`.

**Three findings the build added to the audit:** (a) the flags alone do NOT populate
`z_harm` / `z_harm_a` -- `act()` calls `sense()` without the harm channels; the loop must
pass them (`q081_profile.sense_kwargs_from_obs()`). (b) the E2 stream is E3-CADENCE, not
per-step -- the TPJ comparator is staged at the end of `select_action` (`agent.py:7582`),
after the short-circuit; distinct CONTENT from E3, not a distinct rate, so Q-081 has no
true middle-rate stream. (c) `SleepLoopManager` IS agent-instantiated under
`use_sleep_loop`; `z_goal` needs `z_goal_enabled` to exist at all plus the loop to call
`update_z_goal()`. Any profile run is NON-DEFAULT SUBSTRATE and must declare it. Validation
was by the 18 contracts + downstream consumption; experiment scripts go through
`/queue-experiment` (not this landing).
