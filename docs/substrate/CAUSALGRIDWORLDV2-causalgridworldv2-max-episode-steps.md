## CausalGridWorldV2: max_episode_steps constructor kwarg -- IMPLEMENTED (2026-08-10)
- environment.max_episode_steps -- IMPLEMENTED 2026-08-10 (chip-20260810-fishtank-max-episode-steps).
  Module: `ree_core/environment/causal_grid_world.py` (`CausalGridWorld`, aka CausalGridWorldV2
  when `use_proxy_fields=True`). Not a numbered SD -- this is a pure parameterization of an
  existing termination condition, no new latent field, encoder, or observable; follows the
  SD-022/023/029/047/048/049 precedent of a flat, env-only constructor kwarg not surfaced through
  `REEConfig.from_dims`.
  Origin: `REE_assembly/evidence/planning/organism_lifespan_development_review_906_lineage_2026-08-10.md`
  Section 10 item 1 (uncensored survival-to-death Fishtank successor design) assumed the
  per-segment step cap was already driver-configurable; verified false against source before
  V3-EXQ-912 was designed around a segment-count workaround instead (see that experiment's
  module docstring "SUBSTRATE READINESS FINDING"). This closes the gap for a future TRUE
  single-life uncensored design.
  Problem: `step()` computed `_step_cap_reached = self.steps >= 500` -- a bare literal, no
  constructor parameter anywhere in the class. No driver, regardless of its own
  `steps_per_episode` argument to its eval loop, could make a single continuous segment exceed
  500 steps: `env.step()` forced `done_cause="step_limit"` at `self.steps==500` every time.
  Fix: `max_episode_steps: int = 500` constructor kwarg, stored as `self.max_episode_steps`.
  `_step_cap_reached` now reads `self.steps >= self.max_episode_steps`. The `body_state[9]`
  "episode_progress" proprioceptive observable (previously `self.steps / 500.0`) was also
  changed to `self.steps / float(self.max_episode_steps)` -- left as the literal, it would have
  silently saturated to 1.0 at step 500 of a longer-capped episode, corrupting the observable's
  "fraction of episode elapsed" semantics for any caller that actually uses a raised cap. No
  other reference to the literal 500 exists in `ree_core/` or `tests/` tied to this cap (grepped).
  Backward compatible: default `500` is bit-identical to the pre-existing hardcoded literal --
  no existing caller passes this kwarg. Confirmed via inline smoke test: (a) default-omitted
  construction still caps at exactly step 500 with `done_cause="step_limit"`; (b)
  `max_episode_steps=1500` honored end-to-end (cap fires at step 1500, `episode_steps==1500`);
  (c) `body_state[9]` reads `0.5` at step 500 of a `max_episode_steps=1000` run (tracks the
  configured cap, not the old literal); (d) `max_episode_steps=10` (small-cap sanity). The two
  existing contract tests that exercise the default cap
  (`tests/contracts/test_episode_termination_recording.py`,
  `tests/contracts/test_sd094_subgoal_arrival_and_hazard_free_contamination.py::test_done_cause_reports_step_limit_at_the_cap`)
  construct the env with no `max_episode_steps` override, so they are unaffected by construction.
  Also verified against the full `tests/contracts` corpus (3516 passed, 11 skipped, 43 subtests)
  via `remote_pytest.sh` on `ree-worker-4`, twice, with this change staged.
  Phased training: not applicable (env-only constructor param; no new trainable parameters, no
  new observable channel -- `body_state[9]` already existed). MECH-094: not applicable (env
  observation stream, not replay/simulation content).
  Validation experiment: none queued by this session -- writing the actual uncensored-single-life
  survival experiment that consumes this kwarg is separate `/queue-experiment` follow-on work,
  chipped rather than done here per chip instruction (this change's own scope is substrate
  parameterization only). Readiness tracked in `substrate_queue.json` (sd_id
  SD-FISHTANK-MAX-EPISODE-STEPS).
  See MECH-489/SD-099/SD-100 (unrelated, adjacent entries -- no dependency), V3-EXQ-912 (the
  segment-count workaround this change supersedes as the correct future lever), organism-lifespan
  review Section 10 item 1 (origin).
