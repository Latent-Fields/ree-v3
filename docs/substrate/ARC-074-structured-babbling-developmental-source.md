## ARC-074 structured babbling developmental source (campaign W2a) -- IMPLEMENTED default-OFF (2026-09-25)

**Status:** IMPLEMENTED, default OFF, no consumer on main. Session `bt0925-t1w2a`.
**Plan:** REE_assembly `evidence/planning/coupled_loop_repair_campaign_plan.md` section 3
W2 (W2a); proposed registry row `arc074-structured-babbling-developmental-source`
(section 7 item 4, GFLAG-0504 -- `/governance` owns it). **Form:** the L2 dose of
`evidence/planning/babbling_e2_action_coverage_probe_20260925.md` (0ac69c87446), widened
from {0..3} to all env action classes: that probe's premise 1 found the native Phase-0
generator is the agent's own E3 selection (`argmax % 4`), so stay (4) is never emitted.

**What.** `ree_core/developmental/structured_babbling.py` `StructuredBabbler`: draw a class
uniformly over `n_classes`, hold it for a run length uniform in {1..`max_run`}, repeat.
`next_class()` / `next_action()` (one-hot `[1, n]`), `reset()` abandons the current run.
Randomness only from its own `numpy.random.default_rng(seed)`; never the global RNG.

**Flags** (`REEConfig`, plumbed through `from_dims`): `structured_babbling_enabled`
(False), `structured_babbling_n_classes` (0 -> `config.e2.action_dim`, the env action count
`from_dims` was given), `structured_babbling_max_run` (4), `structured_babbling_seed` (0).

**Data flow.** `REEAgent.__init__` sets `self.structured_babbler` (None when OFF; the module
is not even imported). Nothing in ree_core calls it and it is not wired into any policy or
the InfantCurriculumScheduler. Its consumer (the E2 world-head member's retained babbling
replay, campaign W3) is integration-branch work.

**Contracts:** `tests/contracts/test_structured_babbling.py` B1-B8 (OFF never builds or
imports it; OFF and ON rollouts byte-identical, with a non-blind check; construction and
5,000 draws leave global RNG untouched; class balance 1/n +- 0.02 incl. stay; run lengths
uniform on {1..4}; stream persistence 0.68 +- 0.02; seeded; knob plumbing).
`tests/test_flag_inertness.py`: `structured_babbling_enabled` in PROBED.

**Not claimed.** Nothing about learning. Whether this stream gives a learner action
coverage is the W3 L2R gate's question.
