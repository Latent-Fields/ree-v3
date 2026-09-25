## WAKING-TRAINER: native waking trainer skeleton + harm_eval member (2026-09-25)

**Status:** IMPLEMENTED, default OFF. Evidence domain of the ON mode: D1 (reach) only.
**Design:** REE_assembly `evidence/planning/native_waking_trainer_design_20260925.md`
(0c0f5b76ec + addenda). **Why:** the gradient-reach census (940c690c9dd) measured no
waking gradient learning at `REEConfig` defaults; `e3.harm_eval_head` (read every E3
tick) untrained in 4/5 recipes (GFLAG-0491).

**Flags** (`REEConfig`, plumbed through `from_dims`): `waking_trainer_enabled` (False),
`waking_trainer_every_k` (1), `waking_trainer_harm_eval_lr` (1e-3),
`waking_trainer_batch_size` (16), `waking_trainer_buffer_max` (2000),
`waking_trainer_guard_min_steps` (8; 0 disarms the guard), `waking_trainer_seed` (0).

**Data flow.** `REEAgent.__init__` (last) builds `ree_core/utils/waking_trainer.py`
`WakingTrainer` only when enabled (plain object, not an `nn.Module`).
`update_residue()` -> `_waking_trainer_step()` -> `on_waking_step(harm_signal)`
(waking only; skipped on `hypothesis_tag`): every member records a detached replay
sample; every K waking ticks each ready member takes one optimizer step inside
`torch.random.fork_rng` with a private RNG state (python/numpy states restored),
under `torch.enable_grad()`.

**Members.** `HarmEvalMember` (always, when ON) -- `e3.harm_eval_head(z_world_t.detach())`
MSE against `max(-harm_signal_t, 0)` over uniform replay batches (the tier1 precedent's
pairing; `abs(harm_signal)` would train benefit as harm). NOT registered: E2-world,
SD-070 P0, ZSelfP0, codec, terrain prior -- the coupled integration-branch campaign
plugs them in through `WakingTrainerMember` / `register`.

**T1 members (2026-09-25, bt0925-t1w2a; campaign plan section 3 W-trainer), each
default OFF behind its own knob, read only when `waking_trainer_enabled`:**
- `E1Member` (`waking_trainer_e1_enabled`, `waking_trainer_e1_lr` 1e-3): the NATIVE
  `compute_prediction_loss` over the agent's own detached `_self/_world_experience_buffer`;
  group = every trainable `agent.e1` tensor (write_gate stays in the group so G3 checks
  it on every ON run).
- `E2SelfMember` (`waking_trainer_e2_self_enabled`, `waking_trainer_e2_self_lr` 1e-3):
  records (z_self_t, a_t, z_self_{t+1}) ITSELF from `_current_latent` / `_last_action`
  at each waking `update_residue` (detached at record time -- `_last_action` is undetached
  by default); drops the pending pair at `notify_env_reset` (new one-line hook; `reset()`
  calls it) or when `_step_count` goes backwards; loss = the NATIVE `compute_e2_loss` with
  the member buffer swapped in for the call only. Group = `e2.self_transition` +
  `e2.self_action_encoder`. Its buffer equals the harness `record_transition` stream
  entry for entry (contract T6).
- Retained-graph exposure, stated: an ON update steps parameters in place inside
  `update_residue`; a driver that backprops a live tick graph through a trained module
  AFTER `update_residue` would hit an in-place autograd error. No ree_core path does.

**Guard.** One `GradReachGuard` (`ree_core/utils/grad_reach_guard.py`) per group,
allowlist `FROZEN_BY_DESIGN`, armed for the group's first `guard_min_steps` steps.
FAIL or CANNOT_DETERMINE at window close raises `WakingTrainerReachError`.

**Contracts:** `tests/contracts/test_waking_trainer.py` W1-W6 (OFF builds nothing;
OFF rollout byte-identical to the pre-change path, plus a non-blind check; ON
RNG-neutral; reach + guard PASS; disconnected loss -> moved-check fails and the guard
raises; knob plumbing). `tests/test_flag_inertness.py`: `waking_trainer_enabled` in
PROBED. T1: `tests/contracts/test_waking_trainer_t1_members.py` T1a-T8 (OFF never
constructs the members and a multi-episode rollout is byte-identical to the pre-change
`notify_env_reset`; ON RNG-neutral per update + a non-blind spy; e1 / e2_self reach +
guard PASS + no leak; disconnected loss -> raise / moved-check fails; own-transition
equality with the harness stream; env-reset hook; train-mode run; knob plumbing).
`waking_trainer_e1_enabled` / `waking_trainer_e2_self_enabled` in PROBED.

**Not claimed.** No behavioural effect. At 300 eval-mode ticks the head learns the
harm base rate (loss 0.154 -> 0.018) but does not separate harm from safe ticks.
