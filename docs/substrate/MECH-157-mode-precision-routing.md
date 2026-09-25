## MECH-157 Mode-Conditioned Precision Routing on z_world, option A (2026-09-25)

- MECH-157: latent.mode_precision_routing -- IMPLEMENTED 2026-09-25.
  `ree_core/latent/stack.py` (`LatentStack.encode`, new kwargs `operating_mode`,
  `world_e2_anchor`; new `LatentState.mode_precision_diag`) and
  `ree_core/agent.py` (`REEAgent.sense` mode source + E2 anchor;
  `REEAgent.mode_precision_routing_override`).
  Config: `LatentStackConfig.use_mode_precision_routing` (default False; True to
  enable), `mode_alpha_world` (per-mode ABSOLUTE alpha_world; default
  external_task 0.9 / internal_planning 0.5 / internal_replay 0.2 /
  offline_consolidation 0.1), `mode_world_e2_coupling` (per-mode pull toward
  the E2 forward prediction; default 0.0 / 0.3 / 0.6 / 0.8),
  `mode_world_e2_anchor_norm_cap` (default 1.0; <= 0 disables). All four reach
  `REEConfig.from_dims()` via `kwargs.pop` beside the SELF-1 knobs.
  Data flow: mode source (driver override, else SD-032a
  `SalienceCoordinator.operating_mode` at its previous tick) + anchor
  `E2.world_forward(z_world_prev, a_prev)` (no_grad, detached) ->
  `LatentStack.encode` ->
  `z = (1-g)*(alpha*z_obs + (1-alpha)*z_prev) + g*z_pred`, with
  `alpha = sum_m p_m*mode_alpha_world[m]` (absent mode -> base alpha_world) and
  `g = sum_m p_m*mode_world_e2_coupling[m]` (absent mode -> 0) ->
  `LatentState.z_world` -> every z_world consumer (E2, E3, residue, hippocampus).
  Backward compatible: disabled by default; OFF, `sense()` computes no mode and
  no anchor and `encode()` runs the legacy line byte-for-byte. ON with no mode
  supplied (no coordinator, no override) is also the legacy blend.
  Phased training required: no (no new trainable parameters).
  MECH-094: not applicable -- waking encode only, no replay/memory write.
  Validation experiment: NONE queued, deliberately -- user decision
  rec-20260925-6231be2b: do NOT queue EXP-0861 until modes change behaviour
  under coordinator-driven transitions (MECH-157 precondition).
  See MECH-157, MECH-245 (same locus, pathological pole), SD-008 (floor scoped
  to external_task), MECH-249, SD-032a, MECH-267.

**Design source.** User decision 2026-09-25 ~08:00Z (rec-20260925-6231be2b),
option A of `REE_assembly/evidence/planning/claim_synthesis_MECH-157_20260925.md`
as revised by its red-team `claim_synthesis_MECH-157-039_redteam_20260925.md`.
Two independent scalars so the claim's two dimensions are not collapsed. "1 -
alpha" is the agent's own previous state (temporal smoothing), NOT hippocampal
content (red-team D3); the generative term is a separate pull toward the E2
kernel the hippocampus chains into rollouts, copying the SELF-1 / DR-13
`self_e1_anchor` pattern. Absolute alpha (not a multiplier on the shipped 0.3)
so external_task sits at the SD-008 floor (red-team D2).

**Deliberately not gated on `HippocampalConfig.mode_conditioning_enabled`** so
the MECH-267 rollout consumers and this E1-state consumer can be ablated one at
a time, as the MECH-157 falsifier's manipulation requires.

**Anchor norm cap -- an engineering guard, measured.** The pull closes a loop
`z_t <- g*E2(z_{t-1})`, and `world_forward` is `z + delta(z)`, so an expansive
E2 compounds. 200 real ticks, untrained E2, cap off: `||z_world||` went
0.08 -> 1.4e3 (internal_replay) and 0.04 -> 2.0e10 (offline_consolidation),
while external_task / internal_planning stayed ~0.4-0.6. With the cap
(anchor keeps its direction, norm <= cap*||z_obs||) every mode stayed ~0.35-0.38,
and `||z|| <= ((1-g)*alpha + g*cap)*||z_obs|| / (1-(1-g)*(1-alpha))` is
bounded for any g < 1, alpha > 0. With an untrained E2 the cap binds on ~99% of
internal-mode ticks (`mode_precision_diag["anchor_capped"]`); a driver that
wants the raw pull should train `world_forward` first (SD-056 / SD-PP-B5) and
report `anchor_capped`. Non-finite or mis-shaped anchors are refused (no pull).

**Liveness, measured 2026-09-25 (60 real ticks, CausalGridWorldV2 size 8, seed 0,
alpha_world=0.9 base, forced-mode override, knob ON).** Committed actions that
differ from the external_task arm: internal_planning 23/60, internal_replay
34/60, offline_consolidation 34/60, 50/50 external/planning mixture 0/60
(max |dz_world| 0.03). OFF with a replay override: 0/60, z_world identical.
So a FORCED mode now changes actions through this consumer.

**What is still not live (the EXP-0861 precondition).** With
`use_salience_coordinator=True` and no override, the coordinator's
`operating_mode` stayed at one stationary vector for 200 ticks
(external_task 0.475, each other mode 0.175 -> alpha_eff 0.57 on 199/200
ticks). There was no coordinator-driven within-life transition, so the
falsifier's required event still does not occur. Note also that this
stationary soft vector puts alpha_eff (0.57) BELOW the SD-008 floor while
external_task is the argmax mode: under the coordinator the floor holds only
insofar as external_task probability approaches 1. Relative to forcing
external_task, the coordinator-driven arm changed 10/200 actions -- a
static-blend effect, not a mode transition.

**Readouts.** `LatentState.mode_precision_diag` (plain floats): `alpha_eff`,
`gen_coupling`, `anchor_present`, `anchor_capped`, `obs_weight`,
`prior_weight`, `pred_weight` (manipulation checks only -- red-team D1: these
are functions of the config and the mode vector, not a DV), plus
`dist_to_obs` = ||z_world - z_obs|| and `dist_to_pred` = ||z_world - z_pred||,
which depend on the trained model and can serve as DVs alongside the SD-008
event-selectivity margin keyed by mode.

**Contracts.** `tests/contracts/test_mech157_mode_precision_routing.py` (13
tests; all 13 fail on the pre-build substrate): OFF bit-identical with a
mode + anchor supplied; ON-no-mode is legacy; ON external_task equals a plain
alpha_world=0.9 stack; exact update formula; a mode change moves both the
sensory blend and the E2 pull; soft-mixture weighting; anchor cap and
non-finite refusal; `from_dims` lands all four knobs (fails if the pop is
removed); agent-level override / coordinator -> E2 anchor -> encode. Flag
registered PROBED in `tests/test_flag_inertness.py`.
