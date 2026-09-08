## MECH-463: E3 commit-gate + per-candidate channel-term diagnostics (arousal-conditioned variance decomposition instrumentation) (2026-07-18)

Diagnostics-only extension of the existing V3-EXQ-571 `e3_score_decomp_enabled` flag
(`ree_core/predictors/e3_selector.py`; default False). No new config params, no new SD,
no behaviour path touched. Added so MECH-463 -- "the three global-scalar affective routes
(D1/D2 gain, harm-urgency threshold shrinkage, LC-NE temperature) are a channel-agnostic
VARIANCE AMPLIFIER of the already-dominant selection channel, not a source of behavioural
differentiation" -- becomes measurable. Per the 2026-07-18 instrumentation audit it was
NOT computable from existing runs.

Three additions, all inside `if self.e3_score_decomp_enabled:`:
1. Commit-gate scalars into `last_score_diagnostics`: `urgency_applied`,
   `effective_threshold`, `commit_variance`, plus `commit_gate_mode`
   ("harm_score_variance" | "world_variance") and `committed`. These were pure locals
   that died at end of `select()`; `urgency_applied` escaped only as
   `SelectionResult.urgency`. `commit_variance` is the quantity the gate ACTUALLY
   compared against `effective_threshold` under either commit mode.
2. `self.last_channel_terms`: the per-candidate [K] channel-bias tensors (`_lcg_terms`)
   retained UNREDUCED, keyed by channel name. This is what makes the decomposition
   covariance-correct -- `agent.py:6816-6845` keeps only marginal scalars, losing
   channel-F covariance. Detached clones (the tensors stay live in the recompose /
   eligibility paths).
3. Confirmed `last_score_decomp["per_candidate"][i]["f"]` already isolates the
   per-candidate F component once the flag is on (verified: K-length vector).

Backward compatible: VERIFIED bit-identical with the flag OFF -- seeded 3-seed x 60-step
behavioural-trace digests (actions, final E3 scores, commit flags) match pristine HEAD
exactly. NOTE when writing such a check: `CausalGridWorld` carries its OWN
`np.random.default_rng(seed)`, so omitting its `seed=` kwarg makes the trace
non-reproducible run-to-run and any bit-identity comparison meaningless.

MECH-463 NON-VACUITY GATE -- validated runnable config (inherited from V3-EXQ-643a).
The probe is vacuous, and returns a spurious FLAT F-share (a FALSE REFUTES), unless all
three hold. Measured GREEN at 400 ticks with:
  - `urgency_weight=0.12` + `use_affective_harm_stream=True` -> `urgency_applied`
    non-constant (45 distinct, 0.036-0.128, no saturation). At `urgency_weight=0.5` it
    SATURATES against `urgency_max` (median == p90 == 0.5) and the top deciles collapse.
  - `use_e3_score_diversity` + `use_e3_diversity_entropy_bonus` (MECH-341) -> the only
    channel that genuinely carries cross-candidate range (mech341 range ~0.465).
    Curiosity/tonic-vigor biases alone are UNIFORM across candidates
    (`score_bias_abs_mean=0.025`, `score_bias_range_mean=0.0`), so
    `modulatory_authority_active` can never fire and every non-F channel contributes
    zero variance -> F-share trivially 100% at every decile.
  - SD-056 `e2_action_contrastive_enabled` + `e2_rollout_output_norm_clamp_enabled`
    keeps E3 scores bounded (raw range ~0.034, not the ~1e32 that killed V3-EXQ-643 by
    float32 catastrophic cancellation).
DRIVER CONSTRAINT: `act_with_split_obs()` calls `sense(obs_body, obs_world)` with NO
`obs_harm_a`, so `z_harm_a` is None on that path and `urgency_applied` is pinned at 0.
A MECH-463 driver MUST feed the harm stream explicitly via
`sense(..., obs_harm=..., obs_harm_a=..., obs_harm_history=...)` and then replicate the
wrapper (`clock.advance()` -> `_e1_tick` -> `generate_trajectories` -> `select_action`).

Validation experiment: V3-EXQ-785 (queued). See MECH-463 in
REE_assembly/docs/claims/claims.yaml, MECH-439 (the F-dominance variance monopoly this
claim says arousal amplifies rather than breaks), MECH-359, MECH-390, SD-011.
