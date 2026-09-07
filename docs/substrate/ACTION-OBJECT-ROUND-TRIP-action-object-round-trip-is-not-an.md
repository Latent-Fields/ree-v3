## Action-object round trip is NOT an action source + CEM elite floor (2026-07-22)

Two independent substrate defects found while authoring V3-EXQ-800/801, each
independently reproduced. Both make an experiment produce plausible numbers from
an action stream that cannot respond to its own manipulation.

**DEFECT 1 -- the a -> `E2.action_object(a)` -> `action_object_decoder` round trip
is not invertible, so it is not an action source.**
`argmax(action_object_decoder(traj.get_action_object_sequence()[:, 0, :]))` is a
CONSTANT. A driver selecting that way has an action stream INVARIANT under every
manipulation of the candidate set or its scores -- an arithmetically forced no-op.

**NEITHER component is individually degenerate -- the COMPOSITION is.** Both are
untrained, and the action-object distribution is a small ball sitting far from the
decoder's decision boundaries, so the decoder's own bias-argmax class wins for every
input it is actually given.

Do not read the earlier shorthand ("the decoder spans all classes on N(0,1), the
degeneracy is in its INPUT, the embedding is near action-invariant") as literal --
measured 2026-07-22 (session `epic-burnell-995d28`; seed 42, `world_dim=32`,
`action_dim=5`, `action_object_dim=16`, `hidden_dim=64`), each half of it is wrong in
a way that misdirects a fix:

- **The decoder does not meaningfully "span all classes".** On N(0,1) inputs it puts
  1362/2000 = **68%** on class 3. Every class gets a nonzero count; that is the whole
  of the claim.
- **That health check is run 28x OUT OF DOMAIN.** Real action-object inputs have
  per-dim std **0.036** against the probe's 1.0, so N(0,1) behaviour says nothing
  about behaviour on the inputs the decoder actually sees.
- **The embedding is NOT "near action-invariant".** On the 5 one-hot actions the
  action-object norms are 0.328-0.421 -- genuinely different per action -- and a
  linear probe recovers the action class from state-centred action-objects at
  **100%** (chance 20%), with action variance at 99.6% of total. The action is fully
  present.
- **What it actually lacks is STATE dependence:** 99.5% of its variance is explained
  by the action label alone, so it is a frozen re-encoding of the action rather than
  the state-conditioned consequence O is supposed to hold.
- **The composition still collapses** because those genuine per-action differences
  move the decoder's logits by std 0.007-0.017 against per-class-mean gaps up to
  0.33. Argmax pins to class 3.

**Training does not repair it** (identical at 0 and 40 warmup episodes), and
**training the decoder alone is still the wrong fix** -- a decoder cannot recover
consequences from an embedding that never encoded them. Full measurements:
`REE_assembly/evidence/planning/action_object_invariance_spike_2026-07-22.md` (Sec.
3.5).

Measured (untrained module, action_dim=5, 32 candidates, seed 42):

| arm | `traj.actions[:,0]` | re-decoded `ao_0` |
|---|---|---|
| default (SP-CEM on) | 2 classes {0,3} | **1 class {3}** |
| SP-CEM off | 1 class {3} | **1 class {3}** |
| action-class scaffold | 5 classes {0..4} | **1 class {3}** |

The scaffold row is decisive -- those candidates are CONSTRUCTED with one distinct
one-hot first action per class and still all re-decode to class 3. Note this is
NOT repaired by `use_support_preserving_cem` or
`use_action_class_scaffold_candidates`: both act on `Trajectory.actions`, where
they work as designed, and neither touches the round trip.

- **Correct accessor:** `HippocampalModule.candidate_first_action_class(traj)`
  (reads `trajectory.actions`, the ground truth of what the candidate is).
- **Correct selection:** `agent.select_action(candidates, ticks)` -- E3's J(zeta),
  returns the action directly, never consults the decoder.
- **Sanctioned decoder use, unchanged:** the CEM proposal path, where the
  CONTINUOUS output feeds `E2.rollout_with_world`. The rollout consumes the
  real-valued vector, so candidates differ even when their argmaxes coincide.
- **Live diagnostic:** `action_object_roundtrip_recovery` in the propose
  diagnostics. `roundtrip_unique_classes == 1` while `true_unique_classes > 1` is
  the inert-selection signature. Read `recovery_rate` ONLY alongside those two
  counts -- most CEM candidates were produced BY the decoder, so they agree with
  it trivially and inflate it (0.875 on a fully collapsed round trip).
- **Lint:** `validate_experiments.py --checks action_object_selection` WARNs on
  the argmax-for-action pattern (advisory in both modes; it cannot tell selecting
  from reporting). 6 fires / 1113 scripts at landing: EXQ-114, 120, 266, 266a,
  800, 801. Exempt with `ACTION_OBJECT_SELECTION_EXEMPT = "<reason>"`.
  **All six now dispositioned:**
  - **EXQ-800 / 801 -- FALSE POSITIVES, exempted.** The argmax lives in an uncalled
    helper; selection routes through `agent.select_action`. Landed `7f10f441a6`.
  - **EXQ-114 / 120 -- CONFIRMED invalidated.** Replaced 2026-07-22 by
    `V3-EXQ-114a` / `V3-EXQ-120a` (scripts on origin/main `f2527c7`; queue entries
    in the coordinator DB).
  - **EXQ-266 / 266a -- CONFIRMED invalidated 2026-07-22** (session
    `hopeful-panini-cf272d`). Both feed
    `argmax(action_object_decoder(get_action_object_sequence()[:,0,:]))` straight to
    `env.step` (`v3_exq_266_q020_valence_geometry_pair.py:269`,
    `v3_exq_266a_q020_valence_geometry_pair_fixed.py:281`) -- a SELECTION source.
    Empirical proof: all three published runs report
    `harm_rate_TERRAIN_SHAPED == harm_rate_TERRAIN_FLAT == 0.0060999999999999995`,
    `harm_reduction_frac` exactly 0.0. **EXQ-266a is decisive** -- it repaired
    EXQ-266's ablation bug and its `terrain_harm_corr` DID move (-0.4384 SHAPED vs
    +0.8415 FLAT), so the manipulation reached the terrain while the behaviour
    stayed bit-identical. Replaced by `V3-EXQ-266b` (`5247cfd413`; queue entry in
    the coordinator DB).
  - **The EXQ-266 signature is NOT the EXQ-114 denominator inflation.** Both 266
    arms ran the same constant policy, so their step denominators were EQUAL and the
    artifact presented as a perfect null rather than a ~14x inflation -- the EXQ-114
    diagnostic would not have caught it. `V3-EXQ-266b` carries an explicit
    `arms_bit_identical` detector for that shape.

**DEFECT 2 -- `num_elite` must be >= 2; below that the CEM is NaN-poisoned.**
torch's default `std()` is UNBIASED, so std over a single elite is NaN. It
propagates through `+ 1e-6` and through the SP-CEM `torch.clamp` ao_std floor
(clamp propagates NaN), poisoning ao_std, every later candidate, and the rollouts.
Measured end state: `RuntimeError: probability tensor contains either inf, nan or
element < 0` from `torch.multinomial` inside `e3_selector.select`.

Reachable from the SUBSTRATE DEFAULT, silently: `num_candidates=8` x the default
`elite_fraction=0.2` gives `int(1.6) == 1`. Measured at 10 / 40 / 120 warmup
episodes alike, 1 of 8 candidates scored finite.

Fix: `num_elite = min(n, max(2, int(n * elite_fraction)))`, plus
`HippocampalModule._stack_std()` as a NaN-safe backstop at both ao_std sites
(the differentiable-CEM path sizes its stack by candidates-with-ao-sequences, not
by `num_elite`). `cem_num_elite` is emitted in the propose diagnostics.
**Cannot regress a working configuration** -- every configuration it touches was
already NaN-poisoned.

Contract: `tests/contracts/test_action_object_roundtrip_not_an_action_source.py`
(9 tests). Both defects pin the DECODER's health separately from the ROUND TRIP's,
so a future session cannot "fix" this by training the wrong component.

**EXQ-196 / ARC-018 consequence.** EXQ-196 does NOT use the round trip -- it uses
`argmax(e3.select(...).selected_action)`. But `selected_action` is the selected
candidate's `actions[:, 0, :]`, i.e. decoder output, so the same upstream fact
reaches its DV by a different route. EXQ-196 ran 2026-04-04, six weeks BEFORE
SP-CEM became the default (cb1c6da, 2026-05-17). Measured in that regime on a
trained agent: **68/68 ticks (100%) the whole 32-candidate set offered exactly ONE
first-action class**, so E3's choice could not reach the executed action and both
arms executed identical streams. Under the current default: mean 2.29 classes,
min 2, 0% single-class ticks. That is a complete mechanical account of
harm_advantage_mean = EXACTLY 0.0 on all 3 seeds at e2_world_r2 0.766, and it
means ARC-018 Leg B was UNTESTABLE in April and is testable now. See
`REE_assembly/docs/claims/claims.yaml` ARC-018 `evidence_quality_note`.
