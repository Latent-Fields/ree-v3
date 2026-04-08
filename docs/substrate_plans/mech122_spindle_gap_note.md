# MECH-122 Spindle-Equivalent Bursts — Gap Note
**Gap ID:** MECH-122-spindle
**Readiness:** architecture_unclear
**Blocking claims:** MECH-122

---

## Current State

`ree_core/latent/theta_buffer.py` has a `_consolidation_mode` flag and
`consolidation_summary()` method (recency-weighted average of recent z_world states).
This was the V3 proxy spindle implementation tested in EXQ-246 (x2).

EXQ-246 result (governance-2026-04-08): both runs inconclusive (vacuous PASS).
WITH_SPINDLE and NO_SPINDLE conditions produced **identical metrics** in all 3 seeds --
harm rates, harm_eval values, harm discrimination all identical. The proxy has zero
measurable effect.

User confirmed (governance-2026-04-08): V4 mechanisms being pulled into V3 scope.
MECH-122 substrate should be planned for V3 implementation.

---

## What Is Unclear

The MECH-122 claim has two functional requirements:
1. **Packaging**: spindle bursts package E3/hippocampal replay content for z_theta delivery
2. **Gating**: spindle bursts gate EXTERNAL SENSORY INPUT, protecting the consolidation
   episode from interruption

The current proxy only addresses (1) via a weighted average -- it doesn't address (2) at all.
That's why WITH_SPINDLE = NO_SPINDLE: sensory input is never gated in either condition.

For a V3 proxy of (2) to produce measurable signal, we need to decide:

**Option A: Hard gate**
During `enter_sws_mode()`, stop updating z_world from real sensory observations
(i.e., don't call `latent_stack.encode(obs_world=...)` during consolidation).
z_world is frozen at pre-sleep state for the duration of replay.

**Option B: Soft gate**
During consolidation, discount incoming sensory observation weight:
`obs_world_effective = alpha_gate * obs_world + (1 - alpha_gate) * obs_world_last`
with `alpha_gate ~ 0.1` during consolidation vs `1.0` during waking.

**Option C: Interrupt condition**
Define an "interruption event" (e.g., sudden high z_harm_s value). Test whether
WITH_SPINDLE condition is less affected by interruptions than WITHOUT.

The behavioral prediction that distinguishes these options is unclear.
Without a concrete behavioral hypothesis, it is not possible to design a test
that would produce non-vacuous evidence.

---

## What Would Unblock This

1. **Architecture decision document**: an SD-* doc (currently no SD number assigned for
   V3 spindle proxy) specifying which option (A/B/C above) is the V3 implementation.

2. **Behavioral prediction**: a concrete, testable prediction of what a functional spindle
   proxy would produce. For example:
   - "WITH_SPINDLE: replay quality (trajectory coherence) is maintained even when
     random noise is injected into obs_world during consolidation. WITHOUT_SPINDLE:
     noise injection degrades replay trajectories."
   - This would require adding a noise-injection condition to the experiment design.

3. **Literature review on V3-achievable proxy**: The 8-entry `targeted_review_mech_205`
   focuses on replay/surprise. A targeted review specifically on thalamo-cortical
   gating mechanisms (what minimal behavioral proxy captures the "protection" function)
   would help.

---

## Suggested Next Step

Before writing an implementation plan, write a 1-2 page architecture note answering:
"What would a V3 proxy for MECH-122's gating function need to do, and what experiment
would distinguish it from the null condition?"

This is a design task, not an implementation task. Assign to a governance+claims session,
not a code session.
