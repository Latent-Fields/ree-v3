# Substrate Implementation Plans

Generated: 2026-04-08 (substrate-gap-analysis session)
Updated: 2026-04-09 (SD-022 + SD-023 added; ARC-033 agent-integration gap added)

This directory contains implementation plans and gap notes for V3 substrate gaps --
places where experiments returned non_contributory or inconclusive results specifically
because a required substrate was absent, not because the underlying claim is wrong.

Sources audited: claims.yaml (evidence_quality_notes), claim_evidence.v1.json,
specific experiment manifests (EXQ-245/246/248/251/253/258), WORKSPACE_STATE.md
(governance-2026-04-08 session notes), ree-v3/CLAUDE.md, ree-v3-spec.md.

---

## Implementation Plans (ready_to_plan)

| Gap ID | Blocking Claims | Complexity | File |
|--------|----------------|------------|------|
| SD-022-limb-damage | SD-011, ARC-030, MECH-112, Q-034, ARC-052 | medium | [sd022_directional_limb_damage_impl_plan.md](sd022_directional_limb_damage_impl_plan.md) |
| SD-023-env-gradient-texture | MECH-216, ARC-017, MECH-096, MECH-103 | medium | [sd023_environmental_gradient_texture_impl_plan.md](sd023_environmental_gradient_texture_impl_plan.md) |
| ARC-033-agent-integration | ARC-033, SD-003 (full counterfactual pipeline) | medium | [arc033_e2_harm_s_agent_integration_plan.md](arc033_e2_harm_s_agent_integration_plan.md) |
| SD-011-second-source | SD-011 (superseded by SD-022 -- see note) | — | [sd011_second_source_impl_plan.md](sd011_second_source_impl_plan.md) |
| MECH-205-write | MECH-205, INV-052 (indirect) | small | [mech205_surprise_write_impl_plan.md](mech205_surprise_write_impl_plan.md) |
| MECH-120-wiring | MECH-120, MECH-165 (indirect) | small | [mech120_shy_decay_impl_plan.md](mech120_shy_decay_impl_plan.md) |
| MECH-165-reverse | MECH-165, MECH-092 (indirect) | medium | [mech165_reverse_replay_impl_plan.md](mech165_reverse_replay_impl_plan.md) |

**Note on SD-011-second-source vs SD-022:** The harm_history proxy extension (EXQ-241a/b)
is superseded by SD-022 (directional limb damage). EXQ-241b confirmed r2_s_to_a=0.996
even with history -- the problem is causal independence of information source, not
temporal window length. SD-022 is the correct implementation of the SD-011 affective
stream and replaces the second-source plan. Keep the old plan for historical context.

**Recommended implementation order:**
1. SD-022 (body extension, unlocks all harm-stream experiments)
2. SD-023 (env texture, unlocks MECH-216 / ARC-017 / world-model claims) -- independent of SD-022
3. MECH-120-wiring (small, independent)
4. MECH-205-write (small, independent) -- EXQ-258b already queued as threshold fix
5. MECH-165-reverse (depends on MECH-120)

---

## Resolved (implemented this session)

| Gap ID | Resolution | Experiment |
|--------|-----------|------------|
| SD-015-zresource | VALENCE_WANTING gradient in `_score_trajectory()` (wanting_weight config). No separate z_resource latent needed. | EXQ-259 queued (2026-04-08) |

---

## Gap Notes (architecture_unclear)

| Gap ID | Blocking Claims | What's Missing | File |
|--------|----------------|----------------|------|
| MECH-122-spindle | MECH-122 | V3 proxy behavioral hypothesis | [mech122_spindle_gap_note.md](mech122_spindle_gap_note.md) |
| SD-015-zresource | SD-012 (nav), MECH-112, ARC-030 | RESOLVED -- see Resolved table above | [sd015_zresource_gap_note.md](sd015_zresource_gap_note.md) |
| SD-016-context | SD-016, ARC-041, ARC-042, MECH-153 | Gradient flow audit before redesign | [sd016_context_supervision_gap_note.md](sd016_context_supervision_gap_note.md) |

---

## Already Implemented (no gap)

The following items from the expected checklist are **fully implemented** in ree_core/:
Experiments are non_contributory due to upstream dependencies, not missing substrate code.

| Item | Status | Location | Unblocked by |
|------|--------|----------|--------------|
| SD-012 homeostatic drive | IMPLEMENTED | ree_core/goal.py | Needs SD-015 for nav criterion |
| SD-018 resource proximity head | IMPLEMENTED | ree_core/latent/stack.py | Used by EXQ-257 |
| MECH-186 valence_wanting_floor | IMPLEMENTED | ree_core/goal.py + serotonin.py | Needs SD-012 nav working |
| MECH-188 z_goal_inject | IMPLEMENTED | ree_core/goal.py | Needs SD-012 baseline z_goal |

---

## Downstream Claims Unblocked by Each Plan

Completing all four `ready_to_plan` items would unblock retesting of:

**SD-011-second-source:**
- SD-011 (dual streams genuinely distinct)
- MECH-112 (goal latent dissociable from harm)
- ARC-030 (Go/NoGo symmetry with real z_goal)
- ARC-032 (ThetaBuffer prerequisite z_goal_norm >= 0.05)
- MECH-029 (reflective/DMN mode -- pending z_goal working)
- Q-034 (hazard/resource threshold -- EXQ-248b queued)

**MECH-205-write:**
- MECH-205 (surprise-gated replay -- EXQ-258a)
- INV-052 (tonic regulatory necessity -- indirect)

**MECH-120-wiring:**
- MECH-120 (SHY ordering -- EXQ-245a)
- MECH-165 (reverse replay -- requires SHY first)

**MECH-165-reverse:**
- MECH-165 (replay diversity -- EXQ-244a)
- MECH-092 (indirect -- replay carrying hypothesis tag)

**ARC-033-agent-integration:**
- ARC-033 (E2HarmSForward fully agent-integrated)
- SD-003 (counterfactual causal_sig computable via agent.compute_harm_causal_signal())
- MECH-135, MECH-150 (harm attribution downstream)
