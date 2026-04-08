# SD-015 z_resource Separation — Gap Note
**Gap ID:** SD-015-zresource
**Readiness:** architecture_unclear
**Blocking claims:** SD-012 (navigation criterion), MECH-112 (goal alignment), ARC-030

---

## Current State

**No implementation in ree-v3.** Searches of `ree_core/latent/stack.py` find no
`z_resource`, `resource_indicator`, or `resource_encoder` code. SD-018 (resource
proximity head on z_world) is implemented and distinct from SD-015.

**Literature exists:** `REE_assembly/evidence/literature/targeted_review_sd_015/`
has 3 entries including Whittington2022 on cognitive map building. No architecture
specification document has been written.

**Where SD-015 appears in the evidence record:**
DR-2 (design_forward_analysis_2026-04-06.md): "z_goal norm and alignment are independent
failure modes; SD-015 (resource indicator encoding) is separate prerequisite."

EXQ-085 series (085h through 085l, all FAIL C2): SD-012 seeding (C1 z_goal_norm > 0.1)
passes, but navigation criterion (C2 benefit_ratio) fails. Interpretation: drive_level
successfully seeds z_goal in terms of norm, but z_goal doesn't encode WHERE to navigate.
The resource proximity head (SD-018) gives z_world a resource signal but doesn't give
the goal latent a resource-directional representation.

---

## What Is Unclear

The design of z_resource is ambiguous along several dimensions:

**Dimension 1: What does z_resource represent?**
- Option A: Resource proximity in the current field of view (spatial, like z_harm_s)
  -- but this is already partially handled by SD-018
- Option B: Relative direction to nearest resource from current position (a distance
  vector, not a proximity scalar)
- Option C: A learned resource value map extracted from memory (hippocampal, spatial)

**Dimension 2: How does z_resource inform z_goal?**
- The goal pathway currently: `drive_level * benefit_exposure -> z_goal seeding`
- Where does z_resource enter? Does it replace `benefit_exposure`? Supplement it?
  Provide a directional gradient for trajectory proposals?

**Dimension 3: Is SD-015 a latent stream or an auxiliary head?**
- SD-018 is an auxiliary head ON z_world (predicts max resource proximity)
- SD-015 may require a SEPARATE resource encoder (parallel to z_harm) producing z_resource
  as a dedicated latent that E3 trajectory scoring can use directly

**Dimension 4: Training signal?**
- If z_resource is a separate latent, what is its supervised target?
- Option: predict resource distance (regressor on grid cell distance to nearest resource)
- Option: contrastive -- high resource states vs. low resource states

The CausalGridWorldV2 already provides `resource_field_view` (25-dim). The question is
whether this should produce z_resource (a latent parallel to z_harm_s) or whether
SD-018's auxiliary head on z_world is sufficient.

---

## What Would Unblock This

1. **Architecture decision doc** for SD-015: write `sd_015_zresource_separation.md`
   in `REE_assembly/docs/architecture/` defining which option above and how z_resource
   connects to the goal pathway.

2. **Diagnose EXQ-085 failure more precisely**: with SD-018 now implemented
   (resource_proximity_head), does z_goal_norm still fail C2? If EXQ-257 (SD-018
   validation, queued) passes, it may show SD-018 alone resolves the alignment gap,
   making SD-015 (a full z_resource stream) unnecessary.

3. **Gate on EXQ-257 result**: EXQ-257 is the SD-018 WITH/WITHOUT ablation with
   phased training. If C2 navigation criterion passes in EXQ-257, SD-015 may not be
   needed as a separate stream.

---

## Suggested Next Step

1. Wait for EXQ-257 result.
2. If EXQ-257 C2 passes: SD-015 is superseded by SD-018. Mark SD-015 as "resolved
   via SD-018" in claims.yaml.
3. If EXQ-257 C2 fails: write SD-015 architecture doc specifying a dedicated
   resource direction encoder (z_resource latent, parallel to z_harm_s), then
   convert this gap note to a full implementation plan.

This is a decision-gated gap -- do not implement before EXQ-257 is reviewed.
