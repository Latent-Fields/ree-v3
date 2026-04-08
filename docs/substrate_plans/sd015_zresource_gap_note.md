# SD-015 z_resource Separation — Gap Note
**Gap ID:** SD-015-zresource
**Readiness:** RESOLVED via VALENCE_WANTING gradient (see below)
**Blocking claims:** SD-012 (navigation criterion), MECH-112, ARC-030

---

## Resolution (2026-04-08)

SD-015 (a dedicated z_resource latent encoder) is **not required**. The directional
resource signal is provided by the VALENCE_WANTING component of the residue field,
populated by SerotoninModule.update_benefit_salience() during waking steps.

**Implementation:** `HippocampalModule._score_trajectory()` now accepts a
`wanting_weight` config field (HippocampalConfig, default 0.0). When > 0, the mean
VALENCE_WANTING value along the trajectory is subtracted from the terrain score,
biasing CEM selection toward resource-proximal (high-wanting) regions.

```python
# ree_core/hippocampal/module.py, _score_trajectory()
if self.config.wanting_weight > 0:
    batch, horizon, world_dim = world_seq.shape
    flat = world_seq.reshape(batch * horizon, world_dim)
    valence_flat = self.residue_field.evaluate_valence(flat)
    wanting_score = valence_flat[..., VALENCE_WANTING].mean()
    return terrain_score - self.config.wanting_weight * wanting_score
```

**Why this works instead of SD-015:**
- VALENCE_WANTING accumulates from actual resource encounters (episodic, context-sensitive)
- SD-018 (resource proximity head) makes z_world geographically coherent for resource
  locations, ensuring VALENCE_WANTING tags cluster correctly in latent space
- The combined effect: CEM generates trajectories toward z_world regions that have
  previously yielded resources -- directed navigation without a separate encoder

**Validation experiment:** V3-EXQ-259 (queued 2026-04-08) tests WITH_WANTING
(wanting_weight=0.4) vs WITHOUT (ablation), 3 seeds, 100 ep warmup + 50 ep eval.

---

## Original Gap Description (archived)

The claim was that z_goal norm and z_goal alignment are independent failure modes
(DR-2 from design_forward_analysis_2026-04-06.md), and that a dedicated z_resource
latent was needed for goal alignment. The SD-018 implementation showed resource
proximity can be encoded in z_world via auxiliary supervision. The VALENCE_WANTING
gradient approach leverages this to provide directional navigation without a separate
latent stream.

If EXQ-259 FAILS, re-evaluate whether a separate z_resource encoder is needed.
Gate: EXQ-259 result.
