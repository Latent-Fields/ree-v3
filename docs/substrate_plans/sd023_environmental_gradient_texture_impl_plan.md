# SD-023: Environmental Gradient Texture (World Extension)
**Gap ID:** SD-023-env-gradient-texture
**Complexity:** medium (1-2 sessions)
**Readiness:** ready_to_plan
**Blocking claims:** MECH-216 (E1 predictive wanting), ARC-017 (stream tags),
  MECH-096 (multimodal exteroceptive fusion), MECH-103 (multimodal fusion), E1 world
  model claims generally

---

## Problem

z_world can only be as rich as the world it encodes. CausalGridWorldV2 currently
emits two proximity fields in world_obs: hazard field and resource field. That means
z_world is encoding approximately two scalars regardless of encoder expressiveness.

This limits several claims in distinct ways:

**MECH-216 (E1 predictive wanting):** E1's schema readout head should predict resource
encounters from LSTM hidden state *before* the resource is nearby. But if the only
world features are hazard and resource proximity, there is nothing to predict: at
approach positions, z_world already contains the resource proximity signal. E1 learns
"resource_prox ~ large" not "pattern X in z_world predicts upcoming resource contact."
There is no temporal predictive structure for E1 to model. EXQ-263a will likely FAIL
or produce artifactual salience for this reason, not because MECH-216 is wrong.

**ARC-017 (typed stream separation) / MECH-096 (multimodal fusion):** These claims
require z_world to encode distinct features for different world content. With only
two undifferentiated proximity channels the claim is structurally untestable.

**E1 world model quality generally:** E1's LSTM should build a model of temporal
dynamics in the world. With two proximity fields as input there are almost no
independent dynamics to model -- the world has almost no texture.

Root diagnosis note from 2026-04-09: "There are few signals in the world from which
to develop models of what is going on too." -- This is the SD-023 problem.

---

## Design

### Principle: all objects emit, each type distinctively

Every placed object has its own gradient channel in world_obs. This is principled:
in natural environments all objects have a detectable presence (olfactory, acoustic,
visual texture). Making all objects emit creates continuous world texture rather than
sparse point sources. z_world must then encode a richer state.

### Object types and gradient channels

Extend world_obs with two new field-view channels per new object type:

| Channel block | Source | Dims | Rationale |
|---------------|--------|------|-----------|
| hazard_field_view | hazard proximity (existing) | 25 | |
| resource_field_view | resource proximity (existing) | 25 | |
| landmark_A_field_view | Landmark A proximity | 25 | Navigation anchor, no harm/benefit |
| landmark_B_field_view | Landmark B proximity | 25 | Predictive cue -- biased near resources |

world_obs_dim: 250 -> 300 (50 new dims, 2 new 5x5 field channels).

Landmark object properties:
- **Landmark A ("pillar"):** N_LANDMARKS_A=2-3, placed randomly. Strong proximity
  gradient (scale=1.0), short range (sigma=1.5 cells). Acts as navigation anchor.
  No harm, no benefit.
- **Landmark B ("trace"):** N_LANDMARKS_B=2-3, placed with bias toward resource
  locations (within radius 2 of a resource, prob=0.7). Weak gradient (scale=0.6),
  medium range (sigma=2.5 cells). This creates the predictive co-occurrence that
  MECH-216 requires: E1 can learn that high landmark_B field predicts upcoming
  resource contact because landmark B tends to be near resources.

### Gradient field computation (same mechanism as existing fields)

```python
def _compute_landmark_field(self, landmark_positions, sigma, scale):
    field = np.zeros((self.size, self.size), dtype=np.float32)
    for (lx, ly) in landmark_positions:
        for x in range(self.size):
            for y in range(self.size):
                d2 = (x - lx)**2 + (y - ly)**2
                field[x, y] += scale * np.exp(-d2 / (2 * sigma**2))
    return field
```

The field is static within an episode (landmarks don't move). Recomputed at
episode start. Agent's 5x5 view window is extracted the same way as existing fields.

### harm_obs extension

harm_obs currently: `[hazard_field_view[25], resource_field_view[25], harm_exposure[1]]` = 51 dims.

With SD-023, landmark fields do NOT feed into harm_obs (they are not nociceptive
signals). harm_obs_dim stays 51. Only world_obs grows.

### harm_obs and harm_obs_a still need SD-022

SD-023 makes z_world richer but does NOT fix z_harm_s / z_harm_a independence.
Both SDs are needed. Recommended order: SD-022 first (fixes the harm stream at its
source), SD-023 second (enriches the world model). They are independent and can
be implemented in parallel.

---

## What Needs to Exist

### 1. CausalGridWorldV2: landmark objects

File: `ree_core/environment/causal_grid_world.py`

New `__init__` params:
```python
n_landmarks_a: int = 0,              # backward compat default
n_landmarks_b: int = 0,
landmark_a_sigma: float = 1.5,       # spread of A gradient
landmark_a_scale: float = 1.0,
landmark_b_sigma: float = 2.5,       # spread of B gradient (longer range)
landmark_b_scale: float = 0.6,
landmark_b_resource_bias: float = 0.7,  # prob B placed within 2 cells of a resource
```

In `reset()`, place landmarks:
```python
self.landmark_a_positions = self._place_random(n_landmarks_a)
self.landmark_b_positions = self._place_biased_near_resources(
    n_landmarks_b, self.landmark_b_resource_bias, radius=2
)
# Precompute static gradient fields for this episode
self._landmark_a_field = self._compute_landmark_field(
    self.landmark_a_positions, self.landmark_a_sigma, self.landmark_a_scale)
self._landmark_b_field = self._compute_landmark_field(
    self.landmark_b_positions, self.landmark_b_sigma, self.landmark_b_scale)
```

In `_get_obs_dict()`, add new field views:
```python
if self.n_landmarks_a > 0 or self.n_landmarks_b > 0:
    la_view = self._extract_field_view(self._landmark_a_field)  # [25]
    lb_view = self._extract_field_view(self._landmark_b_field)  # [25]
    result["landmark_a_field_view"] = torch.from_numpy(la_view).float()
    result["landmark_b_field_view"] = torch.from_numpy(lb_view).float()
```

Extend world_state (the flat world_obs used by encoders):
```python
# world_state currently: [body_state | hazard_field_25 | resource_field_25 | ...]
# Extended: append landmark_a_25 and landmark_b_25
# world_obs_dim: 250 -> 300 when landmarks enabled
if self.n_landmarks_a > 0 or self.n_landmarks_b > 0:
    world_state = torch.cat([
        world_state,
        la_view_tensor,
        lb_view_tensor,
    ], dim=0)
```

### 2. REEConfig / experiment configs

When `n_landmarks_a > 0` or `n_landmarks_b > 0`, `world_obs_dim` must be 300, not 250.
Experiments that use landmarks must set this explicitly.

Backward compatible: `n_landmarks_a=0, n_landmarks_b=0` leaves world_obs_dim=250.

### 3. No encoder changes needed

The SplitEncoder (z_world pathway) takes `world_obs_dim` as a parameter. Extending
world_obs from 250 to 300 automatically exposes the landmark gradient channels to
the z_world encoder. The encoder will learn to use them if they carry signal.

No structural changes to LatentStack, AffectiveHarmEncoder, or E3.

---

## Why This Specifically Enables MECH-216

Without SD-023:
- landmark_B does not exist
- The only world feature correlated with resource proximity is the resource proximity field itself
- E1's schema readout learns a redundant function: "resource_prox is high" ≠ prediction

With SD-023:
- landmark_B fields are elevated for ~2-3 cells around each resource, even when the
  agent is outside the resource proximity radius
- E1's LSTM can learn: "high landmark_B field in the recent context predicts upcoming
  resource contact"
- schema_salience should rise when landmark_B is nearby, *before* resource proximity rises
- This is the genuinely predictive wanting signal -- anticipatory, not reactive

The test is cleanest with `landmark_b_resource_bias=1.0` (B always near resources):
salience should spike at landmark_B positions ahead of resource contact.

---

## Smoke Test

Run 200 steps with `n_landmarks_a=2, n_landmarks_b=2`.

Pass criteria:
1. `obs_dict["landmark_a_field_view"]` and `landmark_b_field_view` are non-zero.
2. `world_state.shape[-1] == 300` (not 250).
3. landmark_b positions are within 3 cells of at least one resource (check placement
   bias is working): `assert any(dist(lb, r) <= 3 for lb in landmark_b_positions for r in resources)`
4. Agent can run standard REE agent loop without shape errors.

---

## Dependent Experiments

| EXQ | Claim | Notes |
|-----|-------|-------|
| EXQ-263b (new) | MECH-216 | Re-run with landmark_B as predictive cue; test salience at landmark positions |
| ARC-017 retest | ARC-017 stream tags | Re-run with 4-channel world: typed separation now meaningful |
| MECH-096 retest | MECH-096 three-stream | Re-run with richer world signal |
| MECH-103 retest | MECH-103 multimodal | Landmark channels act as additional modalities |

---

## Notes

The body and world extensions (SD-022 and SD-023) fix orthogonal problems:
- SD-022: harm_obs_a causal independence (body has real state distinct from world)
- SD-023: z_world information depth (world has real structure beyond harm/benefit)

Neither alone is sufficient. Together they establish the minimum environmental/body
richness for the harm-stream and world-model claims to produce interpretable signal.

A possible further extension (not needed for V3): object-presence ambient gradient
where every object emits a small cross-channel "presence" signal independent of its
primary gradient type. This would create truly continuous world texture. Deferred
until SD-022 and SD-023 are validated.
