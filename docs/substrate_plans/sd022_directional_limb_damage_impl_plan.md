# SD-022: Directional Limb Damage (Body Extension)
**Gap ID:** SD-022-limb-damage
**Complexity:** medium (1-2 sessions)
**Readiness:** ready_to_plan
**Blocking claims:** SD-011, ARC-030, MECH-112, Q-034, ARC-052 (all indirect via stream separation)

---

## Problem

harm_obs and harm_obs_a are not causally independent. harm_obs_a is currently a
50-dim EMA of the same hazard/resource proximity fields that compose harm_obs --
just with a slower time constant (alpha=0.05 vs alpha=0.1). EXQ-241b confirmed the
ceiling: r2_s_to_a = 0.996 across all seeds even with the harm_history extension.
This is a structural impossibility, not a calibration failure.

The A-delta / C-fiber distinction requires harm_obs_a to derive from a BODY STATE
that is causally independent of the current world signal. C-fibers carry tissue
damage state, not instantaneous proximity. An agent in a safe location with
accumulated tissue damage should have high z_harm_a and near-zero z_harm_s --
a dissociation CausalGridWorldV2 currently cannot produce.

The directional limb design instantiates this: each limb accumulates damage
independently when used through hazards. The damage state is a body property
that persists (slowly healing) even when the agent leaves hazardous areas.
harm_obs_a is re-sourced from the damage state, not from world proximity.

Root diagnosis note from 2026-04-09: "The world and body is not diverse enough
for the signals to separate. There are few signals in the world from which to
develop models of what is going on too." -- SD-022 addresses the body side;
SD-023 addresses the world side.

---

## Design

### Limb model

Four directional limbs [N=0, E=1, S=2, W=3], each with:

```
damage[d] in [0, 1]       -- accumulated tissue damage state
residual_pain[d]           -- pain emitted when using damaged limb regardless of world state
P(move_fails, d) = damage[d] * failure_prob_scale  -- damage degrades effectiveness
```

**Damage accumulation:** when agent moves in direction d and is currently in/adjacent
to a hazard cell (harm_signal > 0):
```
damage[d] = min(1.0, damage[d] + damage_increment * harm_signal)
```
The limb that did the movement work is the one that accumulates damage.

**Residual pain signal:** using limb d with damage[d] > residual_pain_threshold
generates pain INDEPENDENT of world state:
```
residual_pain = sum(damage) * residual_pain_scale  # scalar, emitted in body_obs
```

**Movement failure:** P(fail) = damage[d] * failure_prob_scale. On failure, agent
stays in current cell (step cost still taken). This gives damage behavioral
consequences that force it to matter for planning.

**Healing:** applied every step:
```
damage *= (1 - heal_rate)   # default heal_rate = 0.002 -> ~500 steps to clear
```

**Episode reset:** damage reset to zeros on env.reset(). Episode-local for
simplicity -- within-episode dissociation is sufficient for stream separation tests
and keeps experiment designs comparable to the current setup.

---

## What Needs to Exist

### 1. CausalGridWorldV2: limb damage state

File: `ree_core/environment/causal_grid_world.py`

New `__init__` params:
```python
limb_damage_enabled: bool = False,   # backward compat default
damage_increment: float = 0.15,
residual_pain_scale: float = 0.5,
failure_prob_scale: float = 0.3,
heal_rate: float = 0.002,
residual_pain_threshold: float = 0.05,
```

In `__init__`:
```python
self.limb_damage = np.zeros(4, dtype=np.float32)  # [N, E, S, W]
```

In `reset()`:
```python
self.limb_damage[:] = 0.0
```

In `step()`, after resolving agent movement direction `d`:
```python
if self.limb_damage_enabled:
    # Accumulate damage on the active limb if in/near hazard
    if harm_signal > 0:
        self.limb_damage[d] = min(1.0,
            self.limb_damage[d] + self.damage_increment * harm_signal)
    # Heal all limbs
    self.limb_damage *= (1.0 - self.heal_rate)
    # Movement failure check
    if self.rng.random() < self.limb_damage[d] * self.failure_prob_scale:
        # Limb fails: agent stays in place (position unchanged)
        self.agent_x, self.agent_y = prev_x, prev_y
```

In `_get_obs_dict()`, extend body_state:
```python
if self.limb_damage_enabled:
    residual_pain = float(np.sum(self.limb_damage) * self.residual_pain_scale)
    # Append [damage[0..3], residual_pain] to body_state (5 new dims)
    # body_obs_dim: 12 -> 17
    damage_vec = torch.from_numpy(self.limb_damage.copy()).float()
    pain_scalar = torch.tensor([residual_pain])
    body_state = torch.cat([body_state, damage_vec, pain_scalar], dim=0)
```

### 2. harm_obs_a re-sourcing

**Critical change:** when `limb_damage_enabled=True`, harm_obs_a is derived from
damage state, not from the EMA proximity fields.

```python
if self.limb_damage_enabled and self.use_proxy_fields:
    residual_pain = float(np.sum(self.limb_damage) * self.residual_pain_scale)
    # [damage[4], max_damage, mean_damage, residual_pain] -- 7 dims
    harm_obs_a_body = np.array([
        *self.limb_damage,
        float(np.max(self.limb_damage)),
        float(np.mean(self.limb_damage)),
        residual_pain,
    ], dtype=np.float32)
    result["harm_obs_a"] = torch.from_numpy(harm_obs_a_body)  # [7]
    # NOTE: harm_obs_a_dim changes from 50 to 7 when limb_damage_enabled=True.
    # AffectiveHarmEncoder must be configured with matching harm_obs_a_dim.
```

### 3. AffectiveHarmEncoder: accept variable harm_obs_a_dim

File: `ree_core/latent/stack.py`

`AffectiveHarmEncoder.__init__` already takes `harm_obs_a_dim` as a parameter.
When `limb_damage_enabled=True`, set `harm_obs_a_dim=7` in `LatentStackConfig`.
The encoder architecture is unchanged; only the input dimension changes.

No structural change needed -- the encoder handles this if the config is correct.

### 4. Config wiring

File: `ree_core/utils/config.py`

Add to `REEConfig.from_dims()`:
```python
limb_damage_enabled: bool = False,
damage_increment: float = 0.15,
failure_prob_scale: float = 0.3,
heal_rate: float = 0.002,
```

When `limb_damage_enabled=True`:
- `body_obs_dim` passed to encoders: 12 -> 17
- `harm_obs_a_dim` in `LatentStackConfig`: 50 -> 7

Backward compatible: `limb_damage_enabled=False` leaves all dims unchanged.

---

## The Dissociation This Enables

The critical experimental test (the thing EXQ-241b could never produce):

| Condition | Agent location | Limb state | Expected |
|-----------|---------------|------------|----------|
| A (damage residue) | Safe area | damage[d] = 0.8 (several hazard transits) | z_harm_a high, z_harm_s near zero |
| B (fresh hazard) | Hazard adjacent | damage[d] = 0.0 (new episode) | z_harm_s high, z_harm_a near zero |

r2_s_to_a should drop from 0.996 to well below 0.5.

---

## Inputs and Outputs

| Signal | Direction | Shape | Notes |
|--------|-----------|-------|-------|
| limb_damage (state) | internal | [4] | N/E/S/W damage scalars |
| damage_vec in body_state | obs | [5] | damage[4] + residual_pain appended |
| harm_obs_a (new source) | obs | [7] | body-derived when limb_damage_enabled |
| harm_obs (unchanged) | obs | [51] | world-derived, A-delta analog |

---

## Smoke Test

Run 200 steps in CausalGridWorldV2 with `limb_damage_enabled=True`, `num_hazards=4`.

Pass criteria:
1. After 10+ hazard transits: at least one `damage[d] > 0.1`.
2. `harm_obs_a` (body-derived) has non-zero values when agent is in safe area with damaged limbs.
3. `cosine_sim(z_harm_s, z_harm_a) < 0.5` after 100 warmup steps (genuine dissociation).
4. `harm_obs` (world-derived) returns to near-zero when agent leaves hazard zone.
5. body_obs.shape[-1] == 17 (not 12).

---

## Dependent Experiments

| EXQ | Claim | Notes |
|-----|-------|-------|
| EXQ-241c (new) | SD-011 stream separation | First test of genuine dissociation -- primary validation |
| EXQ-247c (new) | SD-011/SD-012 co-integration | Re-run with damage state |
| EXQ-178c (new) | SD-011 affective vs sensory | Re-run with damage state |
| EXQ-248c (new) | Q-034 threshold | Re-run once z_harm_a is genuine |

---

## Notes

The harm_history extension (EXQ-241a/b) was a proxy for this fix. It is superseded
by SD-022 -- the temporal extension approach cannot produce causal independence,
only temporal decorrelation. SD-022 should be treated as the correct implementation
of the SD-011 affective stream. The harm_history buffer config can remain for
backward compat but should not be used in new experiments once SD-022 is available.

SD-022 does NOT address world model depth (sparse environment). That is SD-023.
Both are needed for the full harm-stream architecture to be testable.
