# MECH-120 SHY Decay Wiring
**Gap ID:** MECH-120-wiring
**Complexity:** small (< 1 session)
**Readiness:** ready_to_plan
**Blocking claims:** MECH-120, MECH-165 (indirect -- SHY must run before diverse replay)

---

## Problem

`E1DeepPredictor.shy_normalise(decay)` is fully implemented (ree_core/predictors/e1_deep.py,
lines 283-304). It normalizes ContextMemory slot weights toward the slot-mean, flattening
dominant attractors before replay repopulates them (Tononi & Cirelli SHY hypothesis).

`REEAgent.enter_sws_mode()` (agent.py) calls:
1. `self.e1.enter_offline_mode()`  -- suppresses waking writes
2. `self.serotonin.enter_sws()`    -- 5-HT dynamics

But it does NOT call `self.e1.shy_normalise()`. The two-phase ordering requirement:
- Phase 1: SHY normalization (flatten dominant attractors)
- Phase 2: replay (repopulate with diverse content)

Without Phase 1, replay just reinforces the already-dominant trajectory -- the exact
monopoly problem MECH-120 is designed to prevent. EXQ-245 (x2) ran and was
non_contributory because this wiring was absent.

User confirmed (governance-2026-04-08): MECH-120 being pulled into V3 scope.

MECH-120 claim reference: `REE_assembly/docs/claims/claims.yaml`
Depends on: MECH-030, MECH-092, MECH-094, ARC-016

---

## What Needs to Exist

### 1. Call shy_normalise() in enter_sws_mode()

File: `ree_core/agent.py`

In `enter_sws_mode()`, add after `enter_offline_mode()` and before any replay:
```python
def enter_sws_mode(self) -> None:
    """Enter slow-wave sleep analog. Phase 1: SHY normalization. Phase 2: replay."""
    # Phase 0: gate waking writes
    self.e1.enter_offline_mode()
    self.serotonin.enter_sws()
    self._sleep_mode = "sws"

    # Phase 1: SHY normalization (MECH-120)
    # Must run BEFORE any replay call -- flattens dominant attractors.
    if getattr(self.config, "shy_enabled", False):
        self.e1.shy_normalise(
            decay=getattr(self.config, "shy_decay_rate", 0.85)
        )
```

The existing `_do_replay()` call site (wherever SWS replay is triggered) remains
unchanged -- it naturally runs after `enter_sws_mode()` has been called, satisfying
the Phase 1 -> Phase 2 ordering.

### 2. Two new config fields

File: `ree_core/utils/config.py` in `REEConfig`:

```python
# MECH-120: SHY-analog synaptic homeostasis
shy_enabled: bool = False          # master switch (default off for backward compat)
shy_decay_rate: float = 0.85       # EMA decay toward slot-mean; 0.85 per Tononi SHY lit
```

Literature anchor for default: Tononi & Cirelli (2014) SHY hypothesis -- synaptic
weights are reduced by ~10-20% per cycle. decay=0.85 produces 15% reduction toward
mean, compatible with that range.

### 3. Verify ordering in experiment scripts

Any experiment testing MECH-120 must:
- Set `shy_enabled=True` in REEConfig
- Call `agent.enter_sws_mode()` before replay loops (currently this should already
  be the case if the sleep API is used correctly)
- NOT call `shy_normalise()` a second time mid-session

No changes needed to experiment script structure if `enter_sws_mode()` is already
called correctly -- the config flag is sufficient.

---

## Inputs and Outputs

`shy_normalise()` operates in-place on `e1.context_memory.memory.data`:
```python
# From e1_deep.py lines 283-304 (existing implementation):
def shy_normalise(self, decay: float = 0.85) -> None:
    with torch.no_grad():
        m = self.context_memory.memory.data  # [num_slots, memory_dim]
        slot_mean = m.mean(dim=0, keepdim=True)
        m.data = slot_mean + (m - slot_mean) * decay
```

- Input: none (reads from self.context_memory.memory.data)
- Output: none (modifies in-place)
- No gradient: `.data` write bypasses autograd

---

## Training Signal

None. SHY is a normalization operation, not a trainable parameter. The indirect effect:
after SHY normalization, replay-driven E1 updates build on a flattened prior, allowing
diverse trajectories to claim previously dominant memory slots.

---

## Config Knobs

| Field | Type | Default | Location |
|-------|------|---------|----------|
| `shy_enabled` | bool | False | REEConfig |
| `shy_decay_rate` | float | 0.85 | REEConfig |

---

## Smoke Test

1. Create agent with `shy_enabled=True`, `shy_decay_rate=0.85`.
2. Run 50 waking episodes to populate E1 context memory (slots non-uniform).
3. Record `e1.context_memory.memory.data.norm()` as `norm_before`.
4. Call `agent.enter_sws_mode()`.
5. Record `e1.context_memory.memory.data.norm()` as `norm_after`.
6. Assert `norm_after < norm_before` (SHY reduces deviation from mean).

One-liner test: variance of slot norms should decrease. Check
`slot_norms_before.std() > slot_norms_after.std()`.

---

## Dependent Experiments to Re-queue

| EXQ | Claim | Action |
|-----|-------|--------|
| EXQ-245 x2 | MECH-120 | Queue EXQ-245a: SHY_THEN_SWS vs SWS_NO_SHY with `shy_enabled=True` |

EXQ-245a design is identical to EXQ-245 except `shy_enabled=True` replaces the absent
SHY proxy. Use 3 seeds (matching EXQ-245 design), same s1/s2 criteria.

Note: EXQ-245a is also a prerequisite for MECH-165 diversity experiments -- SHY must
be working before replay diversity can be meaningfully tested (MECH-165 depends on MECH-120).
