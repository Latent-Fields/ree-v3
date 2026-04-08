# MECH-205 VALENCE_SURPRISE Write Path
**Gap ID:** MECH-205-write
**Complexity:** small (< 1 session)
**Readiness:** ready_to_plan
**Blocking claims:** MECH-205, INV-052 (indirect -- surprise-gated replay is a component)

---

## Problem

EXQ-258 classified non_contributory (governance-2026-04-08): P1 criterion failed --
`VALENCE_SURPRISE` values were not populated in the residue field despite Tier 1
implementation being present in `agent.py`.

The Tier 1 implementation (governance-2026-04-07-session2) wires:
- PE EMA tracking: `self._pe_ema` updated each step
- Write condition: `surprise = max(0.0, pe_mag - self._pe_ema)` -> `residue_field.update_valence()`

Two plausible failure modes (need diagnosis before fix):

**Failure mode A: `prediction_error` key absent from e3_metrics**
`e3_metrics.get("prediction_error")` returns None. If E3Selector does not emit a
`"prediction_error"` key in its metrics dict, the entire write block is skipped silently.
No write -> P1 FAIL.

**Failure mode B: PE never exceeds EMA baseline**
If the environment and agent are in a relatively stable regime, `pe_mag` tracks `_pe_ema`
closely (EMA alpha=0.1 is fast). `surprise = max(0, pe_mag - pe_ema)` is always near 0.
Write is attempted but with ~0 value -> residue field valence stays near zero -> P1 FAIL.

MECH-205 claim reference: `REE_assembly/docs/claims/claims.yaml` MECH-205.
Lit review: `REE_assembly/evidence/literature/targeted_review_mech_205/`
(8 entries, includes McFadyen2023 on valence-asymmetric replay).

---

## What Needs to Exist

### 1. Diagnose the actual failure mode first (Step 0)

Before writing any code, run a single-session diagnostic:

```python
# In experiment script or manual run:
# Add logging to agent.py step() temporarily:
print(f"e3_metrics keys: {list(e3_metrics.keys())}")
print(f"pe_mag={pe_mag:.4f}, pe_ema={self._pe_ema:.4f}, surprise={surprise:.4f}")
# Run 100 steps in hazard-rich env, check output
```

This determines which failure mode to fix.

### 2a. Fix for Failure Mode A: add prediction_error to E3 metrics

File: `ree_core/predictors/e3_selector.py`

In `E3Selector.evaluate_trajectories()` or wherever `e3_metrics` dict is assembled,
add a `"prediction_error"` key. Candidate definitions:
- **Option 1:** `||z_harm_actual - z_harm_predicted||_2` if E3 has an internal harm prediction
- **Option 2:** `||z_world_new - E2_rollout[-1]||_2` (E2 world-forward discrepancy)
- **Option 3:** `|harm_eval_score - harm_eval_score_prev|` (change in E3 harm score)

Option 3 is simplest and requires no new forward passes. Implement first:
```python
# In e3_selector.py, in select_action() or evaluate_trajectories():
harm_score_now = float(self._last_harm_score) if hasattr(self, "_last_harm_score") else 0.0
pe_proxy = abs(current_harm_score - harm_score_now)
self._last_harm_score = current_harm_score
metrics["prediction_error"] = torch.tensor(pe_proxy)
```

### 2b. Fix for Failure Mode B: EMA alpha and minimum threshold

File: `ree_core/agent.py`

Change default EMA alpha from 0.1 to 0.02 so the baseline is slower to track:
```python
# In __init__:
self._pe_ema_alpha: float = getattr(config, "pe_ema_alpha", 0.02)  # was 0.1
```

Add a minimum absolute threshold to filter out near-zero surprises:
```python
# In step() MECH-205 block:
surprise = max(0.0, pe_mag - self._pe_ema)
if surprise > getattr(self.config, "pe_surprise_threshold", 0.01):
    self.residue_field.update_valence(
        z_world, VALENCE_SURPRISE, surprise, hypothesis_tag=False
    )
```

### 3. Counter for diagnostics

File: `ree_core/agent.py`

Add `self._surprise_write_count: int = 0` to `__init__`. Increment in the write block.
Expose via agent metrics or experiment script logging.

---

## Inputs and Outputs

| Signal | Direction | Source | Notes |
|--------|-----------|--------|-------|
| `e3_metrics["prediction_error"]` | in | E3Selector | Must exist; currently may be absent |
| `_pe_ema` | state | agent.py | EMA of PE magnitude |
| `surprise` (float) | computed | agent.py | `max(0, pe_mag - pe_ema)` |
| `VALENCE_SURPRISE` write | out | residue_field | Populates component index 3 |

---

## Training Signal

None -- VALENCE_SURPRISE is a write operation. The residue field accumulates values;
replay prioritization reads them. No gradient flow.

---

## Config Knobs

All in `REEConfig` (ree_core/utils/config.py):

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `pe_ema_alpha` | float | 0.02 | Was hardcoded 0.1; slower baseline tracks better |
| `pe_surprise_threshold` | float | 0.01 | Minimum surprise magnitude to write |

Both default to backward-compatible behavior when `surprise_gated_replay=False`.

---

## Smoke Test

Run EXQ-258 smoke test (2 conditions x 1 seed, 500 episodes):
1. `surprise_gated_replay=True`, `pe_ema_alpha=0.02`
2. Check `agent._surprise_write_count > 0` after 500 episodes
3. Check `residue_field.evaluate_valence(z_world)[..., 3].mean() > 0` at hazard locations
4. Check P1 criterion: `surprise_populated = (valence_surprise_mean > 1e-3)`

If P1 passes, queue EXQ-258a as the 3-seed full validation.

---

## Dependent Experiments to Re-queue

| EXQ | Claim | Action |
|-----|-------|--------|
| EXQ-258 | MECH-205 | Queue EXQ-258a: 2-condition x 3-seed, same design, fixed write path |

Note: EXQ-258a should also verify P2 (surprise-weighted replay samples are more
diverse) and P3 (replay-prioritized regions overlap hazard map).
