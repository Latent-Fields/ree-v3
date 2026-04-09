# ARC-033: E2HarmSForward Agent Integration
**Gap ID:** ARC-033-agent-integration
**Complexity:** medium (1 session)
**Readiness:** ready_to_plan
**Blocking claims:** ARC-033, SD-003 (full counterfactual attribution pipeline)

---

## Problem

`ree_core/predictors/e2_harm_s.py` contains a working `E2HarmSForward` module
(validated: EXQ-166e PASS, delta_r2=0.641; EXQ-195 harm_forward_r2=0.914). However,
the module is **not wired into `REEAgent`**. `agent.py` has zero references to
`E2HarmSForward`, `counterfactual_forward()`, or any SD-003 causal_sig computation.

Current state:
- `E2HarmSConfig` + `use_e2_harm_s_forward` flag exist in `config.py`
- `LatentStackConfig.use_e2_harm_s_forward` is wired through `REEConfig.from_dims()`
- The flag is set in experiment configs but triggers nothing in the agent
- Experiments (e.g. EXQ-264) instantiate `E2HarmSForward` as a standalone module,
  train it outside the agent, and call `agent.e3.harm_eval_z_harm_head()` directly

This means the SD-003 counterfactual attribution pipeline does NOT exist within the
agent's normal sense/act loop. EXQ-195 had excellent forward_r2=0.914 but
attribution_gap=-0.044 because the causal signal is not routed anywhere meaningful.

Root cause of EXQ-262b `fwd_r2<-0.89` was the MECH-220 hub architecture (cross-stream),
not E2HarmSForward itself -- the forward model component is sound.

---

## Required Implementation

### 1. Agent instantiation (`agent.py`)

In `REEAgent.__init__()`, when `config.latent.use_e2_harm_s_forward is True`:
```python
from ree_core.predictors.e2_harm_s import E2HarmSForward, E2HarmSConfig
self.harm_forward: Optional[E2HarmSForward] = None
if self.config.latent.use_e2_harm_s_forward:
    hf_cfg = E2HarmSConfig(
        use_e2_harm_s_forward=True,
        z_harm_dim=self.config.latent.z_harm_dim,
        action_dim=self.config.action_dim,
    )
    self.harm_forward = E2HarmSForward(hf_cfg).to(self.device)
```

Also add `self.harm_forward_opt: Optional[optim.Adam] = None` initialized similarly.

### 2. Training method (`agent.py`)

```python
def compute_harm_forward_loss(
    self,
    z_harm_s_t: Tensor,      # [1, z_harm_dim] -- current, detached
    action_oh: Tensor,        # [1, action_dim]
    z_harm_s_t1: Tensor,      # [1, z_harm_dim] -- next step, detached (stop-gradient)
) -> Optional[Tensor]:
    """P1 training step for E2HarmSForward. Returns None if module disabled."""
    if self.harm_forward is None:
        return None
    z_pred = self.harm_forward(z_harm_s_t.detach(), action_oh)
    return self.harm_forward.compute_loss(z_pred, z_harm_s_t1.detach())
```

Stop-gradient discipline: both `z_harm_s_t` and `z_harm_s_t1` must be detached.
This prevents forward model gradients drifting the HarmEncoder. See e2_harm_s.py docstring.

### 3. SD-003 causal signal method (`agent.py`)

```python
def compute_harm_causal_signal(
    self,
    z_harm_s: Tensor,      # [1, z_harm_dim] -- current sensory harm latent
    action_actual: Tensor,  # [1, action_dim] -- actual action taken
    action_cf: Tensor,      # [1, action_dim] -- counterfactual action
) -> Optional[float]:
    """SD-003: causal_sig = E3(z_harm_s_actual) - E3(z_harm_s_cf).
    Returns None if harm_forward or harm_eval_z_harm_head not available."""
    if self.harm_forward is None:
        return None
    if not hasattr(self.e3, 'harm_eval_z_harm_head'):
        return None
    with torch.no_grad():
        z_harm_s_act = self.harm_forward(z_harm_s.detach(), action_actual.detach())
        z_harm_s_cf  = self.harm_forward.counterfactual_forward(z_harm_s.detach(), action_cf.detach())
        score_act = self.e3.harm_eval_z_harm_head(z_harm_s_act)
        score_cf  = self.e3.harm_eval_z_harm_head(z_harm_s_cf)
    return float((score_act - score_cf).squeeze().item())
```

### 4. Training loop exposure

The agent should expose `harm_forward_parameters()` for experiment-level optimizer
construction, similar to how experiments currently access `agent.latent_stack.parameters()`.

```python
def harm_forward_parameters(self):
    """Yields parameters for E2HarmSForward optimizer, or empty if disabled."""
    if self.harm_forward is not None:
        yield from self.harm_forward.parameters()
```

---

## Phased Training Protocol (unchanged from standalone)

- **P0** (100 ep): HarmEncoder warmup via `agent.compute_prediction_loss()` + event classifier
- **P1** (80 ep): Train `harm_forward` via `compute_harm_forward_loss()` on replay buffer of
  (z_harm_s_t, action, z_harm_s_t1) tuples accumulated during P0. Stop-gradient on both inputs.
- **P2** (20 ep): Evaluate `compute_harm_causal_signal()` at each step -- compare approach vs neutral.

---

## Acceptance

After agent integration, EXQ-264 should be simplified to just:
```python
causal_sig = agent.compute_harm_causal_signal(z_harm_s, action_actual, action_cf)
```
rather than manually calling `harm_fwd` and `agent.e3.harm_eval_z_harm_head()` separately.

Target: forward_r2 > 0.5 (C1), harm_s_cf_gap_approach > harm_s_cf_gap_neutral (C3).
EXQ-195 confirmed forward_r2=0.914 is achievable; the missing piece is C3 (gap direction).

---

## Downstream Claims Unblocked

- **ARC-033** (harm_stream.sensory_discriminative_forward_model): full agent-integrated test
- **SD-003** (counterfactual attribution): causal_sig usable in E3 gating / blame attribution
- **MECH-135, MECH-150** (harm attribution downstream claims): gate on SD-003 working
