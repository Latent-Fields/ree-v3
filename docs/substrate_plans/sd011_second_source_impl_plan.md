# SD-011 Second Source: z_harm_a Contextual Input
**Gap ID:** SD-011-second-source
**Complexity:** medium (1-2 sessions)
**Readiness:** ready_to_plan
**Blocking claims:** SD-011, MECH-112, ARC-030, ARC-032, MECH-029, Q-034

---

## Problem

`AffectiveHarmEncoder` takes only `harm_obs_a` -- an EMA of the agent's current-cell
hazard/resource values (replicated across 25 dims per channel). This is structurally
identical to z_harm_s with temporal smoothing: both encode the same spatial proximity
signal. EXQ-241 and EXQ-247 confirmed D3 reversal: R2_affective > R2_sensory in all
seeds, meaning z_harm_a predicts the SENSORY target better than z_harm_s -- evidence
of monotone redundancy, not genuine stream separation.

The neurobiological motivation (C-fiber / ACC / paleospinothalamic tract) requires
a signal that ACC/insula receive that is NOT accessible to the sensory (S1/VPL) pathway.
In the biological system that signal is: accumulated harm exposure history + contextual
learned harm associations (amygdala, hippocampus).

SD-011 architecture doc: `REE_assembly/docs/architecture/sd_011_dual_nociceptive_streams.md`
Claims: SD-011, ARC-030, MECH-112, ARC-032

---

## What Needs to Exist

### 1. CausalGridWorldV2: harm history buffer

File: `ree_core/environment/causal_grid_world.py`

Add `harm_history_len: int = 10` to `CausalGridWorldConfig`.

In `CausalGridWorldV2.__init__()`:
```python
self._harm_history = np.zeros(self.config.harm_history_len, dtype=np.float32)
self._harm_history_ptr = 0
```

In `CausalGridWorldV2.step()`, after computing `harm_exposure`:
```python
# SD-011: rolling harm history for affective stream
self._harm_history[self._harm_history_ptr % self.config.harm_history_len] = harm_exposure
self._harm_history_ptr += 1
```

In `_get_obs_dict()`, add to obs_dict:
```python
obs_dict["harm_history"] = np.roll(
    self._harm_history,
    -self._harm_history_ptr,
)  # shape [harm_history_len], oldest-first
```

On episode reset, zero `_harm_history` and `_harm_history_ptr`.

### 2. AffectiveHarmEncoder: extended input

File: `ree_core/latent/stack.py`

Update `AffectiveHarmEncoder.__init__()` to accept `harm_history_len: int = 10`:
```python
self.input_dim = harm_obs_a_dim + harm_history_len  # 50 + 10 = 60
self.encoder = nn.Sequential(
    nn.Linear(self.input_dim, 64),
    nn.ReLU(),
    nn.Linear(64, z_harm_a_dim),
)
# Auxiliary loss head: predict cumulative harm (scalar)
self.harm_accum_head = nn.Linear(z_harm_a_dim, 1)
```

Update `AffectiveHarmEncoder.forward()`:
```python
def forward(
    self,
    harm_obs_a: torch.Tensor,    # [batch, harm_obs_a_dim]
    harm_history: torch.Tensor,  # [batch, harm_history_len]
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.cat([harm_obs_a, harm_history], dim=-1)  # [batch, 60]
    z = self.encoder(x)
    harm_accum_pred = self.harm_accum_head(z)  # [batch, 1] -- for aux loss
    return z, harm_accum_pred
```

### 3. LatentStack: pass harm_history through encode()

File: `ree_core/latent/stack.py`

`LatentStack.encode()` signature change:
```python
def encode(
    self,
    obs_self: Tensor,
    obs_world: Tensor,
    harm_obs: Optional[Tensor] = None,
    harm_obs_a: Optional[Tensor] = None,
    harm_history: Optional[Tensor] = None,  # NEW
    ...
) -> LatentState:
```

In the z_harm_a block:
```python
if harm_obs_a is not None and self.affective_harm_encoder is not None:
    hoa = harm_obs_a.to(device).float()
    hh = harm_history.to(device).float() if harm_history is not None \
         else torch.zeros(batch_size, self.config.harm_history_len, device=device)
    z_harm_a, harm_accum_pred = self.affective_harm_encoder(hoa, hh)
    latent_state.z_harm_a = z_harm_a
    latent_state.harm_accum_pred = harm_accum_pred  # stash for aux loss
```

### 4. Agent: extract harm_history from obs_dict and pass to encode()

File: `ree_core/agent.py`

In `_extract_obs_tensors()` (or wherever obs_dict is unpacked):
```python
harm_history = obs_dict.get("harm_history")
if harm_history is not None:
    harm_history = torch.tensor(harm_history, device=self.device).float().unsqueeze(0)
```

Pass `harm_history=harm_history` to `self.latent_stack.encode(...)`.

### 5. Auxiliary loss in training step

File: `ree_core/agent.py` or `ree_core/predictors/e3_selector.py`

Add to harm-stream loss computation:
```python
if latent.harm_accum_pred is not None and config.z_harm_a_aux_loss_weight > 0:
    # target: cumulative harm in current episode (from obs_dict["accumulated_harm"] scalar)
    harm_accum_target = torch.tensor([[accumulated_harm]], device=device)
    aux_loss = F.mse_loss(latent.harm_accum_pred, harm_accum_target.clamp(0, 1))
    total_loss = total_loss + config.z_harm_a_aux_loss_weight * aux_loss
```

CausalGridWorldV2 must also track `accumulated_harm` in obs_dict (running sum of harm_exposure
per episode, normalized by episode length or clipped to [0, 1]).

---

## Inputs and Outputs

| Signal | Direction | Shape | Notes |
|--------|-----------|-------|-------|
| harm_obs_a (EMA proximity) | in | [batch, 50] | existing |
| harm_history (rolling window) | in | [batch, 10] | NEW from env |
| z_harm_a | out | [batch, 16] | latent |
| harm_accum_pred | out | [batch, 1] | aux loss target |

---

## Training Signal

- **Primary:** downstream E3 harm scoring (unchanged -- z_harm_a feeds E3 urgency gate)
- **Auxiliary:** MSE on `accumulated_harm_scalar` from env (normalize to [0,1])
  - Loss weight: `z_harm_a_aux_loss_weight` (default 0.1)
  - Purpose: forces z_harm_a to integrate temporal information, not just spatial proximity
  - This directly creates gradient pressure for temporal divergence from z_harm_s

---

## Config Knobs

All in `REEConfig` / `LatentStackConfig` (ree_core/utils/config.py):

| Field | Type | Default | Location |
|-------|------|---------|----------|
| `harm_history_len` | int | 10 | CausalGridWorldConfig |
| `harm_obs_a_dim` | int | 50 | LatentStackConfig (unchanged for EMA portion) |
| `harm_history_len` (encoder) | int | 10 | LatentStackConfig (mirror) |
| `z_harm_a_aux_loss_weight` | float | 0.1 | LatentStackConfig |

Backward compatibility: if `harm_history` is None in encode(), use zeros. Old experiments
with `use_affective_harm_stream=True` continue to work but get zero harm_history (degraded
but not crashed).

---

## Smoke Test

Run for 200 steps in CausalGridWorldV2 with hazard-rich config (`hazard_density=0.3`).

Pass criteria:
1. `obs_dict["harm_history"]` is non-zero after first hazard encounter.
2. `latent.z_harm_a` is non-None and non-zero.
3. `cosine_sim(z_harm_a.mean(0), z_harm_s.mean(0)) < 0.95` -- streams are genuinely distinct.
4. `harm_accum_pred` is non-None (aux head produces output).

Confirm with: `python experiments/v3_exq_241_sd011_diagnostic.py --dry-run --steps 200`
(or write minimal inline smoke script).

---

## Dependent Experiments to Re-queue

| EXQ | Claim | Action |
|-----|-------|--------|
| EXQ-241 series | SD-011 diagnostic | Queue EXQ-241a: check D3 reversal resolves |
| EXQ-247 | SD-011/SD-012 co-integration | Queue EXQ-247b with `harm_history_len=10` |
| EXQ-248b | Q-034 threshold sweep | Already queued but uses z_harm_a=0; re-run after fix |
| EXQ-178 series | SD-011 affective vs sensory | Queue lettered iteration with new encoder |

EXQ-248b is the designated re-test per governance-2026-04-08 and should be the
primary validation run after implementation.
