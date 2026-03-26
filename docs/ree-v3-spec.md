# ree-v3 Repository Specification

**Created:** 2026-03-16
**Last updated:** 2026-03-26
**Status:** Living specification — launch doc updated with current V3 state
**Repo name:** `ree-v3`
**Governance epoch:** `ree_hybrid_guardrails_v1` (same as V2 — epoch is per-architecture not per-repo)
**Run ID suffix for governance:** `_v3`

---

## 0. Current V3 State (2026-03-26)

This section supersedes the original launch snapshot. Sections 7 (initial experiment queue),
10 (CLAUDE.md content), and 11 (Build Order) are historical — they document what was planned
at V3 launch, not current state. The authoritative session guide is `ree-v3/CLAUDE.md`.

### Substrate Implementation Status

| SD | Subject | Status |
|---|---|---|
| SD-004 | Action objects as hippocampal map backbone | Implemented |
| SD-005 | z_self / z_world latent split | Implemented |
| SD-006 | Asynchronous multi-rate loop (phase 1: time-multiplexed) | Implemented |
| SD-007 | Perspective-corrected z_world (ReafferencePredictor) | Implemented 2026-03-18 |
| SD-008 | alpha_world >= 0.9 in LatentStackConfig | Implemented (EXQ-040 validated) |
| SD-009 | Event-contrastive CE auxiliary loss for z_world encoder | Implemented (EXQ-020 PASS) |
| SD-010 | Harm stream separated as dedicated pathway (z_harm) | Implemented (EXQ-056c/058b/059c PASS) |
| SD-011 | Dual nociceptive streams: z_harm_s + z_harm_a | **PENDING — current bottleneck** |
| SD-012 | Homeostatic drive modulation for z_goal seeding | **PENDING — not yet implemented** |

SD-003 (self-attribution counterfactual pipeline) was validated at EXQ-030b PASS
(world_forward_r2=0.947, attribution_gap=0.035). Redesign now in progress to use z_harm_s
pipeline (post SD-011), since E3 now takes z_harm rather than z_world as primary harm input.

### Experiment Status

- **~96 experiments run** (EXQ-001 through EXQ-096a), covering SD-003 through SD-010 validation,
  heartbeat architecture (SD-006), reafference (SD-007), encoder fixes (SD-008/009), harm stream
  separation (SD-010), wanting/liking dissociation (MECH-112/117), goal conditioning (MECH-116/ARC-032).
- **Currently queued:** EXQ-074b (MECH-112/117 wanting/liking dissociation, supersedes EXQ-074),
  EXQ-076b (MECH-116/ARC-032 E1 goal conditioning, supersedes EXQ-076).
- **SD-011 blocks:** ~10 experiments are FAILing because the single z_harm stream is architecturally
  insufficient for the SD-003 counterfactual redesign. These will be re-run after SD-011 implementation.

### V3 / V4 Scope Boundary

**V3 scope (waking mechanisms):**
- Volatility interrupt / LC-NE analog (MECH-104)
- BG hysteresis and outcome-valence modulation (MECH-106)
- Hippocampal->BG completion coupling (MECH-105, ARC-028)
- Beta gate committed->uncommitted dynamics (MECH-090)
- Trajectory completion signal from HippocampalModule (ARC-028)

**V4 scope (NOT V3):**
- Sharp-wave ripple (SWR) consolidation during sleep
- Sleep-dependent recalibration of commit thresholds
- Theta-gamma coupling during offline replay
- Self-navigation via z_self hippocampal trajectories (ARC-031): gated on EXQ-075 and EXQ-076
  PASS results and Q-022 dissociation test before any implementation

**MECH-124 diagnostic (V4 risk indicator):** When reviewing EXQ-074b and EXQ-076b results,
check whether z_goal salience is competitive with harm salience. If not, this is an early risk
indicator for consolidation-mediated option-space contraction in V4.

### Q-020 Decision

Q-020 adjudicated 2026-03-16: **ARC-007 strict.** HippocampalModule generates value-flat
proposals. Terrain sensitivity is a consequence of navigating residue-shaped z_world, not a
separate hippocampal value computation.

---

## 1. Purpose

V2 proved four core architectural separation claims (MECH-059, -056, -060, -061 all PASS) and ran a complete SD-003 self-attribution experiment series (EXQ-027, EXQ-028 both FAIL), revealing a precise architectural gap: `z_gamma` conflates the agent's own body state with its world footprint. This means:

- SD-003 causal attribution requires a split latent (SD-005)
- Hippocampal planning horizon is bounded by raw z_gamma dimensionality (SD-004)
- The three BG loops cannot be cleanly separated in a single shared latent (SD-006 + SD-005)
- Seven claims cannot be tested until V3 substrate exists (ARC-007, ARC-016, ARC-018, MECH-025, MECH-033, Q-007 — all `hold_pending_v3_substrate` in governance)

ree-v3 implements the three co-designed substrate changes (SD-004, SD-005, SD-006) needed to open those claims to genuine experimental testing.

---

## 2. What V2 Got Right (Preserve These)

These V2 results are genuine and must not regress:

| Claim | V2 result | What it means for V3 |
|---|---|---|
| MECH-059 | PASS | E1 precision and E3 confidence are structurally independent — preserve two-optimizer design |
| MECH-056 | PASS | Residue accumulates along trajectory, not only at endpoint — preserve incremental residue updates |
| MECH-060 | PASS | Write-locus separation between pre/post-commit channels — preserve commit boundary logic |
| MECH-061 | PASS | Commit boundary correctly separates error channels — preserve error routing at commit |
| SD-003 prereq | PASS | CausalGridWorld provides valid ground truth for `transition_type` — reuse and extend |

The V2 module tree (`ree_core/latent/`, `ree_core/predictors/`, `ree_core/hippocampal/`, `ree_core/trajectory/`, `ree_core/residue/`, `ree_core/environment/`) is the right organisational shape. V3 restructures internals, not the package layout.

---

## 3. Three Core Design Decisions

### SD-004 — Action Objects as Hippocampal Map Backbone

**Problem:** HippocampalModule currently navigates raw `z_gamma` state space via CEM. This caps the planning horizon — CEM must operate at full latent dimensionality, and the map has no compressed representation of action consequences.

**Change:** E2 additionally produces *action objects*: `o_t = E2.action_object(z_world_t, a_t)` — a compressed representation of the world-effect of action `a_t` from state `z_world_t`. The hippocampal map is built in action-object space `O`, not raw `z_world` space.

**Interface contract:**
```python
# E2 forward pass produces TWO outputs:
z_self_next = E2(z_self_t, a_t)           # motor-sensory prediction (SD-005)
o_t = E2.action_object(z_world_t, a_t)    # world-effect action object (SD-004)
```

**HippocampalModule** then proposes trajectories as sequences of action objects `[o_t, o_{t+1}, …]`, navigating the compressed world-effect manifold. Planning horizon extends because action-object space is lower-dimensional and semantically grounded.

**Co-dependency with SD-005:** action objects encode `z_world_t → z_world_{t+1}` under `a_t`, which requires `z_world` to exist as a separate channel.

---

### SD-005 — Self/World Latent Split

**Problem:** `z_gamma` conflates proprioceptive/interoceptive self-state (`z_self`) with exteroceptive world-state (`z_world`). This prevents:
- Clean moral attribution (residue should track world-delta, not self-delta)
- Genuine MECH-069 incommensurability (signals are partially correlated in z_gamma)
- Correct SD-003 V3 attribution (`world_delta` requires z_world to exist)

**Change:** Split the latent encoder into two streams:

```
observation → encoder → {
    body-state channels (proprioception, interoception) → z_self  [E2 domain]
    world-state channels (exteroception, env observations) → z_world  [E3/Hippocampus/ResidueField domain]
}
```

**Module responsibilities after split:**

| Module | Latent domain | Error signal | Heartbeat rate |
|---|---|---|---|
| E1 | z_self + z_world (read-only sensory prior) | Sensory prediction error | E1 rate (fast, every frame) |
| E2 | z_self | Motor-sensory prediction error | E2 rate (motor command rate) |
| E3 complex | z_world | Harm + goal error | E3 rate (deliberation rate, slowest) |
| HippocampalModule | action-object space O (indexed over z_world) | Map consolidation (replay) | E3 rate |
| ResidueField | z_world | — (accumulates world_delta) | E3 rate |

**SD-003 V3 attribution pipeline:**
```
# Step 1: E2 provides z_world dynamics
z_world_actual = E2.world_forward(z_world_t, a_actual)
z_world_cf     = E2.world_forward(z_world_t, a_cf)

# Step 2: E3 evaluates harm of each projected world-state
harm_actual = E3.harm_eval(z_world_actual)
harm_cf     = E3.harm_eval(z_world_cf)

# Step 3: causal signature
causal_signature = harm_actual - harm_cf
world_delta      = ||z_world_actual - z_world_cf||
```

Residue accumulates on `world_delta`, not on `causal_delta(z_gamma)` as in V2.

**Note:** `z_beta` (affective latent, arousal/valence) remains shared and continues to modulate E3 precision and (per MECH-093) E3 heartbeat rate. It is NOT split — affective state integrates self and world signals.

---

### SD-006 — Asynchronous Multi-Rate Loop Execution

**Problem:** V1 and V2 use synchronous single-timestep updates — all loops update on the same discrete clock tick. This means ARC-023 (thalamic heartbeat) and its dependent claims (MECH-089–093) cannot be tested; any experiment testing them produces null results by construction.

**Change:** Implement Hierarchical Temporal Abstraction (HTA). Each loop operates at its own temporal grain:

```
E1 (sensorium loop): updates every env step  →  produces z_self, z_world raw estimates
E2 (action-enacting loop): updates every N_e2 env steps (motor command rate)  →  consumes z_self
E3 (planning-gates loop): updates every N_e3 env steps (deliberation rate, N_e3 >> N_e2)  →  consumes theta-cycle summaries of z_world
```

**Cross-rate integration (MECH-089):** E3 does NOT receive raw E1/E2 output. It receives temporally-abstracted summaries:
```python
# After each E1 step, update rolling theta-cycle buffer
theta_buffer.update(z_world_estimate, z_self_estimate)

# At E3 heartbeat tick (every N_e3 steps): E3 samples the buffer summary
z_world_for_e3 = theta_buffer.summary()   # theta-cycle-averaged world state
```

**Beta-gated policy propagation (MECH-090):** E3 continues updating its internal model at each E3 heartbeat. Beta state controls whether that update propagates to action selection:
```python
if not beta_elevated:   # completion event or stop-change signal
    action_selector.update(e3.current_policy_state)
# else: E3 updates internally but output is held
```

**Phase reset (MECH-091):** Salient events (completion, unexpected harm, commitment crossing) call `e3_clock.phase_reset()`, synchronising the next E3 heartbeat to the event.

**SWR replay (MECH-092):** When E3 heartbeat fires with no pending salient event (quiescent cycle), trigger `hippocampal.replay(theta_buffer.recent)` for viability map consolidation.

**Implementation recommendation:** Use time-multiplexed execution with explicit rate parameters as V3 phase 1 (simpler, testable), design toward full HTA as phase 2. Avoid threading for now (Python GIL complications).

```python
# Config parameters added to V3
e1_steps_per_tick: int = 1     # E1 updates every step
e2_steps_per_tick: int = 3     # E2 updates every 3 steps
e3_steps_per_tick: int = 10    # E3 updates every 10 steps
theta_buffer_size: int = 10    # how many E1 steps per theta summary
```

---

## 3a. Additional Substrate Decisions (Post-Launch)

These SDs were registered and implemented during V3 experimentation. They extend §3 and are
equally binding on V3 implementation.

### SD-007 — Perspective-Corrected World Latent

**Problem:** z_world encoder conflates environmental change with self-caused change because it
has no access to the agent's own motor command. Objects entering the field of view look identical
to the agent moving toward objects — both produce z_world change.

**Solution:** ReafferencePredictor in `ree_core/latent/stack.py`. Applied in `LatentStack.encode()`:
```python
z_world_corrected = z_world_raw - reafference_predictor(z_world_raw_prev, a_prev)
```
Input is `z_world_raw_prev` (NOT z_self_prev — z_self cannot predict visual cell content changes).
Biological basis: MSTd receives visual optic flow plus efference copy (Shenoy et al. 2002).
See MECH-098, MECH-101.

### SD-008 — alpha_world >= 0.9

**Problem:** EMA alpha=0.3 in `LatentStack.encode()` makes z_world a ~3-step weighted average,
suppressing event responses to ~30% peak amplitude and making E3 precision invariant to environmental
drift (stuck at ~188 regardless of hazard rate).

**Solution:** `LatentStackConfig.alpha_world` must be >= 0.9 (default 0.3 kept for compatibility;
always set explicitly). `alpha_self` may remain low (body state is genuinely autocorrelated).
ThetaBuffer (SD-006) provides the temporal integration E3 needs — alpha=0.3 was double-smoothing.

### SD-009 — Event-Contrastive Auxiliary Loss

**Problem:** Reconstruction and E1-prediction losses are invariant to harm-relevance; z_world
learns to reconstruct the grid but does not distinguish hazard cells from empty cells.

**Solution:** Event-type cross-entropy auxiliary loss during z_world encoder training. Only
supervised event discrimination forces z_world to carry hazard-vs-empty-cell information.
EXQ-020 PASS: selectivity_margin=0.882, event_classification_acc=0.692.

### SD-010 — Harm Stream Separation

**Problem:** z_world cannot simultaneously represent (a) world-state for trajectory planning and
residue accumulation and (b) harm signal for E3 commit gating. Fused z_world causes E3 harm
evaluation to contaminate the residue field with harm-correlated, not causal-footprint-correlated,
trace geometry.

**Solution:** Dedicated harm pathway: `CausalGridWorldV2` emits `harm_obs` alongside world
observations. `HarmEncoder(harm_obs → z_harm)` trains on proximity labels. E3 takes z_harm as
primary input to `harm_eval()`. ReafferencePredictor does NOT apply to z_harm (harm proximity is
inherently agent-relative — the action is the relevant context, not optical flow correction).
EXQ-056c/058b/059c all PASS.

### SD-011 — Dual Nociceptive Streams (PENDING)

**Status: NOT YET IMPLEMENTED — current primary bottleneck.**

**Problem:** Single z_harm conflates two neurobiologically incommensurable nociceptive pathways
(Melzack & Casey 1968; Rainville et al. 1997 Science gold-standard dissociation):
- z_harm_s (sensory-discriminative, Adelta-pathway analog): immediate proximity/intensity,
  forward-predictable from action. Lateral spinothalamic tract → VPL → S1/S2 analog.
- z_harm_a (affective-motivational, C-fiber/paleospinothalamic analog): accumulated homeostatic
  deviation, not forward-predictable. Medial pathway → CM/PF → ACC/insula/amygdala analog.

**Why EXQ-093/094 found bridge_r2=0:** z_world ⊥ z_harm by SD-010 design.
`HarmBridge(z_world → z_harm)` is architecturally infeasible. This is not a bug.

**Required implementation:**
```
(a) CausalGridWorldV2: add harm_obs_a (EMA harm accumulator, tau=10-30 steps) alongside harm_obs_s
(b) HarmEncoderS (rename current HarmEncoder) + new HarmEncoderA(harm_obs_a → z_harm_a)
(c) LatentState: add z_harm_a field
(d) E2_harm_s: new forward model (z_harm_s, action) → z_harm_s_next (ARC-033)
(e) E3Selector: take z_harm_s (SD-003 counterfactual) + z_harm_a (commit gating) separately
```

**SD-003 redesign after SD-011:**
```python
z_harm_s_cf = E2_harm_s(z_harm_s, a_cf)
causal_sig = E3.harm_eval(z_harm_s_actual) - E3.harm_eval(z_harm_s_cf)
```
Do NOT attempt HarmBridge counterfactuals — bridge_r2=0 is architectural.

### SD-012 — Homeostatic Drive Modulation (PENDING)

**Status: REGISTERED, not yet implemented.**

**Problem:** `GoalState.update()` does not use `drive_level`. `benefit_exposure` (EMA alpha=0.1
of raw benefit signals) never reliably crosses `benefit_threshold=0.05` during random-walk warmup:
single resource contact produces benefit_exposure ~0.03, which decays before the next contact.
EXQ-085 through EXQ-085d all failed at goal-seeding bottleneck (z_goal_norm < 0.1 in every run).

**Solution:** Drive-scaled benefit signals: when `drive_level` is high (depleted agent), benefit
signal amplitude is multiplied by `drive_scale_factor`, making z_goal seeding reliable during
appetitive approach. `drive_level = 1.0 - agent_energy` is already computable from `obs_body[3]`.
See MECH-112, MECH-113 for the broader homeostatic architecture.

---

## 4. Open Design Gate: Q-020

**CRITICAL — resolve before finalising HippocampalModule.**

Q-020 asks whether rollout proposals from HippocampalModule arrive at E3 pre-weighted by map geometry (MECH-073) or value-neutral (ARC-007 strict).

- If **MECH-073**: HippocampalModule samples proposals with value-correlated frequency before E3 scores them. The E3 input contract includes pre-weighted proposals. The amygdala-analogue write interface (MECH-074) is required before HippocampalModule is finalised.
- If **ARC-007 strict**: HippocampalModule generates value-flat proposals; E3 introduces all weighting. Simpler E3 input contract. MECH-074 may still exist but isn't a prerequisite.

**Working hypothesis (SD-005 dissolution):** Once `z_gamma` is split into `z_self` and `z_world`, the hippocampal map *is* `z_world`, which *is* the residue field's domain. Valence may live in `z_world` structure (residue field curvature), not in a separate hippocampal computation. If so:
- ARC-007 is vindicated: no independent value computation in hippocampus
- MECH-073 is reframed as a consequence of ARC-013 applied to `z_world`
- Q-020 dissolves — it was an artefact of the unsplit `z_gamma`

**Action:** Adjudicate Q-020 theoretically (before implementation) by evaluating the SD-005 dissolution hypothesis. If accepted, proceed with ARC-007-strict HippocampalModule. Test formally with V3-EXQ-006 and V3-EXQ-008 after substrate exists.

---

## 5. V3 Module Architecture

### 5.1 LatentStack changes

```python
@dataclass
class LatentState:
    z_self: torch.Tensor    # [batch, self_dim]  — proprioceptive + interoceptive  (E2 domain)
    z_world: torch.Tensor   # [batch, world_dim] — exteroceptive world model        (E3 domain)
    z_beta: torch.Tensor    # [batch, beta_dim]  — affective latent                 (shared)
    z_theta: torch.Tensor   # [batch, theta_dim] — sequence context                 (shared)
    z_delta: torch.Tensor   # [batch, delta_dim] — regime/motivation                (shared)
    precision: Dict[str, torch.Tensor]
    timestamp: Optional[int] = None
```

`z_gamma` is removed. Encoder is split into `self_encoder` and `world_encoder`. All downstream modules are updated to consume the appropriate stream.

### 5.2 E1 (deep predictor)

Unchanged in function: slow world model, LSTM, trains on sensory prediction error. In V3: produces predictions over both `z_self` and `z_world` channels (associative prior). Provides E1 prior into HippocampalModule (SD-002, already wired in V2).

### 5.3 E2 (fast transition model)

**Expanded interface:**
```python
class E2FastPredictor:
    def forward(self, z_self: Tensor, a: Tensor) -> Tensor:
        """Motor-sensory prediction: z_self_t + a → z_self_{t+1}"""

    def world_forward(self, z_world: Tensor, a: Tensor) -> Tensor:
        """World-state prediction: z_world_t + a → z_world_{t+1}"""
        # Used for SD-003 V3 attribution only

    def action_object(self, z_world: Tensor, a: Tensor) -> Tensor:
        """Produce compressed world-effect action object o_t (SD-004)"""
```

E2 trains on motor-sensory prediction error over `z_self` (primary). `world_forward` and `action_object` may share weights or be lightweight heads.

**Not a harm predictor.** Remove `predict_harm` head from V2 — it belongs to E3.

### 5.4 E3 complex

**New harm evaluation method (SD-003 V3):**
```python
class E3Selector:
    def harm_eval(self, z_world: Tensor) -> Tensor:
        """Evaluate harm of a world-state. Used in SD-003 attribution pipeline."""
```

E3 trains on harm + goal error over `z_world`. E3's harm evaluator is the correct locus for harm prediction — not E2. Precision is E3-derived (from E3 prediction error variance), not hardcoded (required for ARC-016).

**Dynamic precision:**
```python
precision = E3.current_prediction_error_variance.running_mean
# Used to gate commitment threshold (ARC-016)
commit_threshold = precision_to_threshold(precision)
```

### 5.5 HippocampalModule

- Navigates action-object space `O` (from SD-004), not raw `z_world`
- ResidueField operates over `z_world` (from SD-005)
- E1 associative prior wired in (SD-002, V2 resolved — preserve)
- Performs replay during quiescent E3 heartbeat cycles (MECH-092)
- Q-020 adjudication determines whether proposals are value-flat or pre-weighted at E3 input

### 5.6 ResidueField

Operates over `z_world`, not `z_gamma`. Accumulates `world_delta` from SD-003 attribution pipeline. Self-change (`z_self_delta`) does not drive residue accumulation.

### 5.7 Environment: CausalGridWorld V3

Extend to provide explicit self/world observation channels:
```python
# Observation dict (replacing flat vector)
{
    "body_state": [...],       # proprioceptive channels → z_self encoder
    "world_state": [...],      # exteroceptive channels → z_world encoder
    "contamination_view": [...] # 5×5 float grid (world channel)
}
```

Ground truth `transition_type ∈ {agent_caused_hazard, env_caused_hazard, resource, none}` preserved.

---

## 6. Heartbeat Architecture (SD-006 implementation targets)

These claims are V3-scoped. Implement and test in order:

| Claim | What to implement |
|---|---|
| ARC-023 | Three characteristic update rates: `e1_steps_per_tick`, `e2_steps_per_tick`, `e3_steps_per_tick` |
| MECH-089 | `ThetaBuffer` — rolling E1 summary consumed by E3; E3 never sees raw E1 output |
| MECH-090 | `beta_state` flag gates E3 policy propagation to action selection (not E3 internal updating) |
| MECH-091 | `e3_clock.phase_reset()` on salient events; synchronises next E3 tick to event |
| MECH-092 | `hippocampal.replay(theta_buffer.recent)` during quiescent E3 heartbeat cycles |
| MECH-093 | `e3_steps_per_tick` varies with `z_beta` magnitude (arousal → faster E3 rate) |

**Hypothesis tag (MECH-094):** All internally-generated content (replay, DMN/simulation) carries `hypothesis_tag = True`. This categorically blocks the post-commit error channel and prevents residue accumulation from simulated content. Implement as a flag on the LatentState that is checked before any residue write.

---

## 7. V3 Experiment Queue (Historical — Launch Plan)

> **This section is historical.** These were the first 10 experiments planned at V3 launch
> (2026-03-16). All have been run or superseded. The active experiment queue is in
> `ree-v3/experiment_queue.json`; completed runs are in `ree-v3/runner_status.json`.
> Current queue count: EXQ-074b and EXQ-076b (as of 2026-03-26).

These were the first experiments designed after substrate was built:

| ID | Claim | Depends on | Gate |
|---|---|---|---|
| V3-EXQ-001 | SD-005 channel separation validation | SD-005 | First — validates substrate |
| V3-EXQ-002 | Full SD-003 self-attribution (E2+E3 joint) | SD-005, EXQ-018 | SD-005 done |
| V3-EXQ-003 | Action-object planning horizon extension | SD-004 | SD-004 done |
| V3-EXQ-004 | Three-loop incommensurability (ARC-021 full) | SD-005 | V3-EXQ-001 PASS |
| V3-EXQ-005 | World-delta residue accuracy (MECH-072 V3) | SD-005, V3-EXQ-002 | V3-EXQ-002 done |
| V3-EXQ-006 | Intrinsic map valence vs external comparator (Q-020) | SD-005, Q-020 adjudicated | After Q-020 resolved |
| V3-EXQ-007 | Amygdala write operations affect map geometry (MECH-074) | SD-005, Q-020 | After V3-EXQ-006 |
| V3-EXQ-008 | SD-005 dissolves Q-020 (z_world = residue domain) | SD-005 | Pair with V3-EXQ-006 |
| V3-EXQ-009 | Path memory ablation with proper HippocampalModule (ARC-007) | SD-004 | SD-004 done |
| V3-EXQ-010 | Dynamic precision behavioral distinction (ARC-016) | E3-derived precision, wired commit→behavior | ARC-016 circuit complete |

**First priority order: V3-EXQ-001 → V3-EXQ-002 → V3-EXQ-003, V3-EXQ-004 (parallel)**

---

## 8. Governance Integration Requirements

All V3 experiments must produce run packs compatible with REE_assembly governance:

```
REE_assembly/evidence/experiments/claim_probe_{claim_id}/runs/{run_id}_v3/
    manifest.json    # architecture_epoch: "ree_hybrid_guardrails_v1"; run_id ends "_v3"
    metrics.json     # includes fatal_error_count: 0.0
    summary.md
```

Key fields in manifest.json:
```json
{
    "architecture_epoch": "ree_hybrid_guardrails_v1",
    "run_id": "20260401T120000_z_self_world_separation_v3",
    "status": "PASS" | "FAIL"
}
```

`sync_v2_results.py` covers V2. A `sync_v3_results.py` should be written when V3 experiments run (same pattern as sync_v2_results.py but reading from `ree-v3/evidence/experiments/`). Alternatively, V3 experiments can write run packs directly to REE_assembly if the runner is extended.

---

## 9. Repo Structure

```
ree-v3/
├── ree_core/
│   ├── __init__.py
│   ├── agent.py                    # REEAgent — updated for split latent
│   ├── latent/
│   │   ├── stack.py                # LatentState with z_self, z_world
│   │   └── theta_buffer.py         # ThetaBuffer for cross-rate integration (SD-006)
│   ├── predictors/
│   │   ├── e1_deep.py              # Unchanged in function; reads z_self + z_world
│   │   ├── e2_fast.py              # Extended: world_forward + action_object (SD-004/005)
│   │   └── e3_selector.py          # Extended: harm_eval + dynamic precision (ARC-016)
│   ├── hippocampal/
│   │   └── module.py               # Action-object space navigation (SD-004)
│   ├── residue/
│   │   └── field.py                # Operates over z_world (SD-005)
│   ├── heartbeat/
│   │   ├── clock.py                # Multi-rate clock, phase reset (SD-006, ARC-023, MECH-091)
│   │   └── beta_gate.py            # Beta-gated policy propagation (MECH-090)
│   ├── environment/
│   │   └── causal_grid_world.py    # Extended with self/world obs channels
│   └── utils/
│       └── config.py               # Extended with rate params
├── experiments/
│   ├── run.py                      # Experiment runner (inherits V2 pattern)
│   ├── pack_writer.py              # Writes governance run packs
│   ├── v3_exq_001_z_separation.py
│   ├── v3_exq_002_sd003_joint.py
│   └── ...
├── evidence/
│   └── experiments/               # V3 flat JSON results (for sync_v3_results.py)
├── tests/
├── scripts/
│   └── sync_v3_results.py         # Bridges V3 flat JSON → REE_assembly run packs
└── CLAUDE.md                       # Single-branch policy: main; python /opt/local/bin/python3
```

---

## 10. V3 CLAUDE.md Content (repo-level instructions)

```markdown
# ree-v3

## Python
Use /opt/local/bin/python3 for all execution (has torch 2.10.0).
Use sys.executable for subprocesses within experiment runners.

## Branch Policy
No feature branches. All work to `main` directly.
Push: `git push origin HEAD:main`

## Governance
Run packs go to REE_assembly/evidence/experiments/.
run_id must end _v3. architecture_epoch must be "ree_hybrid_guardrails_v1".
After experiments complete: run sync_v3_results.py then build_experiment_indexes.py.

## Key Architecture Constraints
- E2 trains on motor-sensory error (z_self). NOT harm/goal error.
- E3 is the harm evaluator. harm_eval() belongs on E3Selector.
- ResidueField accumulates world_delta (z_world). NOT z_gamma.
- HippocampalModule navigates action-object space O. NOT raw z_world.
- All replay/simulation content must carry hypothesis_tag=True (MECH-094).
- Precision is E3-derived (E3 prediction error variance). NOT hardcoded.
```

---

## 11. Build Order (Historical — COMPLETED 2026-03-16 to 2026-03-18)

> **This section is historical.** The 12-step build order was completed at V3 launch.
> Current implementation status is in §0 (SD table) and `ree-v3/CLAUDE.md`.
> Pending implementation: SD-011 (dual nociceptive streams) and SD-012 (homeostatic drive).

1. **Q-020 adjudication** ✓ — ARC-007 strict decided 2026-03-16
2. **Latent split (SD-005)** ✓ — LatentState z_self/z_world, split encoder
3. **E2 extensions (SD-004)** ✓ — `world_forward` + `action_object` heads
4. **E3 extensions** ✓ — `harm_eval` method, E3-derived dynamic precision
5. **ResidueField update** ✓ — z_world substrate, world_delta accumulation
6. **HippocampalModule update** ✓ — action-object space navigation
7. **CausalGridWorld extension** ✓ — split observation channels
8. **Multi-rate clock (SD-006 phase 1)** ✓ — time-multiplexed, explicit rate params
9. **ThetaBuffer + cross-rate integration** ✓ — E3 consumes theta summaries
10. **Beta gate + phase reset** ✓ — MECH-090, MECH-091
11. **Replay** ✓ — MECH-092
12. **V3-EXQ-001** ✓ — substrate validated; experimentation continues through EXQ-096a+

---

## 12. What V3 Does NOT Need to Implement

- Sleep architecture (docs/architecture/sleep/) — separate system, not V3-scoped
- Full DMN (ARC-014 at minutes-to-hours timescale) — MECH-092 is the micro-DMN; full DMN is later
- Multi-agent / multi-instance scenarios (V4+)
- Production deployment concerns
- z_self hippocampal navigation / ARC-031 (self-navigation) — V4, gated on EXQ-075/076 PASS
- SWR consolidation, slow-wave sleep, theta-gamma replay (V4 scope — see §0 V3/V4 boundary)

---

## Source Documents

All decisions in this spec derive from:

- `REE_assembly/docs/architecture/v2_v3_transition_roadmap.md` — V2 results, V3 targets, transition criteria
- `REE_assembly/docs/architecture/control_plane_heartbeat.md` — ARC-023, MECH-089–093, SD-006
- `REE_assembly/docs/thoughts/2026-03-14_self_world_latent_split_sd003_limitation.md` — SD-005 motivation
- `REE_assembly/docs/architecture/sd_003_experiment_design.md` — SD-003 V2/V3 design, EXQ-027/028 post-mortem
- `REE_assembly/docs/claims/claims.yaml` — claim statuses, v3_pending flags
- `REE_assembly/evidence/experiments/promotion_demotion_recommendations.md` — governance decision queue
