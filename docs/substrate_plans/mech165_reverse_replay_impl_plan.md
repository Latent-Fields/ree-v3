# MECH-165 Reverse Replay Diversity Scheduler
**Gap ID:** MECH-165-reverse
**Complexity:** medium (1-2 sessions)
**Readiness:** ready_to_plan
**Blocking claims:** MECH-165, MECH-092 (indirect)

---

## Problem

`HippocampalModule.replay()` is forward-only with random action sequences (no stored
trajectory replay, no reverse option). EXQ-244 (x2) was non_contributory because:
1. No actual offline phase (proxy substrate only)
2. No exploration-generated source material for diversity
3. No reverse replay implementation

The MECH-165 claim: offline replay must sample trajectory-diverse content (including
non-dominant and counterfactual paths) to prevent Hebbian monopoly. The diversity
mechanism has two components:
1. **Reverse replay** (Diba & Buzsaki 2007): replay stored trajectories in reverse
   temporal order, providing non-forward causal structure
2. **Exploration source material**: diverse waking trajectories (random walks, epsilon-
   greedy) as replay candidates -- without these, forward-only replay just reinforces
   the dominant path

User confirmed (governance-2026-04-08): MECH-165 pulled into V3 scope.
Prerequisite: MECH-120 SHY wiring (see mech120_shy_decay_impl_plan.md) should be
implemented first -- SHY flattening creates the attractor slots that diverse replay
then repopulates.

MECH-165 claim: `REE_assembly/docs/claims/claims.yaml`
Depends on: MECH-120, MECH-121, MECH-092, ARC-007

---

## What Needs to Exist

### 1. Exploration trajectory buffer in HippocampalModule

File: `ree_core/hippocampal/module.py`

Add to `HippocampalModule.__init__()`:
```python
# MECH-165: exploration source buffer for diverse replay
self._exploration_buffer: list = []  # list of Trajectory objects
self._exploration_buffer_maxlen: int = getattr(config, "exploration_buffer_len", 50)
```

Add method `record_exploration_trajectory(trajectory: Trajectory) -> None`:
```python
def record_exploration_trajectory(self, trajectory: "Trajectory") -> None:
    """Record a waking exploration trajectory for use as replay source material."""
    self._exploration_buffer.append(trajectory)
    if len(self._exploration_buffer) > self._exploration_buffer_maxlen:
        self._exploration_buffer.pop(0)  # FIFO eviction
```

### 2. Reverse replay method

File: `ree_core/hippocampal/module.py`

Add `reverse_replay(trajectory: Trajectory) -> Trajectory`:
```python
def reverse_replay(self, trajectory: "Trajectory") -> "Trajectory":
    """
    MECH-165: replay stored trajectory in reverse temporal order.
    Reverses the world_states sequence; no new E2 rollout.
    Carries hypothesis_tag=True (MECH-094).
    """
    world_states = trajectory.get_world_state_sequence()   # [T, world_dim]
    actions = trajectory.get_action_object_sequence()       # [T]

    # Reverse temporal order
    reversed_states = world_states.flip(0)   # [T, world_dim]
    reversed_actions = list(reversed(actions))

    # Construct reversed Trajectory (re-use same Trajectory dataclass)
    from ree_core.predictors.e2_fast import Trajectory as TrajClass
    return TrajClass(
        world_states=reversed_states,
        action_objects=reversed_actions,
        hypothesis_tag=True,
        is_reverse=True,  # tag for logging; add this field to Trajectory if absent
    )
```

If `Trajectory` doesn't have `is_reverse` field, add it with default `False`.

### 3. Diverse replay scheduler

File: `ree_core/hippocampal/module.py`

Add `diverse_replay(recent, drive_state, mode: str = "forward") -> Trajectory`:
```python
def diverse_replay(
    self,
    recent: torch.Tensor,     # theta_buffer_recent [T, batch, world_dim]
    drive_state: Optional[torch.Tensor] = None,
    mode: str = "auto",       # "forward", "reverse", "random", "auto"
) -> Optional["Trajectory"]:
    """
    MECH-165: diversity-scheduled replay.
    Modes:
      "forward"  -- existing replay() behavior
      "reverse"  -- pick stored traj from exploration_buffer, replay in reverse
      "random"   -- existing replay() (random action rollout from recent z_world)
      "auto"     -- sample mode probabilistically per config fractions
    """
    if mode == "auto":
        r = random.random()
        if r < self._reverse_fraction and len(self._exploration_buffer) > 0:
            mode = "reverse"
        elif r < self._reverse_fraction + self._random_fraction:
            mode = "random"
        else:
            mode = "forward"

    if mode == "reverse" and len(self._exploration_buffer) > 0:
        source = random.choice(self._exploration_buffer)
        return self.reverse_replay(source)
    else:
        # forward or random: use existing replay() logic
        return self.replay(recent, drive_state=drive_state)
```

Add fraction config reads in `__init__()`:
```python
self._reverse_fraction: float = getattr(config, "reverse_replay_fraction", 0.3)
self._random_fraction: float = getattr(config, "random_replay_fraction", 0.2)
```

### 4. Agent: call diverse_replay() and record exploration trajectories

File: `ree_core/agent.py`

In `_do_replay()` (or wherever `hippocampal.replay()` is called during SWS):
```python
if getattr(self.config, "replay_diversity_enabled", False):
    trajectory = self.hippocampal.diverse_replay(
        recent, drive_state=drive_state, mode="auto"
    )
else:
    trajectory = self.hippocampal.replay(recent, drive_state=drive_state)
```

In `select_action()` or wherever exploration steps occur, when in exploration mode
(epsilon-greedy or explicit random walk):
```python
if self._is_exploration_step and self.config.replay_diversity_enabled:
    # Record completed waking trajectory to exploration buffer
    if self._current_episode_trajectory is not None:
        self.hippocampal.record_exploration_trajectory(
            self._current_episode_trajectory
        )
```

Episode trajectory tracking (`_current_episode_trajectory`) requires building a
Trajectory object over the waking episode. Add to agent:
```python
self._current_episode_trajectory: Optional[Trajectory] = None
self._episode_world_states: list[torch.Tensor] = []
```

At end of episode (done=True), construct trajectory from `_episode_world_states`.

---

## Inputs and Outputs

| Signal | Direction | Shape | Notes |
|--------|-----------|-------|-------|
| `recent` (theta_buffer) | in | [T, batch, world_dim] | existing |
| `drive_state` | in | [4] | existing (MECH-203) |
| `exploration_buffer` | state | list[Trajectory] | NEW; max 50 trajs |
| replayed Trajectory | out | Trajectory | same type as existing replay |
| `is_reverse` flag | out | bool on Trajectory | NEW field |

---

## Training Signal

Reverse-replayed trajectories feed back into E1 training via the same pathway as
forward replay (MECH-121). No separate loss for reverse vs. forward. The diversity
effect works through the E1 context_memory: reverse-order state sequences promote
different slot activations than forward, reducing dominant-trajectory monopoly.

---

## Config Knobs

All in `REEConfig` (ree_core/utils/config.py):

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `replay_diversity_enabled` | bool | False | Master switch |
| `reverse_replay_fraction` | float | 0.3 | Fraction of replay calls using reverse |
| `random_replay_fraction` | float | 0.2 | Fraction using random rollout |
| `exploration_buffer_len` | int | 50 | Max stored exploration trajectories |

---

## Smoke Test

1. Create agent with `replay_diversity_enabled=True`, `reverse_replay_fraction=0.3`.
2. Run 100 waking steps with epsilon-greedy=0.3 to populate exploration buffer.
3. Assert `len(hippocampal._exploration_buffer) > 0`.
4. Trigger 10 replay calls via `diverse_replay(recent, mode="auto")`.
5. Assert reverse calls occurred at approximately `0.3 * 10 = 3` times (+/-2).
6. Assert each reverse-replayed trajectory has `is_reverse=True`.

---

## Dependent Experiments to Re-queue

| EXQ | Claim | Action |
|-----|-------|--------|
| EXQ-244 x2 | MECH-165 | Queue EXQ-244a: balanced vs forward-only replay WITH exploration buffer |

EXQ-244a design requirements:
- Condition A: `replay_diversity_enabled=False` (forward-only baseline)
- Condition B: `replay_diversity_enabled=True`, `reverse_fraction=0.3`
- Both conditions: 500-step exploration phase (epsilon=0.3) before SWS replay
- 3+ seeds
- Primary metric: behavioral entropy (action distribution) after replay vs. before

Prerequisite: EXQ-245a (MECH-120 SHY wiring) should pass first -- SHY flattening
is required for diverse replay to actually repopulate varied slots.
