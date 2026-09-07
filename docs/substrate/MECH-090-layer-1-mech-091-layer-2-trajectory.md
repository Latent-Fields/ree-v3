## MECH-090 Layer 1 + MECH-091 Layer 2: Trajectory Stepping + Urgency Interrupt (2026-04-15)
- MECH-090 Layer 1: control_plane.committed_trajectory_stepping -- IMPLEMENTED 2026-04-15.
  Module: ree_core/agent.py (REEAgent.select_action, REEAgent.reset).
  Config: E3Config.urgency_interrupt_threshold (float, default 0.8).
  Previously: select_action() always used committed_trajectory.actions[:, 0, :] (first
  action repeated every E3 tick during commitment). Now: _committed_step_idx counter steps
  through actions[:, idx, :] on each E3 tick. Counter clamped to (horizon-1) to guard
  against overflow. Reset on beta_gate.release() and agent.reset().
  Data flow: commit -> _committed_step_idx=0; each E3 tick in committed state ->
  action = committed_trajectory.actions[:, _committed_step_idx, :]; idx += 1 (clamped).
  Biological basis: committed motor sequences are unrolled action-by-action, not
  repeated. Striatal-thalamo-cortical propagation advances through the planned motor
  program at each execution step.
  MECH-094: not applicable (waking action selection, not simulation content).
  See MECH-090, ARC-028, MECH-091.

- MECH-091 Layer 2: control_plane.urgency_interrupt -- IMPLEMENTED 2026-04-15.
  Module: ree_core/agent.py (REEAgent.select_action).
  Config: E3Config.urgency_interrupt_threshold (float, default 0.8).
  When beta is elevated (committed state) and z_harm_a.norm() > urgency_interrupt_threshold:
  beta_gate.release() is called and _committed_step_idx is reset to 0, falling through to
  fresh E3 selection on the same tick.
  Data flow: select_action() -> [beta elevated?] -> z_harm_a.norm() -> [> threshold?] ->
  beta_gate.release(); _committed_step_idx=0 -> E3.select() with fresh state.
  Biological basis: unexpected nociceptive escalation (C-fiber burst) triggers STN -> GPe
  urgency signal, interrupting the committed motor program and returning to deliberative
  planning. Links SD-021 descending modulation (z_harm_s attenuated during commitment) with
  the escape mechanism: z_harm_a (affective load, not gated) drives the interrupt.
  Backward compatible: urgency_interrupt_threshold=0.8 (default); only fires when beta is
  elevated AND z_harm_a.norm() exceeds threshold. Existing experiments unaffected (most
  use default E3Config with urgency_weight=0.0 so z_harm_a.norm() stays low).
  MECH-094: not applicable (waking action selection gate).
  See MECH-091, MECH-090, SD-021, SD-011.
