"""
MultiContentThetaPacket -- joint multi-content theta-burst packet (MECH-294)

Sibling module to MECH-089 ThetaBuffer. ThetaBuffer is single-content temporal
averaging (it buffers z_world / z_self and returns a theta-cycle-averaged
z_world). MECH-294 is the JOINT-binding claim: each ~125 ms theta cycle binds a
{goal_latent, action_proposal, risk_estimate, state_summary} tuple into ONE
phase-aligned packet, so a downstream reader can condition action on the goal and
risk co-bound in the SAME cycle.

This module COMPOSES the existing ThetaBuffer (it does not replace it): the
`state_summary` slot is filled by the caller from `theta_buffer.summary()` at
seal time. MECH-089 stays exactly as it is; MECH-294 is a strict additive layer.

Design memo (the contract):
  REE_assembly/docs/architecture/mech_294_multi_content_theta_packet.md

Load-bearing properties:
  - JOINT binding WITHIN one cycle (not one stream alternating across cycles).
    The substrate exposes three operating regimes -- "joint", "alternation"
    (Kay 2020 cross-cycle parsimonious control), and "shuffled" (independent-
    content control) -- so a discriminative experiment can hold bandwidth and
    content identical and vary ONLY the binding structure.
  - Per-stream verisimilitude vintaging (MECH-269 / MECH-269b reuse): each
    component carries its own temporal vintage. A stream with high per-stream
    V_s contributes its current value; a stream with low V_s contributes its
    last-verified snapshot. The packet is a stream-typed object whose components
    may have different ages.

Pure-arithmetic: no nn.Module, no learned parameters, no gradient flow. Sibling
pattern to the regulator modules (MECH-313 / MECH-320 / MECH-342). All tensors
are detached on observe (a packet is a read-only snapshot, never a gradient
path).

MECH-094: every state-advancing method takes simulation_mode and is a no-op when
True -- a replay / DMN tick must not seal a waking packet.
"""

from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Dict, List, Deque

import torch


# Stream identity keys. Ordering is fixed and is the round-robin order used by
# the "alternation" binding mode (the Kay-2020 one-stream-per-cycle control).
STREAM_GOAL = "goal_latent"
STREAM_RISK_SENSORY = "risk_sensory"
STREAM_RISK_AFFECTIVE = "risk_affective"
STREAM_STATE = "state_summary"

# V_s-gated streams (carry a per_stream_vs entry). action_proposal is a control
# stream with no V_s; its vintage is an E3-tick age, not a V_s value.
VS_STREAM_KEYS = {
    STREAM_GOAL: "z_goal",
    STREAM_RISK_SENSORY: "z_harm_s",
    STREAM_RISK_AFFECTIVE: "z_harm_a",
    STREAM_STATE: "z_world",
}

# Round-robin order for the alternation control (one live stream per cycle).
ALTERNATION_ORDER = [
    STREAM_GOAL,
    STREAM_RISK_SENSORY,
    STREAM_RISK_AFFECTIVE,
    STREAM_STATE,
]

VALID_BINDING_MODES = ("joint", "alternation", "shuffled")


@dataclass
class ThetaPacketVintage:
    """Per-component temporal-vintage metadata.

    is_current: True if the slot took the stream's current value this cycle;
                False if it substituted a held (last-verified) snapshot.
    age_ticks:  cycles since the value the slot carries was current (0 == fresh).
    v_s:        the per-stream V_s reading that drove the snapshot-or-hold
                decision, or None for streams with no V_s (action_proposal).
    """
    is_current: bool = True
    age_ticks: int = 0
    v_s: Optional[float] = None


@dataclass
class ThetaPacket:
    """One sealed, immutable theta-cycle packet.

    Type-separated sub-slots (NOT a flat concat) so the joint-read interface can
    condition action on goal-and-risk; a flat concat would lose the typing the
    claim needs and silently collapse the SD-011 dual-stream distinction.
    """
    cycle_index: int
    binding_mode: str
    goal_latent: Optional[torch.Tensor] = None        # [1, goal_dim]
    action_proposal: Optional[torch.Tensor] = None     # [1, action_object_dim]
    risk_sensory: Optional[torch.Tensor] = None        # [1, harm_dim]  (z_harm_s)
    risk_affective: Optional[torch.Tensor] = None      # [1, harm_dim]  (z_harm_a)
    state_summary: Optional[torch.Tensor] = None       # [1, world_dim] (MECH-089 averaged z_world)
    vintage: Dict[str, ThetaPacketVintage] = field(default_factory=dict)
    action_proposal_age: int = 0

    # -- completeness / vintage diagnostics --------------------------------

    def is_complete(self) -> bool:
        """True when all four content streams are populated (current or held)."""
        return (
            self.goal_latent is not None
            and self.risk_sensory is not None
            and self.risk_affective is not None
            and self.state_summary is not None
        )

    def n_distinct_vintages(self) -> int:
        """Number of distinct (is_current, age_ticks) signatures across the
        V_s-gated components -- the stream-typed-object heterogeneity check
        (G1). A homogeneous current-latent packet returns 1."""
        sigs = set()
        for name in VS_STREAM_KEYS:
            v = self.vintage.get(name)
            if v is not None:
                sigs.add((v.is_current, v.age_ticks))
        return len(sigs)

    def is_component_stale(self, name: str) -> bool:
        v = self.vintage.get(name)
        return bool(v is not None and not v.is_current)

    # -- joint-read interface (S5) -----------------------------------------

    def _typed_components(self) -> List[torch.Tensor]:
        """Type-ordered list of populated component tensors, each prefixed with
        a fixed (non-learned) scalar type tag so a reader can tell streams
        apart. The ONE place a flat vector is produced. Stale components
        substitute their held snapshot (already done at seal time)."""
        out: List[torch.Tensor] = []
        ordered = [
            (0.0, self.goal_latent),
            (1.0, self.action_proposal),
            (2.0, self.risk_sensory),
            (3.0, self.risk_affective),
            (4.0, self.state_summary),
        ]
        for tag, comp in ordered:
            if comp is None:
                continue
            c = comp.reshape(1, -1)
            tagcol = torch.full((1, 1), tag, dtype=c.dtype, device=c.device)
            out.append(torch.cat([tagcol, c], dim=-1))
        return out

    def joint_context(self) -> Optional[torch.Tensor]:
        """Type-tagged concat of all populated components (for a consumer that
        wants a single context tensor, e.g. a bias head). None if the packet is
        empty. Components are flattened to a common dim before concat via the
        type-tag prefix; the reader gets a [1, sum_of_tagged_dims] vector."""
        comps = self._typed_components()
        if not comps:
            return None
        return torch.cat(comps, dim=-1)

    def action_conditioned_on(self, goal: bool = True, risk: bool = True) -> Optional[torch.Tensor]:
        """Return the action_proposal slot annotated with the co-bound goal and
        risk -- the literal "which action is on the table, read against the goal
        and risk co-bound this cycle" operation. This is the joint read the
        proposer / E3 uses. Returns None when no action proposal is bound."""
        if self.action_proposal is None:
            return None
        parts = [self.action_proposal.reshape(1, -1)]
        if goal and self.goal_latent is not None:
            parts.append(self.goal_latent.reshape(1, -1))
        if risk:
            if self.risk_sensory is not None:
                parts.append(self.risk_sensory.reshape(1, -1))
            if self.risk_affective is not None:
                parts.append(self.risk_affective.reshape(1, -1))
        return torch.cat(parts, dim=-1)

    def risk_vector(self) -> Optional[torch.Tensor]:
        """Combined risk read over the two SD-011 sub-slots (concat preserves
        the dual-stream distinction; a reducing consumer may collapse). None
        when neither risk sub-slot is bound."""
        parts = []
        if self.risk_sensory is not None:
            parts.append(self.risk_sensory.reshape(1, -1))
        if self.risk_affective is not None:
            parts.append(self.risk_affective.reshape(1, -1))
        if not parts:
            return None
        return torch.cat(parts, dim=-1)


@dataclass
class MultiContentThetaPacketConfig:
    binding_mode: str = "joint"               # "joint" / "alternation" / "shuffled"
    snapshot_refresh_threshold: float = 0.5    # MECH-269b reuse
    hold_threshold: float = 0.4                # MECH-269b reuse (0.4-0.5 dead-band)
    history_length: int = 8                    # per-stream history depth for "shuffled"
    unknown_stream_passes: bool = True         # a stream with no V_s entry passes (current)


class MultiContentThetaPacket:
    """Per-cycle joint multi-content theta packet.

    Lifecycle per theta cycle (the interval between two E3 heartbeat ticks):
      1. open  -- a binding window opens at the E3-heartbeat boundary (implicit;
                  the window is the accumulation state between two seal() calls).
      2. observe(...) -- each E1 tick pushes the current per-stream latents into
                  the open window (latest-value-wins within the window).
      3. observe_action_proposal(...) -- the proposer pushes the current best /
                  committed first-action when it refits (at the E3 tick).
      4. seal(state_summary) -- at the next E3-heartbeat boundary, produce one
                  immutable ThetaPacket: apply per-stream V_s snapshot-or-hold
                  vintaging, fill state_summary from the MECH-089 ThetaBuffer
                  average, apply the binding-mode transform, and open a fresh
                  window.

    The sealed packet is exposed by the agent as `agent.last_theta_packet` and is
    read by the proposer / E3 on the NEXT cycle as a joint object.
    """

    def __init__(self, config: Optional[MultiContentThetaPacketConfig] = None):
        self.config = config or MultiContentThetaPacketConfig()
        if self.config.binding_mode not in VALID_BINDING_MODES:
            raise ValueError(
                "binding_mode must be one of %s, got %r"
                % (VALID_BINDING_MODES, self.config.binding_mode)
            )

        # Open-window accumulators (latest value wins within the cycle).
        self._win_goal: Optional[torch.Tensor] = None
        self._win_risk_s: Optional[torch.Tensor] = None
        self._win_risk_a: Optional[torch.Tensor] = None
        self._win_vs: Dict[str, float] = {}
        self._win_action: Optional[torch.Tensor] = None
        self._action_last_refit_cycle: int = -1

        # Held snapshots (last-verified value) for V_s-hold + alternation.
        self._snapshots: Dict[str, torch.Tensor] = {}
        self._snapshot_age: Dict[str, int] = {}

        # Per-stream history of sealed current-values for the "shuffled" control.
        hl = max(2, int(self.config.history_length))
        self._history: Dict[str, Deque[torch.Tensor]] = {
            k: deque(maxlen=hl) for k in VS_STREAM_KEYS
        }

        self._cycle_index: int = 0

        # Diagnostics (read by the validation experiment / manifests).
        self.n_seals: int = 0
        self.n_observes: int = 0
        self.n_action_observes: int = 0
        self.n_simulation_skipped: int = 0
        self.n_held_substitutions: int = 0
        self.last_packet: Optional[ThetaPacket] = None

    # -- window accumulation ----------------------------------------------

    def observe(
        self,
        z_goal: Optional[torch.Tensor],
        z_harm_s: Optional[torch.Tensor],
        z_harm_a: Optional[torch.Tensor],
        per_stream_vs: Optional[Dict[str, float]] = None,
        simulation_mode: bool = False,
    ) -> None:
        """Push the current per-stream latents into the open binding window.
        Called once per E1 tick. Latest value within the cycle wins. Detaches
        every tensor (the packet is a read-only snapshot, never a grad path)."""
        if simulation_mode:
            self.n_simulation_skipped += 1
            return
        if z_goal is not None:
            self._win_goal = z_goal.detach().reshape(1, -1).clone()
        if z_harm_s is not None:
            self._win_risk_s = z_harm_s.detach().reshape(1, -1).clone()
        if z_harm_a is not None:
            self._win_risk_a = z_harm_a.detach().reshape(1, -1).clone()
        if per_stream_vs:
            # Copy only the scalar V_s readings we care about.
            for name, vskey in VS_STREAM_KEYS.items():
                if vskey in per_stream_vs:
                    self._win_vs[name] = float(per_stream_vs[vskey])
        self.n_observes += 1

    def observe_action_proposal(
        self,
        action_obj: Optional[torch.Tensor],
        simulation_mode: bool = False,
    ) -> None:
        """Push the current best / committed first-action object into the open
        window. Called when the CEM proposer refits (at the E3 tick)."""
        if simulation_mode:
            self.n_simulation_skipped += 1
            return
        if action_obj is not None:
            self._win_action = action_obj.detach().reshape(1, -1).clone()
            self._action_last_refit_cycle = self._cycle_index
            self.n_action_observes += 1

    # -- vintaging ---------------------------------------------------------

    def _vintaged_value(self, name: str, current: Optional[torch.Tensor]):
        """Apply the MECH-269b snapshot-or-hold discipline for one V_s-gated
        stream. Returns (value, ThetaPacketVintage). Refresh the snapshot when
        V_s >= refresh threshold; hold (substitute snapshot) when V_s < hold
        threshold; the 0.4-0.5 dead-band keeps the prior decision (here: take
        current and refresh, since we have a fresh observation)."""
        vs = self._win_vs.get(name, None)
        refresh = self.config.snapshot_refresh_threshold
        hold = self.config.hold_threshold

        # No V_s reading available -> pass current (or held if no current).
        if vs is None:
            if current is not None:
                self._snapshots[name] = current.detach().clone()
                self._snapshot_age[name] = 0
                return current, ThetaPacketVintage(is_current=True, age_ticks=0, v_s=None)
            held = self._snapshots.get(name)
            if held is not None:
                age = self._snapshot_age.get(name, 0)
                return held, ThetaPacketVintage(is_current=False, age_ticks=age, v_s=None)
            return None, ThetaPacketVintage(is_current=True, age_ticks=0, v_s=None)

        if vs < hold and (name in self._snapshots):
            # Hold: substitute the last-verified snapshot, mark stale + aged.
            self._snapshot_age[name] = self._snapshot_age.get(name, 0) + 1
            self.n_held_substitutions += 1
            held = self._snapshots[name]
            return held, ThetaPacketVintage(
                is_current=False, age_ticks=self._snapshot_age[name], v_s=vs
            )

        # Refresh-or-deadband: take current and refresh the snapshot.
        if current is not None:
            self._snapshots[name] = current.detach().clone()
            self._snapshot_age[name] = 0
            return current, ThetaPacketVintage(is_current=True, age_ticks=0, v_s=vs)

        # No current value this window but V_s allows current -> fall back to held.
        held = self._snapshots.get(name)
        if held is not None:
            age = self._snapshot_age.get(name, 0)
            return held, ThetaPacketVintage(is_current=False, age_ticks=age, v_s=vs)
        return None, ThetaPacketVintage(is_current=True, age_ticks=0, v_s=vs)

    # -- binding-mode transforms ------------------------------------------

    def _history_other_cycle(self, name: str, fallback: Optional[torch.Tensor]):
        """For the shuffled control: return a value for `name` drawn from a
        DIFFERENT (earlier) cycle than the others, so the four components are
        real and current-looking but were never co-observed. Deterministic
        per-stream offset into the history ring keeps it seed-stable."""
        hist = self._history.get(name)
        if not hist or len(hist) < 2:
            return fallback
        # Distinct offset per stream so no two slots draw the same cycle.
        offset = 1 + (ALTERNATION_ORDER.index(name) % (len(hist) - 1))
        return hist[-1 - offset] if (len(hist) - 1 - offset) >= 0 else hist[0]

    # -- seal --------------------------------------------------------------

    def seal(
        self,
        state_summary: Optional[torch.Tensor],
        simulation_mode: bool = False,
    ) -> Optional[ThetaPacket]:
        """Seal the open window into one immutable ThetaPacket and open a fresh
        window. `state_summary` is the MECH-089 ThetaBuffer average (caller
        passes theta_buffer.summary()). MECH-094: no-op under simulation_mode."""
        if simulation_mode:
            self.n_simulation_skipped += 1
            return None

        # Stage the window's current values for the V_s-gated streams.
        state_current = (
            state_summary.detach().reshape(1, -1).clone()
            if state_summary is not None else None
        )
        staged = {
            STREAM_GOAL: self._win_goal,
            STREAM_RISK_SENSORY: self._win_risk_s,
            STREAM_RISK_AFFECTIVE: self._win_risk_a,
            STREAM_STATE: state_current,
        }

        # Capture the prior (end-of-previous-cycle) snapshots BEFORE this cycle's
        # refresh. Alternation holds non-live streams at these prior values; if
        # we let _vintaged_value refresh every stream first (as joint/shuffled
        # need), the "held" value would equal this cycle's current and the
        # alternation control would be vacuous.
        prior_snapshots: Dict[str, Optional[torch.Tensor]] = {
            name: (self._snapshots[name].clone() if name in self._snapshots else None)
            for name in ALTERNATION_ORDER
        }

        vintaged: Dict[str, Optional[torch.Tensor]] = {}
        vintage_meta: Dict[str, ThetaPacketVintage] = {}
        mode = self.config.binding_mode

        if mode == "alternation":
            # Exactly ONE stream live this cycle (round-robin); the other three
            # held at their PRIOR snapshots (the value from the last cycle each
            # was live). Same four slots, same bandwidth across four cycles, but
            # no within-cycle co-binding. Only the live stream refreshes.
            live = ALTERNATION_ORDER[self._cycle_index % len(ALTERNATION_ORDER)]
            for name in ALTERNATION_ORDER:
                if name == live:
                    val, meta = self._vintaged_value(name, staged[name])
                    vintaged[name] = val
                    vintage_meta[name] = meta
                    if val is not None and meta.is_current:
                        self._history[name].append(val.detach().clone())
                else:
                    held = prior_snapshots[name]
                    if held is not None:
                        self._snapshot_age[name] = self._snapshot_age.get(name, 0) + 1
                        vintaged[name] = held
                        vintage_meta[name] = ThetaPacketVintage(
                            is_current=False,
                            age_ticks=self._snapshot_age[name],
                            v_s=self._win_vs.get(name),
                        )
                    else:
                        # Cold start (no prior snapshot yet): refresh from current.
                        val, meta = self._vintaged_value(name, staged[name])
                        vintaged[name] = val
                        vintage_meta[name] = meta
            bound = dict(vintaged)

        else:
            # joint + shuffled: refresh every stream (V_s snapshot-or-hold) and
            # push the current values into per-stream history so "shuffled" can
            # draw genuine prior cycles.
            for name in ALTERNATION_ORDER:
                val, meta = self._vintaged_value(name, staged[name])
                vintaged[name] = val
                vintage_meta[name] = meta
                if val is not None and meta.is_current:
                    self._history[name].append(val.detach().clone())

            if mode == "joint":
                bound = dict(vintaged)
            else:  # "shuffled" -- independent-content control
                # Each slot filled from a DIFFERENT prior cycle's value for that
                # stream (matched marginals, never co-observed).
                bound = {}
                for name in ALTERNATION_ORDER:
                    bound[name] = self._history_other_cycle(name, vintaged[name])
                    vintage_meta[name] = ThetaPacketVintage(
                        is_current=False,
                        age_ticks=max(1, vintage_meta[name].age_ticks),
                        v_s=vintage_meta[name].v_s,
                    )

        # action_proposal vintage: age in E3-heartbeat ticks (no V_s).
        action_age = (
            self._cycle_index - self._action_last_refit_cycle
            if self._action_last_refit_cycle >= 0 else 0
        )
        vintage_meta["action_proposal"] = ThetaPacketVintage(
            is_current=(action_age == 0), age_ticks=max(0, action_age), v_s=None,
        )

        packet = ThetaPacket(
            cycle_index=self._cycle_index,
            binding_mode=mode,
            goal_latent=bound.get(STREAM_GOAL),
            action_proposal=self._win_action,
            risk_sensory=bound.get(STREAM_RISK_SENSORY),
            risk_affective=bound.get(STREAM_RISK_AFFECTIVE),
            state_summary=bound.get(STREAM_STATE),
            vintage=vintage_meta,
            action_proposal_age=max(0, action_age),
        )

        # Open a fresh window: clear per-cycle accumulators (snapshots and
        # history persist across cycles by design).
        self._win_goal = None
        self._win_risk_s = None
        self._win_risk_a = None
        self._win_vs = {}
        # action proposal persists until the proposer refits (its age tracks
        # staleness); do NOT clear _win_action here.

        self._cycle_index += 1
        self.n_seals += 1
        self.last_packet = packet
        return packet

    # -- joint-read pass-through (optional E3-bias compose path, parameter-free)

    def compose_e3_bias(
        self,
        candidate_first_actions: torch.Tensor,
        bias_scale: float = 0.1,
    ) -> Optional[torch.Tensor]:
        """OPTIONAL, read-only-first-by-default consumer hook (S5). Produces a
        per-candidate E3 score-bias [K] from the SEALED packet WITHOUT any
        trained head (pure arithmetic, so no phased training): bias favours
        candidates whose first action aligns with the action co-bound in the
        packet against goal+risk. Clamped to [-bias_scale, +bias_scale]; never
        dominates. Returns None when no action is bound or the packet is empty.

        This path is gated behind theta_packet_compose_into_e3_bias (default
        False) at the call site, so the substrate-readiness validation measures
        the packet read-only before any behavioural-authority experiment depends
        on it.
        """
        if self.last_packet is None or self.last_packet.action_proposal is None:
            return None
        ref = self.last_packet.action_proposal.reshape(1, -1)
        cand = candidate_first_actions.reshape(candidate_first_actions.shape[0], -1)
        d = min(ref.shape[-1], cand.shape[-1])
        if d == 0:
            return None
        ref_n = torch.nn.functional.normalize(ref[:, :d], dim=-1)
        cand_n = torch.nn.functional.normalize(cand[:, :d], dim=-1)
        align = (cand_n * ref_n).sum(dim=-1)  # [K] cosine in [-1, 1]
        # REE lower-is-better: favour aligned candidates -> negative bias.
        bias = -bias_scale * align
        return bias.clamp(-bias_scale, bias_scale)

    # -- diagnostics / reset ----------------------------------------------

    def get_diagnostics(self) -> Dict[str, float]:
        last = self.last_packet
        return {
            "mech294_n_seals": float(self.n_seals),
            "mech294_n_observes": float(self.n_observes),
            "mech294_n_action_observes": float(self.n_action_observes),
            "mech294_n_held_substitutions": float(self.n_held_substitutions),
            "mech294_n_simulation_skipped": float(self.n_simulation_skipped),
            "mech294_last_complete": float(last.is_complete()) if last else 0.0,
            "mech294_last_n_distinct_vintages": float(last.n_distinct_vintages()) if last else 0.0,
            "mech294_cycle_index": float(self._cycle_index),
        }

    def reset(self) -> None:
        """Per-episode reset. Clears window, snapshots, history, counters."""
        self._win_goal = None
        self._win_risk_s = None
        self._win_risk_a = None
        self._win_vs = {}
        self._win_action = None
        self._action_last_refit_cycle = -1
        self._snapshots.clear()
        self._snapshot_age.clear()
        for k in self._history:
            self._history[k].clear()
        self._cycle_index = 0
        self.n_seals = 0
        self.n_observes = 0
        self.n_action_observes = 0
        self.n_simulation_skipped = 0
        self.n_held_substitutions = 0
        self.last_packet = None
