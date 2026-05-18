"""
VsRolloutGate -- MECH-269b: Symmetric V_s gating on E1/E2 cortical rollouts.

Per-stream snapshot store + per-stream per-side threshold table + held-snapshot
substitution at the E1 sensory predictor and E2 forward model call sites.

The gate is a CONSUMER of MECH-269 Phase 1 per-stream V_s (computed by
HippocampalModule.update_per_stream_vs). It is NOT a producer of V_s. When V_s for a
stream falls below the side-specific threshold, the gate returns the last-trusted
snapshot of that stream (held value) instead of the current latent value, so cortical
forward predictors do not roll forward off stale-but-confident-looking inputs.

Default no-op: with V_s seeded at 1.0 and aligned latents the gate fires zero times.
Held-snapshot substitutions appear only when V_s drops under the failure-mode
conditions MECH-287 broadcasts respond to.

See REE_assembly/docs/architecture/mech_269b_vs_rollout_gating.md.
See MECH-269 (parent), MECH-269b (THIS), MECH-284 (online staleness arm), MECH-094
(call-site scoping for waking-vs-replay).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class VsRolloutGateConfig:
    """Configuration for the MECH-269b V_s rollout gate.

    Disabled by default for full backward compatibility. Set
    use_vs_rollout_gating=True on REEConfig.from_dims to enable. Requires
    use_per_stream_vs=True (the gate has nothing to read otherwise).

    snapshot_refresh_threshold sits ABOVE max(e1_threshold, e2_threshold) by design
    so a stream straddling the gate threshold oscillates within a refresh-vs-hold
    dead band (lightweight Schmitt-trigger-style hysteresis without a streak
    counter).

    use_staleness_lookup wires the MECH-284 region staleness accumulator into
    the gate's threshold comparison. When True, callers supply a per-stream
    staleness dict (aggregated by HippocampalModule from
    staleness_accumulator + active anchors) and the gate computes
        effective_vs = raw_vs - per_stream_staleness[s]
    before the threshold check. This is the Q-040b strong-reading wiring:
    stale streams fall below threshold even when raw V_s remains high
    (the realistic regime), so the hold path becomes exercisable without
    smoke-level threshold overrides. Default False preserves the legacy
    raw-V_s-only path used by EXQ-490 / EXQ-490b / EXQ-490c.
    """

    streams: Tuple[str, ...] = (
        "z_world", "z_self", "z_harm_s", "z_harm_a", "z_goal", "z_beta",
    )
    snapshot_refresh_threshold: float = 0.5
    e1_threshold: float = 0.4
    e2_threshold: float = 0.4
    e1_threshold_per_stream: Dict[str, float] = field(default_factory=dict)
    e2_threshold_per_stream: Dict[str, float] = field(default_factory=dict)
    # When True (default): if no snapshot has been recorded yet for a stream
    # whose V_s falls below threshold, pass the current value through unchanged
    # rather than substituting None / zeros. Prevents agent loop from ever
    # consuming a None latent.
    unknown_stream_passes: bool = True
    # MECH-284 staleness wiring (Q-040b strong reading). When True, the gate
    # reads per_stream_staleness from gate() / gate_stream() callers and
    # subtracts it from raw V_s before comparing to the threshold. Backward
    # compatible: when False (default) or when caller supplies no staleness
    # dict, behaviour is bit-identical to the legacy raw-V_s path.
    use_staleness_lookup: bool = False


class VsRolloutGate:
    """MECH-269b: Per-stream snapshot gate for cortical forward-prediction inputs.

    Public API:
        update_snapshots(latent_state, per_stream_vs, goal_state=None)
            Refresh per-stream snapshots from current latent values when their
            V_s is at or above snapshot_refresh_threshold. Call once per tick
            inside agent.sense(), AFTER hippocampal.update_per_stream_vs().

        gate(latent_state, per_stream_vs, side, goal_state=None) -> LatentState
            Return a gated copy of the LatentState dataclass. For each stream s
            in self.config.streams that exists on the LatentState, if
            V_s[s] < threshold[side, s], substitute the held snapshot. Streams
            absent from LatentState (z_goal lives on GoalState; z_harm_s aliases
            LatentState.z_harm) are handled at the appropriate caller site.

        gate_stream(stream_name, current_value, per_stream_vs, side) -> Tensor
            Single-stream variant for the E2_harm_a per-tick forward call (and
            for z_goal seeding into E1). Returns either current_value or the
            held snapshot for that stream, by the same threshold rule.

        reset()
            Clear all snapshots and diagnostic counters. Called from
            agent.reset() per episode.

        get_diagnostics() -> dict
            Returns a flat dict of diagnostic counters for inclusion in
            experiment manifests (n_held_e1 / n_held_e2 per stream, refresh
            counts, last-tick held flags).

    MECH-094 compliance: call-site scoping. The gate is invoked only from waking
    paths (agent.sense, agent._e1_tick, agent.select_action). It does not
    author replay/simulation content; no hypothesis_tag check is required at
    this layer.
    """

    # Map a stream name (as exposed in self.config.streams) to the LatentState
    # attribute name. z_harm_s aliases LatentState.z_harm because the SD-010
    # nociceptive-separation field on LatentState is named z_harm but
    # represents the sensory-discriminative stream that MECH-269 / MECH-269b
    # refer to as z_harm_s.
    _LATENT_FIELD_BY_STREAM = {
        "z_world": "z_world",
        "z_self": "z_self",
        "z_harm_s": "z_harm",
        "z_harm_a": "z_harm_a",
        "z_beta": "z_beta",
        "z_resource": "z_resource",
    }

    def __init__(self, config: Optional[VsRolloutGateConfig] = None) -> None:
        self.config = config or VsRolloutGateConfig()
        # Snapshot store: stream_name -> last-trusted detached tensor
        self._snapshots: Dict[str, torch.Tensor] = {}
        # Diagnostics
        self._refresh_count: Dict[str, int] = {s: 0 for s in self.config.streams}
        self._held_count_e1: Dict[str, int] = {s: 0 for s in self.config.streams}
        self._held_count_e2: Dict[str, int] = {s: 0 for s in self.config.streams}
        self._last_held_e1: Dict[str, bool] = {s: False for s in self.config.streams}
        self._last_held_e2: Dict[str, bool] = {s: False for s in self.config.streams}
        # MECH-284 staleness diagnostics: tracks the max per-stream staleness
        # the gate has been asked to subtract this episode (use_staleness_lookup
        # path only). Stays at 0.0 when staleness lookup is disabled or no
        # caller ever supplies a per_stream_staleness dict.
        self._max_staleness_seen: Dict[str, float] = {
            s: 0.0 for s in self.config.streams
        }
        self._staleness_lookup_calls: int = 0

    # ------------------------------------------------------------------
    # snapshot maintenance
    # ------------------------------------------------------------------
    def update_snapshots(
        self,
        latent_state: Any,
        per_stream_vs: Dict[str, float],
        goal_state: Optional[Any] = None,
    ) -> None:
        """Refresh held snapshots for streams whose V_s is at or above
        snapshot_refresh_threshold. Call once per waking tick inside
        agent.sense() AFTER hippocampal.update_per_stream_vs().

        Reads the canonical (un-gated) latent values so the snapshot reflects
        the most recent trusted observation, not a previously-held value.
        """
        thresh = self.config.snapshot_refresh_threshold
        for stream in self.config.streams:
            vs = per_stream_vs.get(stream)
            if vs is None or vs < thresh:
                continue
            value = self._read_stream(stream, latent_state, goal_state)
            if value is None:
                continue
            self._snapshots[stream] = value.detach().clone()
            self._refresh_count[stream] += 1

    # ------------------------------------------------------------------
    # gating
    # ------------------------------------------------------------------
    def gate(
        self,
        latent_state: Any,
        per_stream_vs: Dict[str, float],
        side: str,
        goal_state: Optional[Any] = None,
        per_stream_staleness: Optional[Dict[str, float]] = None,
    ) -> Any:
        """Return a gated copy of LatentState for E1 / E2 consumers.

        For each configured stream that lives on LatentState, if V_s is below
        the side threshold, substitute the held snapshot. Streams that do not
        live on LatentState (e.g. z_goal on GoalState) must be gated separately
        via gate_stream() at the appropriate caller site.

        The returned LatentState is constructed via dataclasses.replace so all
        non-gated fields (precision dict, hypothesis_tag, aux predictions,
        etc.) are preserved unchanged.

        MECH-284 staleness wiring (Q-040b strong reading): when
        config.use_staleness_lookup is True AND per_stream_staleness is
        supplied, the threshold comparison runs on
            effective_vs = raw_vs - per_stream_staleness[s]
        instead of raw_vs. With either condition off the gate falls back to
        the raw-V_s path (bit-identical to the legacy gate semantics).
        """
        if side not in ("e1", "e2"):
            raise ValueError(
                f"VsRolloutGate.gate side must be 'e1' or 'e2'; got {side!r}"
            )
        from dataclasses import replace as _dc_replace

        per_side_held = (
            self._last_held_e1 if side == "e1" else self._last_held_e2
        )
        per_side_count = (
            self._held_count_e1 if side == "e1" else self._held_count_e2
        )
        # Reset this-tick held flags for streams covered by this gate call.
        for stream in self.config.streams:
            if stream in self._LATENT_FIELD_BY_STREAM:
                per_side_held[stream] = False

        substitutions: Dict[str, torch.Tensor] = {}
        for stream in self.config.streams:
            field_name = self._LATENT_FIELD_BY_STREAM.get(stream)
            if field_name is None:
                # Stream lives off LatentState (e.g. z_goal); handled by
                # gate_stream() at caller site.
                continue
            if not hasattr(latent_state, field_name):
                continue
            current_value = getattr(latent_state, field_name)
            if current_value is None:
                continue
            held_value, held = self._gate_value(
                stream, current_value, per_stream_vs, side,
                per_stream_staleness=per_stream_staleness,
            )
            if held:
                substitutions[field_name] = held_value
                per_side_held[stream] = True
                per_side_count[stream] += 1

        if not substitutions:
            return latent_state
        return _dc_replace(latent_state, **substitutions)

    def gate_stream(
        self,
        stream_name: str,
        current_value: Optional[torch.Tensor],
        per_stream_vs: Dict[str, float],
        side: str,
        per_stream_staleness: Optional[Dict[str, float]] = None,
    ) -> Optional[torch.Tensor]:
        """Single-stream variant for callers operating on a bare tensor (e.g.
        the per-tick E2_harm_a forward call in agent.select_action, or the
        z_goal seed read in agent._e1_tick).

        Returns the held snapshot when V_s is below the side threshold and a
        snapshot exists; otherwise returns current_value unchanged. None
        propagates as None (no snapshot to substitute).

        per_stream_staleness participates in the threshold comparison the same
        way as in gate() (MECH-284 wiring).
        """
        if side not in ("e1", "e2"):
            raise ValueError(
                f"VsRolloutGate.gate_stream side must be 'e1' or 'e2'; got {side!r}"
            )
        if current_value is None:
            return None
        held_value, held = self._gate_value(
            stream_name, current_value, per_stream_vs, side,
            per_stream_staleness=per_stream_staleness,
        )
        per_side_held = (
            self._last_held_e1 if side == "e1" else self._last_held_e2
        )
        per_side_count = (
            self._held_count_e1 if side == "e1" else self._held_count_e2
        )
        # Single-stream calls advance the held flag/counter only for the named
        # stream and only on this side, leaving other streams untouched.
        if stream_name in self.config.streams:
            per_side_held[stream_name] = held
            if held:
                per_side_count[stream_name] += 1
        return held_value if held else current_value

    # ------------------------------------------------------------------
    # diagnostics + reset
    # ------------------------------------------------------------------
    def get_diagnostics(self) -> Dict[str, Any]:
        """Return a flat dict of gate diagnostics for experiment manifests."""
        out: Dict[str, Any] = {}
        for stream in self.config.streams:
            out[f"vs_gate_held_e1_{stream}"] = int(self._held_count_e1.get(stream, 0))
            out[f"vs_gate_held_e2_{stream}"] = int(self._held_count_e2.get(stream, 0))
            out[f"vs_gate_refresh_{stream}"] = int(self._refresh_count.get(stream, 0))
        out["vs_gate_n_snapshots"] = int(len(self._snapshots))
        out["vs_gate_total_held_e1"] = int(sum(self._held_count_e1.values()))
        out["vs_gate_total_held_e2"] = int(sum(self._held_count_e2.values()))
        # MECH-284 staleness diagnostics.
        out["vs_gate_use_staleness_lookup"] = bool(
            self.config.use_staleness_lookup
        )
        out["vs_gate_staleness_lookup_calls"] = int(
            self._staleness_lookup_calls
        )
        for stream in self.config.streams:
            out[f"vs_gate_max_staleness_{stream}"] = float(
                self._max_staleness_seen.get(stream, 0.0)
            )
        return out

    def reset(self) -> None:
        """Clear snapshots and diagnostic counters. Called per-episode from
        agent.reset()."""
        self._snapshots.clear()
        for d in (
            self._refresh_count, self._held_count_e1, self._held_count_e2,
        ):
            for k in d:
                d[k] = 0
        for d in (self._last_held_e1, self._last_held_e2):
            for k in d:
                d[k] = False
        for k in self._max_staleness_seen:
            self._max_staleness_seen[k] = 0.0
        self._staleness_lookup_calls = 0

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _gate_value(
        self,
        stream: str,
        current_value: torch.Tensor,
        per_stream_vs: Dict[str, float],
        side: str,
        per_stream_staleness: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, bool]:
        """Return (value_to_use, held_flag). held=True means snapshot
        substituted; held=False means current value passes through.

        MECH-284 staleness wiring: when use_staleness_lookup AND a
        per_stream_staleness dict is supplied, effective_vs = raw_vs -
        staleness[stream] (clipped at -1.0 floor for numerical stability;
        the gate's threshold check tolerates negative values, so the
        clip is purely diagnostic). With either condition off the
        comparison runs on raw_vs unchanged.
        """
        threshold = self._side_threshold(stream, side)
        vs = per_stream_vs.get(stream)
        # No V_s reading -> passthrough (gate has nothing to act on yet).
        if vs is None:
            return current_value, False
        # MECH-284: subtract per-stream staleness from V_s before threshold.
        if (
            self.config.use_staleness_lookup
            and per_stream_staleness is not None
        ):
            self._staleness_lookup_calls += 1
            staleness = float(per_stream_staleness.get(stream, 0.0))
            if staleness < 0.0:
                staleness = 0.0
            if staleness > self._max_staleness_seen.get(stream, 0.0):
                self._max_staleness_seen[stream] = staleness
            effective_vs = float(vs) - staleness
            if effective_vs < -1.0:
                effective_vs = -1.0
            vs = effective_vs
        if vs >= threshold:
            return current_value, False
        snapshot = self._snapshots.get(stream)
        if snapshot is None:
            # No held snapshot to substitute. Per config, default is to pass
            # current value through unchanged.
            if self.config.unknown_stream_passes:
                return current_value, False
            return current_value, False
        # Reshape snapshot to match current tensor shape on the leading
        # batch axis if shapes diverge by the batch dim only. The encoder
        # emits [batch, dim] tensors and the snapshot was captured in the
        # same form, so under normal use shapes match; this guard handles
        # the rare case where batch dim grows mid-episode.
        if snapshot.shape != current_value.shape:
            try:
                snapshot = snapshot.expand_as(current_value).clone()
            except RuntimeError:
                return current_value, False
        return snapshot.to(current_value.device), True

    def _side_threshold(self, stream: str, side: str) -> float:
        if side == "e1":
            override = self.config.e1_threshold_per_stream.get(stream)
            return float(override) if override is not None else float(self.config.e1_threshold)
        override = self.config.e2_threshold_per_stream.get(stream)
        return float(override) if override is not None else float(self.config.e2_threshold)

    def _read_stream(
        self,
        stream: str,
        latent_state: Any,
        goal_state: Optional[Any],
    ) -> Optional[torch.Tensor]:
        """Read a stream's current tensor from LatentState (preferred) or
        GoalState (z_goal). Returns None if unavailable on this tick."""
        field_name = self._LATENT_FIELD_BY_STREAM.get(stream)
        if field_name is not None and hasattr(latent_state, field_name):
            value = getattr(latent_state, field_name)
            if value is not None:
                return value
        if stream == "z_goal" and goal_state is not None:
            z_goal = getattr(goal_state, "z_goal", None)
            return z_goal
        return None
