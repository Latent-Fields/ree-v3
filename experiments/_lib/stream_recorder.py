"""
Shared per-step multi-stream trace recorder (Q-081 / MECH-466 / INV-091 / ARC-112).

Why this exists
---------------
The Q-081 telemetry audit
(REE_assembly/evidence/planning/q081_cross_stream_telemetry_audit.md) found that no
manifest in the 665-run corpus carries a per-timestep multi-stream trace, and that the
one partial precedent -- the `*_episode_log.json` block copy-pasted across 15 experiment
scripts -- cannot serve Q-081 because it records NORMS (z_world_norm, z_harm_norm,
z_beta_val), collapsing a 32-D latent to one scalar and destroying exactly the
"recurrent system-level configuration" structure the question is about. There was no
shared writer in experiments/_lib/. This module is that writer.

The three properties that make it usable for Q-081 (each from a specific audit finding):

1. VECTORS, NOT NORMS (audit section 1). Every latent stream is stored at full width.
   Norms are trivially derivable downstream; the reverse is not.

2. PER-SIGNAL FRESHNESS FLAGS (audit section 3) -- the load-bearing one. REE's clock
   defaults to E1 every step, E2 every 3, E3 every 10 (config.py:2122-2124), and
   select_action short-circuits between E3 ticks (agent.py:5463), so E3 candidate
   scores, operating_mode, hippocampal proposals and commitment state are HELD STALE on
   9 of every 10 steps. A recorder without freshness flags writes 9 duplicates + 1 fresh
   value per E3 stream, and cross-stream analysis over that finds strong regular shared
   structure that is a SAMPLER ARTEFACT -- Outcome B of the Q-081 taxonomy, the exact
   null the experiment exists to exclude. Every stream therefore carries a per-step
   `<name>__fresh` flag, and each stream declares HOW its flag was derived:

     identity -- the source cache object was REASSIGNED this step. Exact: the recorder
                 holds a strong reference to the previous object, so its id() cannot be
                 recycled by the garbage collector and an identity comparison is sound.
     clock    -- derived from agent.clock. E3 freshness is read as
                 `clock._e3_phase_step == 0` after act(), which is exact under BOTH
                 MECH-091 phase resets and MECH-093 arousal rate modulation; a hardcoded
                 `step % 10` would be wrong under either.
     value    -- the value differs from the previous step. A PROXY: a recompute that
                 happens to yield an identical value reads as held. Used only where no
                 cache object or clock edge exists (z_goal, sleep phase, offline flag).
     event    -- fresh iff at least one event fired this step. Legitimately sparse.

3. STORE BY REFERENCE (audit section 4 item 2). finalize() hands the arrays to
   trace_store.TraceStore and returns a small pointer dict. The blob never enters the
   git-tracked coordination plane, which four concurrent phase3 writers share.

Where to call it
----------------
    rec = StreamTraceRecorder(agent, run_id=run_id)
    for t in range(n_steps):
        action = agent.act(obs)                 # or act_with_split_obs / act_with_log_prob
        obs, reward, done, info = env.step(action)
        agent.update_z_goal(...)                # z_goal is FLAT unless the loop drives it
        rec.on_step(extras={"reward": reward})  # AFTER act(), BEFORE the next act()
    pointer = rec.finalize()
    manifest["stream_trace"] = pointer

The "after act(), before the next act()" placement is a HARD requirement, not a style
preference (audit section 3). agent.py:4217-4220 calls drain_boundary_events() /
drain_broadcast_events() and DISCARDS the returned lists, then clears the queue at the
start of the next sense(). Sampling anywhere else records nothing, silently. The
recorder reads `hippocampal._boundary_event_queue` NON-DESTRUCTIVELY -- it copies and
never drains, so the agent's own flush behaviour is bit-identical with the recorder
attached.

The recorder mutates no agent state and calls no method with side effects. `get_state()`
and `get_commitment_state()` are pure constructors; every other read is an attribute
fetch. A contract test asserts the action sequence is unchanged with the recorder
attached.

Streams that need the LOOP to drive them (audit section 4 item 5): `update_z_goal()` is
loop-invoked -- z_goal stays zero if the experiment never calls it, and the recorder will
faithfully record a flat stream. Sleep-phase markers need `use_sleep_loop=True`
(see q081_profile).

ASCII-only output (repo rule).
"""

from __future__ import annotations

import socket
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:  # torch is present in every experiment process; guard keeps import cheap for tools
    import torch
except Exception:  # pragma: no cover - torch is always available in experiment runs
    torch = None  # type: ignore

from .trace_store import TraceStore

TRACE_SCHEMA = "stream_trace/v1"

# Freshness derivation kinds (see module docstring).
FRESH_IDENTITY = "identity"
FRESH_CLOCK = "clock"
FRESH_VALUE = "value"
FRESH_EVENT = "event"

# Fixed key order for the salience operating_mode soft vector, so the recorded vector
# is comparable across steps and across runs.
OPERATING_MODES = (
    "external_task",
    "internal_reflection",
    "offline_consolidation",
)


def _vec(t: Any) -> Optional[np.ndarray]:
    """Tensor -> flat float32 numpy row. Batch dim is dropped (row 0). None passes through."""
    if t is None:
        return None
    if torch is not None and isinstance(t, torch.Tensor):
        a = t.detach().cpu().numpy()
    else:
        a = np.asarray(t)
    a = np.asarray(a, dtype=np.float32)
    if a.ndim >= 2:
        a = a[0]
    return a.reshape(-1)


class _Stream:
    """One recorded signal: values + per-step fresh/valid flags."""

    def __init__(self, name: str, freshness: str, note: str = ""):
        self.name = name
        self.freshness = freshness
        self.note = note
        self.values: List[Optional[np.ndarray]] = []
        self.fresh: List[bool] = []
        self.width: Optional[int] = None

    def append(self, value: Optional[np.ndarray], fresh: bool) -> None:
        if value is not None:
            if self.width is None:
                self.width = int(value.shape[0])
            elif int(value.shape[0]) != self.width:
                raise ValueError(
                    f"stream '{self.name}' changed width: {self.width} -> "
                    f"{int(value.shape[0])}. A stream must have fixed width for the "
                    f"whole run."
                )
        self.values.append(value)
        self.fresh.append(bool(fresh))

    def arrays(self) -> Dict[str, np.ndarray]:
        n = len(self.values)
        width = self.width if self.width is not None else 1
        out = np.full((n, width), np.nan, dtype=np.float32)
        valid = np.zeros(n, dtype=bool)
        for i, v in enumerate(self.values):
            if v is not None:
                out[i] = v
                valid[i] = True
        return {
            self.name: out,
            f"{self.name}__fresh": np.asarray(self.fresh, dtype=bool),
            f"{self.name}__valid": valid,
        }

    def descriptor(self) -> Dict[str, Any]:
        return {
            "freshness_source": self.freshness,
            "width": int(self.width) if self.width is not None else 0,
            "n_valid": int(sum(v is not None for v in self.values)),
            "n_fresh": int(sum(self.fresh)),
            "note": self.note,
        }


class StreamTraceRecorder:
    """Per-step multi-stream trace recorder. One instance per run (across episodes).

    Args:
        agent: a live REEAgent.
        run_id: the run_id this trace belongs to; recorded in the blob meta.
        store: a TraceStore (default: the machine-local content-addressed store).
        substrate_declaration: the q081_profile.q081_substrate_declaration() block, so a
            fetched trace is self-describing about which non-default flags were on.
        capture_e1_hidden: record the E1 LSTM hidden state (h,c concatenated). On by
            default -- it is signal 1 of the Q-081 checklist.
    """

    def __init__(
        self,
        agent,
        run_id: Optional[str] = None,
        store: Optional[TraceStore] = None,
        substrate_declaration: Optional[Dict[str, Any]] = None,
        capture_e1_hidden: bool = True,
        machine: Optional[str] = None,
    ):
        self.agent = agent
        self.run_id = run_id
        self.store = store if store is not None else TraceStore()
        self.substrate_declaration = substrate_declaration
        self.capture_e1_hidden = bool(capture_e1_hidden)
        self.machine = machine if machine is not None else socket.gethostname()

        self._streams: Dict[str, _Stream] = {}
        self._prev_obj: Dict[str, Any] = {}   # strong refs -- see docstring on identity
        self._prev_val: Dict[str, Any] = {}
        self._n_steps = 0
        self._episode_index = 0
        self._finalized = False

        # Clock block, one row per step.
        self._clock_rows: List[List[float]] = []
        self._clock_cols = (
            "global_step", "e1_tick", "e2_tick", "e3_tick", "theta_tick",
            "sweep_active", "e3_steps_per_tick", "e3_phase_step", "episode_index",
        )

        # Ragged E3 candidate scores.
        self._scores_values: List[np.ndarray] = []
        self._scores_offsets: List[int] = [0]
        self._scores_fresh: List[bool] = []

        # Event records (strings + floats) -- kept in the blob meta, not the manifest.
        self._events: List[Dict[str, Any]] = []

        # Caller-supplied per-step extras (reward, position, ...). Numeric only.
        self._extras: Dict[str, List[float]] = {}

    # -- freshness helpers ------------------------------------------------

    def _fresh_by_identity(self, key: str, obj: Any) -> bool:
        """True iff `obj` is not the same object as last step.

        Holds a STRONG ref to the previous object so it cannot be freed and have its
        id() recycled by the allocator -- an id()-only comparison would then report a
        recompute as a hold, at random. Exactly one previous object is retained per key,
        so the retention is O(1) in run length; it can pin one step's autograd graph in
        a training loop, which is bounded and does not affect autograd itself.
        """
        prev = self._prev_obj.get(key, _MISSING)
        self._prev_obj[key] = obj
        if prev is _MISSING:
            return True
        return obj is not prev

    def _fresh_by_value(self, key: str, value: Optional[np.ndarray]) -> bool:
        prev = self._prev_val.get(key, _MISSING)
        self._prev_val[key] = None if value is None else value.copy()
        if prev is _MISSING:
            return True
        if prev is None or value is None:
            return (prev is None) != (value is None)
        if prev.shape != value.shape:
            return True
        return not np.array_equal(prev, value)

    def _stream(self, name: str, freshness: str, note: str = "") -> _Stream:
        s = self._streams.get(name)
        if s is None:
            s = _Stream(name, freshness, note)
            # Backfill any steps recorded before this stream first appeared, so every
            # stream has exactly one row per step.
            for _ in range(self._n_steps):
                s.append(None, False)
            self._streams[name] = s
        return s

    def _put(self, name: str, freshness: str, value: Optional[np.ndarray],
             fresh: bool, note: str = "") -> None:
        self._stream(name, freshness, note).append(value, fresh)

    # -- the per-step call ------------------------------------------------

    def on_step(self, extras: Optional[Dict[str, float]] = None) -> None:
        """Sample every stream once. Call AFTER agent.act(), BEFORE the next act()."""
        if self._finalized:
            raise RuntimeError("on_step() after finalize()")
        agent = self.agent

        clock = getattr(agent, "clock", None)
        e3_tick = False
        if clock is not None:
            # Exact under MECH-091 phase reset and MECH-093 rate modulation: advance()
            # zeroes _e3_phase_step on the tick and increments it on every other step.
            e3_tick = int(getattr(clock, "_e3_phase_step", -1)) == 0
            gstep = int(getattr(clock, "global_step", 0))
            e2_per = max(1, int(getattr(clock, "e2_steps_per_tick", 1)))
            e1_per = max(1, int(getattr(clock, "e1_steps_per_tick", 1)))
            theta = max(1, int(getattr(clock, "theta_buffer_size", 1)))
            self._clock_rows.append([
                float(gstep),
                float(gstep % e1_per == 0),
                float(gstep % e2_per == 0),
                float(e3_tick),
                float(gstep % theta == 0),
                float(bool(getattr(clock, "sweep_active", False))),
                float(int(getattr(clock, "e3_steps_per_tick", 0))),
                float(int(getattr(clock, "_e3_phase_step", -1))),
                float(self._episode_index),
            ])
        else:
            self._clock_rows.append([float("nan")] * len(self._clock_cols))

        # --- primary ungated snapshot (audit section 2, "most useful single find") ---
        state = agent.get_state()
        latent = state.latent_state
        lat_fresh = self._fresh_by_identity("latent", latent)
        for name, attr in (
            ("z_self", "z_self"), ("z_world", "z_world"), ("z_beta", "z_beta"),
            ("z_theta", "z_theta"), ("z_delta", "z_delta"),
            ("z_harm", "z_harm"), ("z_harm_a", "z_harm_a"),
        ):
            v = _vec(getattr(latent, attr, None)) if latent is not None else None
            self._put(name, FRESH_IDENTITY, v, lat_fresh and v is not None,
                      note="LatentState field; re-encoded every step (E1 rate)")

        # DELIBERATELY per-step quantities ONLY. get_state() also carries precision,
        # running_variance and is_committed, but those are E3-DERIVED: bundling them here
        # would flag them fresh every step under the latent's identity and hand an
        # analyst exactly the artefact the freshness flags exist to prevent. They are
        # recorded once, in `e3_commitment`, with clock-derived freshness.
        self._put("agent_scalars", FRESH_IDENTITY, np.asarray([
            float(state.step),
            float(state.harm_accumulated),
            float(bool(state.beta_elevated)),
            float(int(state.e3_steps_per_tick)),
        ], dtype=np.float32), lat_fresh,
            note="get_state() PER-STEP scalars only: step, harm_accumulated, "
                 "beta_elevated, e3_steps_per_tick. The E3-derived precision / "
                 "running_variance / is_committed live in e3_commitment, which flags "
                 "them with the E3 clock edge instead of every step.")

        # --- E1 hidden state (signal 1) ---
        if self.capture_e1_hidden:
            hidden = getattr(getattr(agent, "e1", None), "_hidden_state", None)
            fresh = self._fresh_by_identity("e1_hidden", hidden)
            v = None
            if hidden is not None:
                parts = [_vec(x) for x in hidden] if isinstance(hidden, (tuple, list)) else [_vec(hidden)]
                parts = [p for p in parts if p is not None]
                if parts:
                    v = np.concatenate(parts).astype(np.float32)
            self._put("e1_hidden", FRESH_IDENTITY, v, fresh and v is not None,
                      note="E1 LSTM (h, c) concatenated")

        # --- E2 per-step self prediction error (signal 2, TPJ comparator) ---
        tpj = getattr(agent, "_tpj_last_agency_signal", None)
        fresh = self._fresh_by_identity("e2_self_pe", tpj)
        self._put("e2_self_pe", FRESH_IDENTITY, _vec(tpj), fresh and tpj is not None,
                  note="TRUE E2 predict_next_self error (agent.py:3047-3062), NOT the E3 "
                       "world-rollout error. Requires use_tpj_comparator. E3 CADENCE, "
                       "not per-step: the efference copy is staged at the end of "
                       "select_action (agent.py:7582), after the between-tick "
                       "short-circuit, so no comparison is staged on a held step.")

        # --- E3 candidate scores (signal 3): ragged, E3-rate ---
        e3 = getattr(agent, "e3", None)
        scores = getattr(e3, "last_scores", None) if e3 is not None else None
        scores_fresh = self._fresh_by_identity("e3_scores", scores)
        sv = _vec(scores)
        if sv is None:
            sv = np.zeros(0, dtype=np.float32)
        self._scores_values.append(sv)
        self._scores_offsets.append(self._scores_offsets[-1] + int(sv.shape[0]))
        self._scores_fresh.append(bool(scores_fresh and scores is not None))
        self._put("e3_n_candidates", FRESH_IDENTITY,
                  np.asarray([float(sv.shape[0])], dtype=np.float32),
                  scores_fresh and scores is not None,
                  note="candidate count for the ragged e3_scores block")

        # --- E3 commitment state (signal 4): clock-derived freshness ---
        if e3 is not None and hasattr(e3, "get_commitment_state"):
            cs = e3.get_commitment_state()
            self._put("e3_commitment", FRESH_CLOCK, np.asarray([
                float(cs.get("precision", float("nan"))),
                float(cs.get("running_variance", float("nan"))),
                float(cs.get("commit_threshold", float("nan"))),
                float(bool(cs.get("committed_now", False))),
                float(bool(cs.get("is_committed", False))),
            ], dtype=np.float32), e3_tick,
                note="get_commitment_state() builds a NEW dict per call, so identity "
                     "cannot detect recompute; freshness is the E3 clock edge. "
                     "precision, running_variance, commit_threshold, committed_now, is_committed")

        # --- hippocampal proposals (signal 13): E3-rate ---
        cands = getattr(agent, "_committed_candidates", None)
        fresh = self._fresh_by_identity("hippocampal_proposals", cands)
        n_cands = float(len(cands)) if cands is not None else float("nan")
        self._put("hippocampal_proposals", FRESH_IDENTITY,
                  np.asarray([n_cands], dtype=np.float32),
                  fresh and cands is not None,
                  note="len(agent._committed_candidates); reassigned on E3 tick")

        # --- operating_mode (signal 10): salience coordinator, E3-rate ---
        sal = getattr(agent, "_salience_last_tick", None)
        fresh = self._fresh_by_identity("operating_mode", sal)
        mode_vec = None
        if isinstance(sal, dict):
            om = sal.get("operating_mode") or {}
            mode_vec = np.asarray(
                [float(om.get(k, 0.0)) for k in OPERATING_MODES]
                + [float(OPERATING_MODES.index(sal["current_mode"]))
                   if sal.get("current_mode") in OPERATING_MODES else float("nan"),
                   float(bool(sal.get("mode_switch_trigger", False))),
                   float(sal.get("salience_aggregate", float("nan")))],
                dtype=np.float32,
            )
        self._put("operating_mode", FRESH_IDENTITY, mode_vec,
                  fresh and mode_vec is not None,
                  note="soft prob over %s, then current_mode index, mode_switch_trigger, "
                       "salience_aggregate. Requires use_salience_coordinator."
                       % (list(OPERATING_MODES),))

        # --- beta gate (signal 11): every step ---
        bg = getattr(agent, "beta_gate", None)
        bv = None
        if bg is not None:
            bv = np.asarray([float(bool(getattr(bg, "is_elevated", False)))], dtype=np.float32)
        self._put("beta_elevated", FRESH_CLOCK, bv, True,
                  note="beta gate is evaluated every step, so fresh is always True")

        # --- z_goal (signal 9): loop-driven, value-proxy freshness ---
        gs = getattr(agent, "goal_state", None)
        zg = _vec(getattr(gs, "z_goal", None)) if gs is not None else None
        self._put("z_goal", FRESH_VALUE, zg, self._fresh_by_value("z_goal", zg),
                  note="agent.goal_state.z_goal. FLAT unless the experiment loop calls "
                       "agent.update_z_goal(). Freshness is a value-change proxy.")

        # --- boundary / broadcast events (signal 12): NON-DESTRUCTIVE read ---
        hipp = getattr(agent, "hippocampal", None)
        for stream_name, queue_attr in (
            ("boundary_events", "_boundary_event_queue"),
            ("broadcast_events", "_broadcast_event_queue"),
        ):
            queue = getattr(hipp, queue_attr, None) if hipp is not None else None
            n = 0
            strength = float("nan")
            if queue:
                # COPY, never drain -- the agent flushes these itself at the start of
                # the next sense() (agent.py:4217-4220).
                evs = list(queue)
                n = len(evs)
                posts = [float(getattr(e, "posterior", float("nan"))) for e in evs]
                finite = [p for p in posts if np.isfinite(p)]
                strength = max(finite) if finite else float("nan")
                for e in evs:
                    self._events.append({
                        "stream": stream_name,
                        "step_index": self._n_steps,
                        "t": int(getattr(e, "t", -1)),
                        "posterior": float(getattr(e, "posterior", float("nan"))),
                        "scale": str(getattr(e, "scale", getattr(e, "source_scale", ""))),
                        "segment_id_old": str(getattr(e, "segment_id_old",
                                                      getattr(e, "source_segment_id_old", ""))),
                        "segment_id_new": str(getattr(e, "segment_id_new",
                                                      getattr(e, "source_segment_id_new", ""))),
                    })
            self._put(stream_name, FRESH_EVENT,
                      np.asarray([float(n), strength], dtype=np.float32), n > 0,
                      note="[count, max posterior] read non-destructively from "
                           "hippocampal.%s. Legitimately sparse." % queue_attr)

        # --- offline / sleep-phase markers ---
        offline = getattr(getattr(agent, "e1", None), "_offline_mode", None)
        ov = None if offline is None else np.asarray([float(bool(offline))], dtype=np.float32)
        self._put("offline_mode", FRESH_VALUE, ov, self._fresh_by_value("offline_mode", ov),
                  note="agent.e1._offline_mode")

        sleep_loop = getattr(agent, "sleep_loop", None)
        phase_v = None
        phase_name = ""
        if sleep_loop is not None:
            phase = getattr(getattr(sleep_loop, "state", None), "phase", None)
            if phase is not None:
                phase_name = getattr(phase, "name", str(phase))
                phase_v = np.asarray([float(_phase_code(phase_name))], dtype=np.float32)
        self._put("sleep_phase", FRESH_VALUE, phase_v,
                  self._fresh_by_value("sleep_phase", phase_v),
                  note="agent.sleep_loop.state.phase code (see meta.sleep_phase_codes). "
                       "Requires use_sleep_loop; flat otherwise.")
        # --- caller extras ---
        if extras:
            for k, v in extras.items():
                col = self._extras.setdefault(k, [float("nan")] * self._n_steps)
                col.append(float(v))
        for k, col in self._extras.items():
            if len(col) == self._n_steps:
                col.append(float("nan"))

        self._n_steps += 1

    def on_episode_end(self) -> None:
        """Advance the episode index recorded in the clock block."""
        self._episode_index += 1

    # -- finalize ---------------------------------------------------------

    def finalize(self, extra_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Write the trace to the content-addressed store; return the manifest pointer.

        The returned dict is small and JSON-safe. It is the ONLY thing that should reach
        a manifest -- the arrays stay in the blob.
        """
        if self._finalized:
            raise RuntimeError("finalize() called twice")
        self._finalized = True

        arrays: Dict[str, np.ndarray] = {}
        descriptors: Dict[str, Any] = {}
        for name, s in sorted(self._streams.items()):
            arrays.update(s.arrays())
            descriptors[name] = s.descriptor()

        arrays["clock"] = np.asarray(self._clock_rows, dtype=np.float32).reshape(
            self._n_steps, len(self._clock_cols)
        )
        arrays["e3_scores__values"] = (
            np.concatenate(self._scores_values).astype(np.float32)
            if self._scores_values else np.zeros(0, dtype=np.float32)
        )
        arrays["e3_scores__offsets"] = np.asarray(self._scores_offsets, dtype=np.int64)
        arrays["e3_scores__fresh"] = np.asarray(self._scores_fresh, dtype=bool)
        for k, col in sorted(self._extras.items()):
            arrays[f"extra__{k}"] = np.asarray(col[: self._n_steps], dtype=np.float32)

        meta = {
            "trace_schema": TRACE_SCHEMA,
            "run_id": self.run_id,
            "machine": self.machine,
            "n_steps": self._n_steps,
            "n_episodes": self._episode_index + (1 if self._n_steps else 0),
            "clock_columns": list(self._clock_cols),
            "operating_modes": list(OPERATING_MODES),
            "sleep_phase_codes": dict(_PHASE_CODES),
            "streams": descriptors,
            "freshness_semantics": {
                FRESH_IDENTITY: "source cache object was reassigned this step (exact)",
                FRESH_CLOCK: "derived from agent.clock tick edges (exact)",
                FRESH_VALUE: "value differs from previous step (PROXY: an identical "
                             "recompute reads as held)",
                FRESH_EVENT: "at least one event fired this step",
            },
            "ragged": {
                "e3_scores": "values[offsets[i]:offsets[i+1]] is step i's candidate scores"
            },
            "events": self._events,
            "audit": "REE_assembly/evidence/planning/q081_cross_stream_telemetry_audit.md",
        }
        if self.substrate_declaration is not None:
            meta["non_default_substrate"] = self.substrate_declaration
        if extra_meta:
            meta.update(extra_meta)

        pointer = self.store.put(arrays, meta=meta)
        pointer["trace_schema"] = TRACE_SCHEMA
        pointer["run_id"] = self.run_id
        pointer["n_steps"] = self._n_steps
        pointer["stream_names"] = sorted(self._streams)
        # Trim the per-array shape map to the PRIMARY streams: the __fresh / __valid
        # companions are mechanically derivable and would roughly triple the pointer for
        # no information. The full descriptor set lives in the blob meta.
        pointer["streams"] = {
            k: v for k, v in pointer["streams"].items()
            if not k.endswith("__fresh") and not k.endswith("__valid")
        }
        # Deliberately NOT copied into the pointer: the arrays, the event list, and the
        # per-stream descriptors. The manifest gets a reference, not a payload.
        return pointer


class _Missing:
    __slots__ = ()


_MISSING = _Missing()

# Sleep-phase name -> integer code, populated as phases are observed. Recorded in meta
# so the numeric stream is decodable.
_PHASE_CODES: Dict[str, int] = {"WAKING": 0, "SWS_ANALOG": 1, "REM_ANALOG": 2}


def _phase_code(name: str) -> int:
    if name not in _PHASE_CODES:
        _PHASE_CODES[name] = len(_PHASE_CODES)
    return _PHASE_CODES[name]


def derive_norms(arrays: Dict[str, np.ndarray], stream: str) -> np.ndarray:
    """Convenience: per-step L2 norm of a recorded vector stream.

    Provided so a consumer can reproduce the legacy `*_norm` episode-log fields from the
    full vectors. The reverse direction is impossible, which is why the recorder stores
    vectors (audit section 1).
    """
    return np.linalg.norm(np.nan_to_num(arrays[stream], nan=0.0), axis=1)


def rate_matched_shuffle_index(
    clock: np.ndarray,
    clock_columns: Sequence[str],
    fresh: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Index permutation for the Q-081 rate-matched shuffle control.

    Q-081's `what_would_answer` names this control as not optional, and the audit
    (section 3) sharpens it: the shuffle must be matched to the ACTUAL tick schedule,
    not to a nominal `% 10`. This permutes only the FRESH samples among themselves and
    leaves held samples where they are, so the shuffled surrogate keeps the identical
    hold pattern and destroys only the cross-stream alignment. A naive whole-series
    shuffle would destroy the hold structure too and make the control pass trivially.
    """
    del clock, clock_columns  # accepted for call-site clarity; schedule comes from `fresh`
    idx = np.arange(len(fresh))
    fresh_pos = np.flatnonzero(np.asarray(fresh, dtype=bool))
    if fresh_pos.size > 1:
        idx[fresh_pos] = fresh_pos[rng.permutation(fresh_pos.size)]
    return idx
