"""ARC-006 / MECH-045 -- Token-instance object-file / entity-persistence buffer.

The TOKEN projection of the ARC-080 type/token/anchor triad: a per-entity store
keyed by an identity TOKEN that survives the entity MOVING to a new cell, with
NO symbolic/type label, attention-gated, with precision-weighted continuity.
This is the missing third store (the TYPE store is SD-057 IncentiveTokenBank in
goal.py; the ANCHOR store is the SD-039/MECH-292 ghost-goal bank in
hippocampal/). Design memo (the spec this implements):
REE_assembly/docs/architecture/mech_045_object_file_buffer.md.

Functional contract (memo Section 2):
  C1 Token KEY   -- per-entity token id by spatiotemporal-continuity data-
                    association over z_world-local features, NO type label;
                    the token survives the entity moving to a new cell.
  C2 Feature buf -- a running, precision-weighted bound feature estimate per
                    token (the "file").
  C3 Persistence -- keep a token alive for a bounded number of ticks when the
                    entity is occluded / out of view, then evict (track death).
  C4 Attention   -- binding/update gated by salience; tokens compete for a
                    bounded buffer (the FINST ~4-5 capacity analogue).
  C5 Precision   -- the association decision and feature update are precision-
                    weighted; high-precision observations bind more strongly.

The update rule is DeepSORT-style data association (memo Section 4.2):
predict -> motion-gate -> appearance-cost match -> update -> birth -> death.

This module is NON-TRAINABLE (no nn.Module, no parameters, no gradient flow);
it is a stateful regulator over detached z_world-local features, mirroring the
IncentiveTokenBank (goal.py), BlockedAgency, and EscapeAffordanceBridge pattern.

MECH-094 (memo Section 4.2): the buffer updates ONLY on the waking stream.
Under simulation / replay / sleep (simulation_mode=True) update() is a no-op
(no births, no feature writes) so a moved-imagined-object cannot rewrite the
waking object-file.

v1 detector dependency (memo Section 4.4): the buffer is feature-source
agnostic -- update() consumes an explicit list of EntityObservation (grid
position + a z_world-local feature vector). The caller (experiment / agent
harness) builds those from the SD-049 per-type resource-field views + the
grid's discrete object/hazard cells. A learned figure-ground proposer is a
later, separate concern (it overlaps MECH-278 object-schema formation, V4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class EntityObservation:
    """A single currently-perceived candidate entity (memo Section 4.2).

    Attributes:
        pos:          grid position [2] (row, col) as a float tensor.
        z:            z_world-local feature vector [world_dim]. The
                      re-identification "appearance"; the d_cos feature term of
                      the association cost is computed over this.
        salience:     attention salience in [0, 1]; the C4 birth gate
                      (obf_min_birth_salience floor) reads this.
        precision:    observation precision in [0, 1]; the C5 weight on the
                      feature-EMA update and on the precision-weighted feature
                      distance.
        resource_tag: optional SD-049 per-type tag observed when the entity is
                      in a resource cell. Stored on the token as an OPTIONAL
                      type_hint (wiring hook); NEVER used as the association
                      key (the key is continuity, not type).
    """

    pos: torch.Tensor
    z: torch.Tensor
    salience: float = 1.0
    precision: float = 1.0
    resource_tag: Optional[int] = None


@dataclass
class ObjectFile:
    """One per-token object-file (memo Section 4.1)."""

    token_id: int
    z_features: torch.Tensor          # [world_dim] -- C2 bound feature buffer (EMA)
    last_pos: torch.Tensor            # [2] -- last observed (row, col); continuity gate
    precision: float                  # C5 running confidence of this track
    last_seen_tick: int               # C3 persistence / eviction clock
    hits: int = 1                     # n re-observations (track maturity)
    type_hint: Optional[int] = None   # SD-049 tag if last seen in a resource cell


@dataclass
class ObjectFileBufferConfig:
    """No-op defaults: with use_object_file_buffer=False the agent never builds
    the buffer; the defaults below match memo Section 4.3."""

    use_object_file_buffer: bool = False
    max_tokens: int = 5               # obf_max_tokens -- FINST capacity cap (C4)
    continuity_radius: float = 2.0    # obf_continuity_radius -- motion gate, cells (C1)
    w_motion: float = 1.0             # obf_w_motion -- association cost weight
    w_feat: float = 1.0               # obf_w_feat -- association cost weight
    feature_alpha: float = 0.3        # obf_feature_alpha -- base feature EMA rate (C2/C5)
    persist_ttl: int = 8              # obf_persist_ttl -- ticks a token survives unseen (C3)
    min_birth_salience: float = 0.0   # obf_min_birth_salience -- attention floor to birth (C4)
    use_precision_weighting: bool = True  # obf_use_precision_weighting -- C5 on/off


def _as_vec(t: torch.Tensor) -> torch.Tensor:
    """Coerce [1, D] / [D] -> [D] (detached clone)."""
    t = t.detach()
    if t.dim() == 2:
        t = t.reshape(-1)
    return t.clone()


def _pos_vec(p: torch.Tensor) -> torch.Tensor:
    p = p.detach().reshape(-1).to(dtype=torch.float32)
    return p.clone()


def _cos_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """1 - cosine_similarity, in [0, 2]. Degenerate (zero-norm) -> 1.0 (neutral)."""
    na = float(a.norm().item())
    nb = float(b.norm().item())
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    cos = float(torch.dot(a, b).item()) / (na * nb)
    cos = max(-1.0, min(1.0, cos))
    return 1.0 - cos


class ObjectFileBuffer:
    """Stateful, non-trainable token-instance object-file buffer (MECH-045).

    One call to update() per waking tick. Pure dict/tensor state + the
    data-association rule; no learned parameters.
    """

    def __init__(self, config: Optional[ObjectFileBufferConfig] = None) -> None:
        self.config = config or ObjectFileBufferConfig()
        self._tokens: Dict[int, ObjectFile] = {}
        self._next_id: int = 0
        self._tick: int = 0
        # Diagnostics (reset per episode).
        self._n_births: int = 0
        self._n_matches: int = 0
        self._n_deaths: int = 0
        self._n_capacity_evictions: int = 0
        self._n_simulation_skipped: int = 0

    # -- lifecycle -------------------------------------------------------
    def reset(self) -> None:
        """Per-episode reset: clear tokens + counters. _next_id continues so
        token ids stay globally unique within a session (diagnostic clarity)."""
        self._tokens.clear()
        self._tick = 0
        self._n_births = 0
        self._n_matches = 0
        self._n_deaths = 0
        self._n_capacity_evictions = 0
        self._n_simulation_skipped = 0

    # -- core update -----------------------------------------------------
    def update(
        self,
        observations: List[EntityObservation],
        simulation_mode: bool = False,
    ) -> Dict[int, int]:
        """Advance the buffer one waking tick over the perceived entities.

        Returns a mapping {observation_index -> token_id} for every observation
        that was matched to an existing token or that birthed a surviving new
        token. MECH-094: a no-op returning {} under simulation_mode.
        """
        if simulation_mode:
            self._n_simulation_skipped += 1
            return {}

        self._tick += 1
        tick = self._tick
        cfg = self.config

        assignment: Dict[int, int] = {}
        if not observations:
            self._evict_stale(tick)
            return assignment

        # 1. PREDICT (zero-order): predicted_pos == last_pos for each live token.
        # 2. GATE + 3. MATCH: build the gated cost list and greedily assign.
        live_ids = list(self._tokens.keys())
        candidate_costs = []  # (cost, obs_idx, token_id)
        for oi, obs in enumerate(observations):
            o_pos = _pos_vec(obs.pos)
            o_z = _as_vec(obs.z)
            o_prec = float(obs.precision) if cfg.use_precision_weighting else 1.0
            o_prec = max(0.0, min(1.0, o_prec))
            for tid in live_ids:
                tok = self._tokens[tid]
                d_pos = float((o_pos - tok.last_pos).norm().item())
                # Hard motion gate (C1): continuity, not nearest-cell.
                if d_pos > cfg.continuity_radius:
                    continue
                d_feat = _cos_distance(tok.z_features, o_z)
                # Precision-weighted feature term (C5): a low-precision
                # observation contributes a softened feature distance, so an
                # uncertain match is not forced by appearance alone.
                feat_term = d_feat * (o_prec if cfg.use_precision_weighting else 1.0)
                cost = cfg.w_motion * d_pos + cfg.w_feat * feat_term
                candidate_costs.append((cost, oi, tid))

        candidate_costs.sort(key=lambda c: c[0])
        matched_obs: set = set()
        matched_tok: set = set()
        for cost, oi, tid in candidate_costs:
            if oi in matched_obs or tid in matched_tok:
                continue
            matched_obs.add(oi)
            matched_tok.add(tid)
            assignment[oi] = tid
            # 4. UPDATE (C2/C5): precision-weighted feature EMA + bookkeeping.
            obs = observations[oi]
            o_prec = float(obs.precision) if cfg.use_precision_weighting else 1.0
            o_prec = max(0.0, min(1.0, o_prec))
            alpha = cfg.feature_alpha * (o_prec if cfg.use_precision_weighting else 1.0)
            tok = self._tokens[tid]
            o_z = _as_vec(obs.z)
            tok.z_features = (1.0 - alpha) * tok.z_features + alpha * o_z
            tok.precision = (1.0 - cfg.feature_alpha) * tok.precision + cfg.feature_alpha * o_prec
            tok.last_pos = _pos_vec(obs.pos)
            tok.last_seen_tick = tick
            tok.hits += 1
            if obs.resource_tag is not None and int(obs.resource_tag) > 0:
                tok.type_hint = int(obs.resource_tag)
            self._n_matches += 1

        # 5. BIRTH (C4): unmatched observations with sufficient salience open a
        # new token. Low-salience unmatched observations are ignored.
        for oi, obs in enumerate(observations):
            if oi in matched_obs:
                continue
            if float(obs.salience) < cfg.min_birth_salience:
                continue
            tid = self._birth(obs, tick)
            assignment[oi] = tid

        # C4 capacity: cap simultaneously-maintained tokens at max_tokens.
        self._enforce_capacity(tick, assignment)

        # 6. DEATH (C3): evict tokens unseen for > persist_ttl ticks.
        self._evict_stale(tick)

        return assignment

    # -- helpers ---------------------------------------------------------
    def _birth(self, obs: EntityObservation, tick: int) -> int:
        tid = self._next_id
        self._next_id += 1
        o_prec = float(obs.precision)
        o_prec = max(0.0, min(1.0, o_prec)) if self.config.use_precision_weighting else 1.0
        self._tokens[tid] = ObjectFile(
            token_id=tid,
            z_features=_as_vec(obs.z),
            last_pos=_pos_vec(obs.pos),
            precision=o_prec,
            last_seen_tick=tick,
            hits=1,
            type_hint=(int(obs.resource_tag)
                       if obs.resource_tag is not None and int(obs.resource_tag) > 0
                       else None),
        )
        self._n_births += 1
        return tid

    def _enforce_capacity(self, tick: int, assignment: Dict[int, int]) -> None:
        """Evict lowest-precision / least-recently-seen tokens past the cap
        (C4 FINST capacity). Freshly-seen tokens (last_seen_tick == tick) sort
        last, so they are protected; the weakest stale tokens go first."""
        cap = max(1, int(self.config.max_tokens))
        if len(self._tokens) <= cap:
            return
        # Evict by ascending (last_seen_tick, precision): oldest + least
        # confident first.
        order = sorted(
            self._tokens.values(),
            key=lambda t: (t.last_seen_tick, t.precision),
        )
        n_evict = len(self._tokens) - cap
        evicted_ids = set()
        for tok in order[:n_evict]:
            evicted_ids.add(tok.token_id)
            del self._tokens[tok.token_id]
            self._n_capacity_evictions += 1
        if evicted_ids:
            # Drop any just-assigned obs whose token lost its slot.
            for oi in [k for k, v in assignment.items() if v in evicted_ids]:
                del assignment[oi]

    def _evict_stale(self, tick: int) -> None:
        ttl = int(self.config.persist_ttl)
        dead = [tid for tid, t in self._tokens.items()
                if (tick - t.last_seen_tick) > ttl]
        for tid in dead:
            del self._tokens[tid]
            self._n_deaths += 1

    # -- query / introspection ------------------------------------------
    def query(self, token_id: int) -> Optional[ObjectFile]:
        return self._tokens.get(int(token_id))

    def active_tokens(self) -> List[ObjectFile]:
        return list(self._tokens.values())

    def n_active(self) -> int:
        return len(self._tokens)

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "obf_n_active_tokens": len(self._tokens),
            "obf_n_births": self._n_births,
            "obf_n_matches": self._n_matches,
            "obf_n_deaths": self._n_deaths,
            "obf_n_capacity_evictions": self._n_capacity_evictions,
            "obf_n_simulation_skipped": self._n_simulation_skipped,
            "obf_tick": self._tick,
            "obf_next_id": self._next_id,
        }

    def get_state(self) -> Dict[str, Any]:
        return self.get_diagnostics()
