"""Contract tests for SD-PP-3 `hippocampal.replay_provenance` (2026-09-22 build).

See `REE_assembly/docs/architecture/precision_provenance_substrate_spec.md`
section 4. SD-PP-1 (`ree_core/precision/observation_reliability.py`) and
SD-PP-2 (`ree_core/precision/world_forward_epistemic_precision.py`) are the
two PRODUCER modules this carrier reads from; they are built CONCURRENTLY by
other work and may not exist on disk yet, so every test here except the
final importorskip'd integration test uses small stub stand-ins that
implement the duck-typed protocol the recorder actually calls:

    epistemic.prediction_at_test(e2, z_prev, a_onehot) -> Tensor[1, D]
    epistemic.precision_at(z_prev, a_onehot, head=head) -> object with
        .pi_epi .v_tot .v_ale .v_epi .source
    epistemic.observe_outcome(pred, z_now, evidence_variance_z) -> float
    reliability.snapshot() -> Dict[str, float]

CONTRACTS
  1  alignment pin -- against a REAL E2FastPredictor (agent.e2.world_forward),
     p[i+1].pred_at_test == e2.world_forward(world[i], action[i+1]) for every i.
  2  trim lockstep -- record 1005, len == 1000, buffer_index continuity
     through the trim (get(i) still names the right original call).
  3  no-future-info -- pi_hist reflects the epistemic state BEFORE the
     outcome mutates it (observe_outcome runs strictly after precision_at).
  4  surprise formula -- pi_hist * max(pe - noise_gain*evidence_variance_z, 0).
  5  episode-start placeholder -- z_prev=None skips every epistemic call.
  6  schema_version stamped on every packet.
  +  integration (importorskip) -- real SD-PP-1/SD-PP-2 producers end to end.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.hippocampal.replay_provenance import (
    PROVENANCE_SCHEMA_VERSION,
    ReplayProvenancePacket,
    ReplayProvenanceRecorder,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _build_agent(seed: int = 7, **flags):
    """Same shape as test_e2_world_forward_sleep_trainer.py::_build."""
    from ree_core.agent import REEAgent
    from ree_core.environment.causal_grid_world import CausalGridWorldV2
    from ree_core.utils.config import REEConfig

    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        seed=seed, size=5, num_hazards=1, num_resources=1, use_proxy_fields=True
    )
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        self_dim=16,
        world_dim=16,
        **flags,
    )
    agent = REEAgent(cfg)
    agent.reset()
    _flat, od = env.reset()
    b = od["body_state"]
    w = od["world_state"]
    if b.dim() == 1:
        b = b.unsqueeze(0)
    if w.dim() == 1:
        w = w.unsqueeze(0)
    return agent, b, w


def _onehot(dim: int, idx: int) -> torch.Tensor:
    t = torch.zeros(1, dim)
    t[0, idx] = 1.0
    return t


class _StubRead:
    def __init__(self, pi_epi, v_tot, v_ale, v_epi, source):
        self.pi_epi = pi_epi
        self.v_tot = v_tot
        self.v_ale = v_ale
        self.v_epi = v_epi
        self.source = source


class _DummyE2:
    """Minimal e2 stand-in when the real predictor is not needed."""

    def world_forward(self, z_world: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return z_world + action.sum(dim=-1, keepdim=True)


class StubEpistemic:
    """Duck-typed stand-in for WorldForwardEpistemicPrecision (SD-PP-2).

    Enforces the ordering contract itself: `observe_outcome` raises if it is
    called before a matching `precision_at` for the same transition (tracked
    by a pending-read counter), and `pi_epi` visibly mutates AFTER each
    `observe_outcome` so a test can distinguish a pre-outcome read from a
    post-outcome one.
    """

    def __init__(self, pi_epi=2.0, v_tot=0.5, v_ale=0.1, source="ema"):
        self.pi_epi = pi_epi
        self.v_tot = v_tot
        self.v_ale = v_ale
        self.source = source
        self.calls = []
        self._pending_reads = 0

    def prediction_at_test(self, e2, z_prev, a_onehot):
        self.calls.append("prediction_at_test")
        with torch.no_grad():
            return e2.world_forward(z_prev[:1], a_onehot[:1]).detach().clone()

    def precision_at(self, z_prev, a_onehot, head=None):
        self.calls.append("precision_at")
        self._pending_reads += 1
        v_epi = max(self.v_tot - self.v_ale, 1e-6)
        return _StubRead(
            pi_epi=self.pi_epi, v_tot=self.v_tot, v_ale=self.v_ale,
            v_epi=v_epi, source=self.source,
        )

    def observe_outcome(self, pred, z_now, evidence_variance_z):
        if self._pending_reads == 0:
            raise AssertionError(
                "observe_outcome called before precision_at for this "
                "transition -- ordering contract violated"
            )
        self._pending_reads -= 1
        self.calls.append("observe_outcome")
        pe = float(torch.mean((pred - z_now) ** 2).item())
        # mutate AFTER computing pe, so a read captured before this call
        # differs from one captured after it.
        self.pi_epi = self.pi_epi + 1.0
        return pe


class StubReliability:
    def __init__(
        self,
        sigma_obs=0.1,
        kappa=1.0,
        evidence_variance_z=0.02,
        evidence_precision_z=50.0,
        ready=True,
    ):
        self._snap = {
            "sigma_obs": sigma_obs,
            "sigma_obs_sq": sigma_obs * sigma_obs,
            "precision_obs": 1.0 / (sigma_obs * sigma_obs),
            "kappa": kappa,
            "evidence_variance_z": evidence_variance_z,
            "evidence_precision_z": evidence_precision_z,
            "ready": ready,
        }

    def snapshot(self):
        return dict(self._snap)


# ----------------------------------------------------------------------
# 1 -- alignment pin against a real E2FastPredictor
# ----------------------------------------------------------------------
def test_alignment_pin_against_real_e2():
    agent, _b, _w = _build_agent(seed=3)
    world_dim = int(agent.config.latent.world_dim)
    action_dim = 4
    n = 6

    torch.manual_seed(99)
    world = [torch.randn(1, world_dim) for _ in range(n)]
    action = [_onehot(action_dim, i % action_dim) for i in range(n)]

    epistemic = StubEpistemic()
    recorder = ReplayProvenanceRecorder(
        epistemic=epistemic, reliability=None, noise_gain=2.0, max_len=1000
    )

    packets_by_index = {}
    for j in range(1, n):
        pkt = recorder.record(
            agent.e2, world[j - 1], action[j], world[j], buffer_index=j, tick=j
        )
        packets_by_index[j] = pkt

    with torch.no_grad():
        for i in range(n - 1):
            expected = agent.e2.world_forward(world[i], action[i + 1]).detach()
            assert torch.equal(packets_by_index[i + 1].pred_at_test, expected), (
                f"packet p[{i + 1}] misaligned against e2.world_forward(world[{i}], "
                f"action[{i + 1}])"
            )


# ----------------------------------------------------------------------
# 2 -- trim lockstep
# ----------------------------------------------------------------------
def test_trim_lockstep_preserves_buffer_index_alignment():
    epistemic = StubEpistemic()
    reliability = StubReliability()
    recorder = ReplayProvenanceRecorder(
        epistemic=epistemic, reliability=reliability, noise_gain=2.0, max_len=1000
    )

    world_dim = 4
    action_dim = 3
    n_total = 1005

    z_prev = None
    for i in range(n_total):
        z_now = torch.full((1, world_dim), float(i))
        a = _onehot(action_dim, i % action_dim)
        recorder.record(_DummyE2(), z_prev, a, z_now, buffer_index=i, tick=i)
        z_prev = z_now

    assert len(recorder.packets) == 1000

    offset = n_total - 1000
    for i in range(1000):
        pkt = recorder.get(i)
        assert pkt is not None
        assert pkt.buffer_index == offset + i
        assert pkt.tick == offset + i

    assert recorder.get(1000) is None
    assert recorder.get(-1).buffer_index == n_total - 1


# ----------------------------------------------------------------------
# 3 -- no-future-info: pi_hist reflects the PRE-outcome epistemic state
# ----------------------------------------------------------------------
def test_no_future_info_pi_hist_is_pre_outcome():
    epistemic = StubEpistemic(pi_epi=2.0)
    reliability = StubReliability()
    recorder = ReplayProvenanceRecorder(
        epistemic=epistemic, reliability=reliability, noise_gain=2.0
    )

    z_prev = torch.randn(1, 4)
    a = _onehot(3, 0)
    z_now = torch.randn(1, 4)

    pi_before = epistemic.pi_epi
    pkt = recorder.record(_DummyE2(), z_prev, a, z_now, buffer_index=0, tick=0)

    assert pkt.pi_hist == pi_before
    assert pkt.v_tot_hist == epistemic.v_tot
    assert pkt.v_ale_hist == epistemic.v_ale
    assert pkt.precision_source == epistemic.source
    # observe_outcome mutated the epistemic state AFTER the read the packet
    # actually carries -- the packet must not have picked up the new value.
    assert epistemic.pi_epi != pi_before
    assert pkt.pi_hist != epistemic.pi_epi

    # ordering was exercised, not merely asserted: precision_at strictly
    # before observe_outcome, and prediction_at_test before both.
    assert epistemic.calls == ["prediction_at_test", "precision_at", "observe_outcome"]


# ----------------------------------------------------------------------
# 4 -- surprise formula
# ----------------------------------------------------------------------
def test_surprise_formula():
    z_prev = torch.zeros(1, 4)
    a = _onehot(4, 0)  # _DummyE2 offset = a.sum() = 1.0 -> pred = [1,1,1,1]
    pred_exact = torch.ones(1, 4)

    # Case A: pe == 0 (z_now == pred) -> surprise clamps to 0 even though
    # evidence_variance_z is nonzero.
    reliability_a = StubReliability(evidence_variance_z=5.0)
    recorder_a = ReplayProvenanceRecorder(
        epistemic=StubEpistemic(), reliability=reliability_a, noise_gain=2.0
    )
    pkt_a = recorder_a.record(
        _DummyE2(), z_prev, a, pred_exact.clone(), buffer_index=0, tick=0
    )
    assert pkt_a.pe == pytest.approx(0.0, abs=1e-12)
    assert pkt_a.surprise == pytest.approx(0.0, abs=1e-12)

    # Case B: no reliability producer (evidence_variance_z == 0) -> surprise
    # == pi_hist * pe exactly.
    z_now_b = torch.full((1, 4), 5.0)
    epistemic_b = StubEpistemic(pi_epi=3.0)
    recorder_b = ReplayProvenanceRecorder(
        epistemic=epistemic_b, reliability=None, noise_gain=2.0
    )
    pkt_b = recorder_b.record(_DummyE2(), z_prev, a, z_now_b, buffer_index=0, tick=0)
    expected_pe = float(torch.mean((pred_exact - z_now_b) ** 2).item())
    assert pkt_b.pe == pytest.approx(expected_pe)
    assert pkt_b.evidence_variance_z == 0.0
    assert pkt_b.surprise == pytest.approx(pkt_b.pi_hist * expected_pe)


# ----------------------------------------------------------------------
# 5 -- episode-start placeholder
# ----------------------------------------------------------------------
def test_episode_start_placeholder_skips_epistemic_calls():
    epistemic = StubEpistemic()
    reliability = StubReliability(evidence_variance_z=0.03, ready=True)
    recorder = ReplayProvenanceRecorder(
        epistemic=epistemic, reliability=reliability, noise_gain=2.0
    )

    z_now = torch.randn(1, 6)
    a = _onehot(3, 1)
    pkt = recorder.record(_DummyE2(), None, a, z_now, buffer_index=0, tick=0)

    assert pkt.has_prev is False
    assert math.isnan(pkt.pe)
    assert math.isnan(pkt.pi_hist)
    assert math.isnan(pkt.surprise)
    assert pkt.pred_at_test.shape == (1, 6)
    assert torch.equal(pkt.pred_at_test, torch.zeros(1, 6))
    assert pkt.schema_version == PROVENANCE_SCHEMA_VERSION
    assert epistemic.calls == []  # no epistemic method called at all

    # evidence fields still come from reliability.snapshot()
    snap = reliability.snapshot()
    assert pkt.evidence_variance_z == snap["evidence_variance_z"]
    assert pkt.evidence_ready == snap["ready"]

    # and when reliability is entirely absent, the placeholder uses zeros/False
    recorder2 = ReplayProvenanceRecorder(
        epistemic=StubEpistemic(), reliability=None, noise_gain=2.0
    )
    pkt2 = recorder2.record(_DummyE2(), None, a, z_now, buffer_index=1, tick=1)
    assert pkt2.evidence_variance_z == 0.0
    assert pkt2.evidence_precision_z == 0.0
    assert pkt2.sigma_obs == 0.0
    assert pkt2.kappa == 0.0
    assert pkt2.evidence_ready is False


# ----------------------------------------------------------------------
# 6 -- schema_version stamped
# ----------------------------------------------------------------------
def test_schema_version_stamped():
    assert PROVENANCE_SCHEMA_VERSION == 1

    epistemic = StubEpistemic()
    reliability = StubReliability()
    recorder = ReplayProvenanceRecorder(
        epistemic=epistemic, reliability=reliability, noise_gain=2.0
    )

    z_prev = torch.randn(1, 4)
    a = _onehot(3, 0)
    z_now = torch.randn(1, 4)

    pkt_real = recorder.record(_DummyE2(), z_prev, a, z_now, buffer_index=0, tick=0)
    pkt_placeholder = recorder.record(
        _DummyE2(), None, a, z_now, buffer_index=1, tick=1
    )

    assert pkt_real.schema_version == PROVENANCE_SCHEMA_VERSION
    assert pkt_placeholder.schema_version == PROVENANCE_SCHEMA_VERSION
    assert isinstance(pkt_real, ReplayProvenancePacket)


# ----------------------------------------------------------------------
# stats() / get_metrics() -- basic shape sanity (not in the six pinned
# contracts above, but load-bearing for the SD-PP-4 consumer that reads them)
# ----------------------------------------------------------------------
def test_stats_and_get_metrics_shape():
    epistemic = StubEpistemic()
    reliability = StubReliability()
    recorder = ReplayProvenanceRecorder(
        epistemic=epistemic, reliability=reliability, noise_gain=2.0
    )

    # empty recorder: has_prev-conditioned stats are nan, n counters are 0
    empty_stats = recorder.stats()
    assert empty_stats["n"] == 0.0
    assert empty_stats["n_has_prev"] == 0.0
    assert math.isnan(empty_stats["frac_has_prev"])
    assert math.isnan(empty_stats["pe_mean"])

    a = _onehot(3, 0)
    z_prev = torch.randn(1, 4)
    for i in range(3):
        z_now = torch.randn(1, 4)
        recorder.record(_DummyE2(), z_prev, a, z_now, buffer_index=i, tick=i)
        z_prev = z_now
    recorder.record(_DummyE2(), None, a, z_prev, buffer_index=3, tick=3)

    stats = recorder.stats()
    assert stats["n"] == 4.0
    assert stats["n_has_prev"] == 3.0
    assert stats["frac_has_prev"] == pytest.approx(0.75)
    assert stats["pe_min"] <= stats["pe_mean"] <= stats["pe_max"]

    metrics = recorder.get_metrics()
    assert metrics["replay_provenance_n"] == 4.0
    assert metrics["replay_provenance_n_has_prev"] == 3.0
    assert set(metrics.keys()) == {f"replay_provenance_{k}" for k in stats.keys()}


def test_trim_to_and_get_out_of_range():
    epistemic = StubEpistemic()
    reliability = StubReliability()
    recorder = ReplayProvenanceRecorder(
        epistemic=epistemic, reliability=reliability, noise_gain=2.0
    )

    a = _onehot(3, 0)
    z_prev = None
    for i in range(5):
        z_now = torch.full((1, 4), float(i))
        recorder.record(_DummyE2(), z_prev, a, z_now, buffer_index=i, tick=i)
        z_prev = z_now

    assert recorder.get(5) is None
    assert len(recorder.packets) == 5

    recorder.trim_to(2)
    assert len(recorder.packets) == 2
    assert recorder.packets[0].buffer_index == 3
    assert recorder.packets[1].buffer_index == 4

    recorder.trim_to(0)
    assert len(recorder.packets) == 0
    assert recorder.get(0) is None


# ----------------------------------------------------------------------
# Integration (skips cleanly if SD-PP-1/SD-PP-2 are not yet on disk --
# they are being built concurrently by other sessions)
# ----------------------------------------------------------------------
def test_integration_real_producers_end_to_end():
    pytest.importorskip("ree_core.precision.observation_reliability")
    pytest.importorskip("ree_core.precision.world_forward_epistemic_precision")

    from ree_core.precision.observation_reliability import (
        ObservationReliabilityConfig,
        ObservationReliabilityEstimator,
    )
    from ree_core.precision.world_forward_epistemic_precision import (
        WorldForwardEpistemicPrecision,
        WorldForwardEpistemicPrecisionConfig,
    )

    agent, _b, _w = _build_agent(seed=5)
    world_dim = int(agent.config.latent.world_dim)

    reliability = ObservationReliabilityEstimator(
        ObservationReliabilityConfig(use_observation_reliability=True)
    )
    epi_cfg = WorldForwardEpistemicPrecisionConfig(
        use_world_forward_epistemic_precision=True
    )
    epistemic = WorldForwardEpistemicPrecision(epi_cfg, world_dim=world_dim)

    recorder = ReplayProvenanceRecorder(
        epistemic=epistemic,
        reliability=reliability,
        noise_gain=epi_cfg.noise_gain,
        max_len=1000,
    )

    torch.manual_seed(21)
    z_prev = None
    for i in range(5):
        z_now = torch.randn(1, world_dim)
        a = _onehot(4, i % 4)
        pkt = recorder.record(agent.e2, z_prev, a, z_now, buffer_index=i, tick=i)
        assert pkt.schema_version == PROVENANCE_SCHEMA_VERSION
        z_prev = z_now

    assert len(recorder.packets) == 5
    stats = recorder.stats()
    assert stats["n"] == 5.0
    assert stats["n_has_prev"] == 4.0
