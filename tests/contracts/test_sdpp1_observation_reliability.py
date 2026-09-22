"""Contract tests for SD-PP-1 observation reliability estimator (2026-09-22 build).

Contract: REE_assembly/docs/architecture/precision_provenance_substrate_spec.md
section 2. Six tests: (1) config default False; (2) sigma recovery (clean at
sigma_floor, noisy within 25% after 50 frames); (3) kappa positive and finite
after >= 2 latent frames; (4) no RNG consumption; (5) on_episode_reset makes
the next observe a no-op; (6) evidence_precision_z falls monotonically as
noise s rises.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from ree_core.precision.observation_reliability import (
    ObservationReliabilityConfig,
    ObservationReliabilityEstimator,
)

WORLD_DIM = 250
N_CHANGE = 30  # ~12% of 250 elements flip between frames


def _make_base_and_changing_indices(seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    base = torch.zeros(WORLD_DIM)
    # a handful of "on" one-hot-ish entries, mostly-static field
    on_idx = torch.randperm(WORLD_DIM, generator=gen)[: WORLD_DIM // 5]
    base[on_idx] = 1.0
    change_idx = torch.randperm(WORLD_DIM, generator=gen)[:N_CHANGE]
    return base, change_idx, gen


def _make_frame_stream(n_frames: int, sigma: float, seed: int = 0):
    """Mostly-static 250-vector stream: ~12% of elements flip per frame, plus
    optional additive N(0, sigma^2) noise from a TEST-owned RNG (never the
    estimator's)."""
    base, change_idx, gen = _make_base_and_changing_indices(seed)
    frames = []
    cur = base.clone()
    for t in range(n_frames):
        nxt = cur.clone()
        # flip the designated "changing" indices this frame (toggle 0/1)
        nxt[change_idx] = 1.0 - nxt[change_idx]
        if sigma > 0.0:
            noise = torch.randn(WORLD_DIM, generator=gen) * sigma
            nxt = nxt + noise
        frames.append(nxt)
        cur = nxt
    return frames


def test_config_default_false():
    cfg = ObservationReliabilityConfig()
    assert cfg.use_observation_reliability is False


def test_sigma_recovery_clean_and_noisy():
    # clean stream (sigma=0) -> sigma_obs sits AT sigma_floor exactly
    cfg_clean = ObservationReliabilityConfig()
    est_clean = ObservationReliabilityEstimator(cfg_clean)
    for frame in _make_frame_stream(n_frames=50, sigma=0.0, seed=0):
        est_clean.observe_obs(frame)
    assert est_clean.sigma_obs == cfg_clean.sigma_floor

    # noisy streams -> recovered within 25% after 50 frames
    for s in (0.03, 0.12):
        cfg = ObservationReliabilityConfig()
        est = ObservationReliabilityEstimator(cfg)
        for frame in _make_frame_stream(n_frames=50, sigma=s, seed=0):
            est.observe_obs(frame)
        rel_err = abs(est.sigma_obs - s) / s
        assert rel_err < 0.25, f"sigma={s}: recovered {est.sigma_obs}, rel_err={rel_err}"


def test_kappa_positive_finite_after_two_latent_frames():
    cfg = ObservationReliabilityConfig()
    est = ObservationReliabilityEstimator(cfg)
    obs_frames = _make_frame_stream(n_frames=5, sigma=0.03, seed=1)
    gen = torch.Generator().manual_seed(7)
    z_dim = 16
    z_prev = torch.randn(z_dim, generator=gen)
    for i, obs in enumerate(obs_frames):
        est.observe_obs(obs)
        z = torch.randn(z_dim, generator=gen) * 0.1 + z_prev
        est.observe_latent(z)
        z_prev = z
    assert est.ready is True
    assert est.kappa > 0.0
    assert est.kappa == est.kappa  # not NaN
    assert est.kappa != float("inf")


def test_no_rng_consumption():
    cfg = ObservationReliabilityConfig()
    est = ObservationReliabilityEstimator(cfg)

    gen = torch.Generator().manual_seed(0)
    obs_frames = [torch.randn(WORLD_DIM, generator=gen) for _ in range(101)]
    z_frames = [torch.randn(16, generator=gen) for _ in range(101)]

    state_before = torch.get_rng_state()
    for i in range(100):
        est.observe_obs(obs_frames[i])
        est.observe_latent(z_frames[i])
    state_after = torch.get_rng_state()

    assert torch.equal(state_before, state_after)


def test_episode_reset_makes_next_observe_a_noop():
    cfg = ObservationReliabilityConfig()
    est = ObservationReliabilityEstimator(cfg)
    frames = _make_frame_stream(n_frames=5, sigma=0.05, seed=2)
    for frame in frames:
        est.observe_obs(frame)
    n_frames_before = est.n_frames
    sigma_before = est.sigma_obs_sq

    est.on_episode_reset()
    # first observe_obs after reset only stores the frame -- no statistic update
    est.observe_obs(frames[0])
    assert est.n_frames == n_frames_before
    assert est.sigma_obs_sq == sigma_before

    # and observe_latent with no cached obs-diff (post-reset, only one obs seen) is a no-op
    ready_before = est.ready
    kappa_before = est.kappa
    est.observe_latent(torch.zeros(16))
    assert est.ready == ready_before
    assert est.kappa == kappa_before


def test_evidence_precision_z_monotonic_in_noise():
    z_dim = 16
    vals = []
    for s in (0.0, 0.03, 0.12):
        cfg = ObservationReliabilityConfig()
        est = ObservationReliabilityEstimator(cfg)
        obs_frames = _make_frame_stream(n_frames=30, sigma=s, seed=3)
        gen = torch.Generator().manual_seed(11)
        z_prev = torch.randn(z_dim, generator=gen)
        for obs in obs_frames:
            est.observe_obs(obs)
            z = torch.randn(z_dim, generator=gen) * 0.1 + z_prev
            est.observe_latent(z)
            z_prev = z
        vals.append(est.evidence_precision_z)

    assert vals[0] > vals[1] > vals[2], vals


def test_snapshot_and_get_metrics_keys():
    cfg = ObservationReliabilityConfig()
    est = ObservationReliabilityEstimator(cfg)
    for frame in _make_frame_stream(n_frames=5, sigma=0.03, seed=4):
        est.observe_obs(frame)

    snap = est.snapshot()
    expected_keys = {
        "sigma_obs",
        "sigma_obs_sq",
        "precision_obs",
        "kappa",
        "evidence_variance_z",
        "evidence_precision_z",
        "ready",
    }
    assert set(snap.keys()) == expected_keys

    metrics = est.get_metrics()
    expected_metric_keys = {f"obs_reliability_{k}" for k in expected_keys}
    expected_metric_keys.add("obs_reliability_n_frames")
    assert set(metrics.keys()) == expected_metric_keys


def test_t8_get_set_state_round_trip_restores_reads_and_drops_frame_cache():
    """Pre-freeze repair (V3-EXQ-1073): a calibration window's EMA state is
    portable to a fresh estimator; the frame cache is not carried."""
    import torch
    from ree_core.precision.observation_reliability import (
        ObservationReliabilityConfig, ObservationReliabilityEstimator)
    cfg = ObservationReliabilityConfig(use_observation_reliability=True)
    a = ObservationReliabilityEstimator(cfg)
    g = torch.Generator().manual_seed(0)
    base = torch.zeros(250); base[:30] = 1.0
    for _ in range(40):
        o = base + torch.randn(250, generator=g) * 0.05
        a.observe_obs(o)
        a.observe_latent(torch.randn(16, generator=g) * 0.01)
    st = a.get_state()
    b = ObservationReliabilityEstimator(cfg)
    b.set_state(st)
    assert b.snapshot() == a.snapshot()
    assert b.ready == a.ready and b.n_frames == a.n_frames
    # frame cache dropped: the next observe is a first-frame no-op
    n0 = b.n_frames
    b.observe_obs(base)
    assert b.n_frames == n0
