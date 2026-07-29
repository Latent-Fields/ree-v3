"""SD-083: offline policy-consolidation window for the mech457 retention testbed.

Unblocks the two `blocked_substrate` arms of MECH-476's retrograde-interference falsifier -- the
A->B offline INTERVAL arm (V3-EXQ-836b) and the Moncada-2007 behavioural-tagging NOVELTY arm
(V3-EXQ-836c). See REE_assembly/docs/architecture/sd_083_offline_policy_consolidation_window.md.

WHAT THIS IS, AND WHAT ITS SIBLING ALREADY DID. The retention lineage already has ONE
policy-protection mechanism, PolicyKLAnchor (MECH-475), in mech457_explorer_classes.py: an
ONLINE, GLOBAL, no-interval KL trust-region to a frozen post-install snapshot. This module is the
OFFLINE, TRACE-SELECTIVE, INTERVAL-ACCUMULATED, NOVELTY-GATED counterpart -- exactly the
sleep-dependent consolidation the MECH-476 literature review (Walker 2003) records REE as
LACKING. It is deliberately built in experiments/_lib/ alongside PolicyKLAnchor (the same locus,
the same testbed), NOT in ree_core/: it is a falsifier INSTRUMENT, not a cognifold faculty. The
cognifold port into the ONE SD-017 sleep loop is a registered follow-on gated on MECH-476 coming
back SUPPORTED.

THE WINDOW BUILDS PROTECTION; IT DOES NOT RETRAIN. The installed policy parameters theta are left
UNCHANGED across the window -- no optimiser step touches them -- so post_bc competence is
invariant to the interval. That invariance is the load-bearing property that makes the INTERVAL
axis ORTHOGONAL to the DOSE axis: dose moves theta* itself (a stronger install), while the
interval only builds the trace-selective anchor erected AROUND a fixed theta*. Biological
consolidation is protection of a trace, not re-acquisition of it.

THREE PARTS, mapped to synaptic-tagging-and-capture theory (Moncada 2007, Bin-Ibrahim 2024):

  1. THE TAG PATTERN -- a Fisher diagonal F over the policy parameters, estimated by offline
     replay of the demonstrator's on-expert states (SWS-analog). F_i is the importance of
     parameter i to the installed competence; this is the trace SELECTIVITY (iii) a global KL
     coefficient cannot express. Fisher QUALITY (offline_fisher_samples) is held FIXED across
     interval arms on purpose, so the interval effect is the capture clock below, NOT a
     Fisher-noise artifact.

  2. THE CAPTURE CLOCK -- a resource c(N) = capture_max * (1 - exp(-N/tau)) that integrates over
     the offline interval N = offline_window_steps. N=0 -> c=0 -> no anchor -> the unconstrained
     control. This is the elapsed-INTERVAL dependence (ii): the tag is set at induction and
     consolidated over an offline window.

  3. NOVELTY GATING -- Moncada behavioural tagging. The lineage's OWN RNDModule (RepAgent feature
     space; the MECH-441 novelty PRINCIPLE, correctly instantiated for this stack) measures the
     novelty of an exposure run inside the window; a weak ("sub-threshold") tag is captured only
     when a novelty episode is paired with it.

The effective EWC coefficient is coef = offline_ewc_max_coef * c(N) * novelty_factor, and the
returned OfflineEWCAnchor adds coef * sum_i F_i (theta_i - theta*_i)^2 to every policy update in
train_a2c.

RNG NEUTRALITY. consolidate_offline_window() snapshots and restores the torch / numpy / python
RNG streams around its entire body (same discipline as train_a2c's mid-training probe), so the
ONLY thing an N>0 window changes for the downstream RL refinement is the returned anchor -- never
the RNG stream. An N=0 (or zero-coefficient) window returns None and touches nothing, so it is
bit-identical to a no-window control. This is what lets the INTERVAL arm read as a clean
manipulation of protection strength alone.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734
from experiments._lib.mech457_explorer_classes import RNDModule


# --- The offline-derived, Fisher-weighted (EWC) anchor -------------------------------------
class OfflineEWCAnchor:
    """Parameter-space, Fisher-weighted trust-region anchor to the installed policy snapshot.

    penalty() returns coef * sum_i F_i * (theta_i - theta*_i)^2 over the LIVE policy parameters,
    graph-connected so train_a2c's loss.backward() flows the EWC gradient into them. Because F is
    estimated from the gradient of the policy LOG-LIKELIHOOD log pi(a|s), value-head-only
    parameters receive ~0 Fisher and are therefore left unconstrained -- the same locus discipline
    as PolicyKLAnchor (the shared trunk moves on both; the value-estimator locus stays with
    MECH-459 / the distributional critic).

    Fails LOUDLY on a mis-wire (empty params, length mismatch, non-positive coef) rather than
    producing an ON arm that is silently the control -- the degenerate-arm-read-as-a-verdict
    failure this family guards against everywhere.
    """

    def __init__(self, params: List[torch.nn.Parameter], fisher_diag: List[torch.Tensor],
                 coef: float) -> None:
        if not params:
            raise ValueError(
                "OfflineEWCAnchor: empty parameter list -- the offline window needs a live "
                "policy to anchor. A silently un-anchored ON arm is the control wearing the "
                "treatment label."
            )
        if len(params) != len(fisher_diag):
            raise ValueError(
                "OfflineEWCAnchor: fisher_diag length %d != params length %d; the Fisher "
                "diagonal must align element-wise with the trained parameters."
                % (len(fisher_diag), len(params))
            )
        if not (float(coef) > 0.0) or not math.isfinite(float(coef)):
            raise ValueError(
                "OfflineEWCAnchor: coef must be finite and > 0 (got %r); a zero-weight anchor "
                "is the OFF arm wearing the ON label. consolidate_offline_window returns None "
                "when the capture resource is zero rather than constructing a dead anchor."
                % (coef,)
            )
        self.coef = float(coef)
        self._params = list(params)
        # Frozen snapshot theta* (the tag pattern's reference point) and the per-parameter Fisher.
        self._theta_star = [p.detach().clone() for p in params]
        self._fisher = [f.detach().clone() for f in fisher_diag]
        self._penalty_recent: List[float] = []
        self.fisher_mass = float(sum(float(f.sum().item()) for f in self._fisher))

    def penalty(self) -> torch.Tensor:
        """coef * sum_i F_i * (theta_i - theta*_i)^2 over the live params (grad-connected)."""
        total = torch.zeros((), device=self._params[0].device)
        for p, star, f in zip(self._params, self._theta_star, self._fisher):
            total = total + (f * (p - star) ** 2).sum()
        pen = self.coef * total
        self._penalty_recent.append(float(pen.item()))
        if len(self._penalty_recent) > 200:
            self._penalty_recent = self._penalty_recent[-200:]
        return pen

    def mean_realised_penalty(self) -> float:
        return (
            float(sum(self._penalty_recent) / len(self._penalty_recent))
            if self._penalty_recent else 0.0
        )


# --- The window ----------------------------------------------------------------------------
def _collect_expert_states(rep: Any, demo: Any, env_kwargs: Dict[str, Any], seed: int, *,
                           steps: int, n_states: int, max_episodes: int = 64) -> List[torch.Tensor]:
    """Roll the demonstrator to generate on-expert states; return their detached policy inputs.

    No gradient, no optimiser step, no policy change -- this only SAMPLES the state distribution
    the installed competence lives on (the SWS-analog replay the tag pattern is estimated over).
    Builds a fresh env per episode (never touches the training env).
    """
    zs: List[torch.Tensor] = []
    ep = 0
    while len(zs) < int(n_states) and ep < int(max_episodes):
        env = x734._make_env(seed + 1009 + ep, env_kwargs)
        _flat, obs_dict = env.reset()
        rep.reset_episode()
        for _step in range(int(steps)):
            a_expert = int(demo.act(env, obs_dict))
            state = rep.encode(obs_dict)
            zs.append(rep.z_detached(state).detach().reshape(1, -1))
            _flat, _harm, done, _info, obs_dict = env.step(a_expert)
            if len(zs) >= int(n_states) or done:
                break
        ep += 1
    return zs[: int(n_states)]


def _estimate_fisher_diag(rep: Any, zs: List[torch.Tensor]) -> List[torch.Tensor]:
    """Diagonal Fisher of the policy log-likelihood over the replayed states.

    For each state z: forward the policy, take the MAP action a* = argmax logits (the mode carries
    almost all mass on a BC-cloned greedy policy, so this deterministic empirical-Fisher-at-the-
    mode estimate avoids consuming RNG), backprop log pi(a*|z), and accumulate grad^2 per
    parameter. F_i = mean_z (d log pi(a*|z) / d theta_i)^2. Value-head-only params get exactly 0
    (log pi does not depend on them).
    """
    params = rep.params()
    policy = rep.policy()
    fisher = [torch.zeros_like(p) for p in params]
    n = 0
    for z in zs:
        out = policy(z)
        logits = out[0].reshape(-1)
        logp = torch.log_softmax(logits, dim=-1)
        a_star = int(torch.argmax(logits).item())
        for p in params:
            if p.grad is not None:
                p.grad = None
        logp[a_star].backward()
        for i, p in enumerate(params):
            if p.grad is not None:
                fisher[i] = fisher[i] + p.grad.detach() ** 2
        n += 1
    if n > 0:
        fisher = [f / float(n) for f in fisher]
    # Leave params clean for the caller (no residual grads carried into RL).
    for p in params:
        if p.grad is not None:
            p.grad = None
    return fisher


def _measure_novelty(rep: Any, env_kwargs: Dict[str, Any], seed: int, *,
                     steps: int, exposure_episodes: int) -> float:
    """Mean normalized RND predictor-error over a fresh-region exposure (the novelty drive).

    Uses the lineage's OWN RNDModule over the RepAgent feature space (the MECH-441 novelty
    principle for this stack). A fresh env the predictor has never seen yields high error =
    high novelty; this is the capture signal for the paired condition and the guard against a
    mislabeled (non-novel) exposure.
    """
    rnd = RNDModule(int(rep.feature_dim))
    scores: List[float] = []
    for e in range(int(exposure_episodes)):
        env = x734._make_env(seed + 4801 + e, env_kwargs)
        _flat, obs_dict = env.reset()
        rep.reset_episode()
        state = rep.encode(obs_dict)
        z_prev = rep.z_detached(state)
        for _step in range(int(steps)):
            step = rep.step(state, deterministic=True)
            a_idx = int(step.action.reshape(-1)[0].item())
            _flat, _harm, done, _info, obs_dict = env.step(a_idx)
            next_state = rep.encode(obs_dict)
            z_next = rep.z_detached(next_state)
            scores.append(float(rnd.intrinsic_reward(z_prev, a_idx, z_next)))
            z_prev = z_next
            state = next_state
            if done:
                break
        rnd.update()
    return float(sum(scores) / len(scores)) if scores else 0.0


def capture_resource(n_steps: int, tau: float, capture_max: float) -> float:
    """c(N) = capture_max * (1 - exp(-N/tau)). The tag-and-capture systems-consolidation clock."""
    if int(n_steps) <= 0 or float(tau) <= 0.0:
        return 0.0
    return float(capture_max) * (1.0 - math.exp(-float(n_steps) / float(tau)))


def consolidate_offline_window(
    rep: Any, demo: Any, env_kwargs: Dict[str, Any], seed: int, *,
    steps: int,
    offline_window_steps: int,
    offline_ewc_max_coef: float,
    offline_capture_tau: float = 200.0,
    offline_capture_max: float = 1.0,
    offline_fisher_samples: int = 256,
    offline_novelty_pairing: str = "none",
    novelty_ref: float = 1.0,
    tag_leak: float = 0.05,
    novelty_exposure_episodes: int = 8,
) -> Dict[str, Any]:
    """Run the offline policy-consolidation window on an INSTALLED policy.

    Returns a dict with `anchor` (an OfflineEWCAnchor or None) plus diagnostics. Returns
    anchor=None when the capture resource or coefficient is zero (N=0, coef=0, or a fully-leaked
    unpaired tag), which is the unconstrained control -- and in that case does NOT touch the RNG
    stream, so the cell is bit-identical to a no-window run.

    novelty_pairing:
      "none"     -- novelty_factor = 1.0 (INTERVAL arm; novelty is not the manipulation).
      "paired"   -- measure novelty over a fresh exposure; novelty_factor = clamp(nov/ref, leak, 1).
      "unpaired" -- novelty_factor = tag_leak (the weak tag is not captured; NOVELTY-arm null).
    """
    pairing = str(offline_novelty_pairing)
    if pairing not in ("none", "paired", "unpaired"):
        raise ValueError(
            "consolidate_offline_window: offline_novelty_pairing must be one of "
            "none/paired/unpaired (got %r)." % (offline_novelty_pairing,)
        )
    diag: Dict[str, Any] = {
        "anchor": None,
        "offline_window_steps": int(offline_window_steps),
        "offline_novelty_pairing": pairing,
        "capture_resource": 0.0,
        "novelty_factor": 0.0,
        "measured_novelty": None,
        "effective_ewc_coef": 0.0,
        "fisher_mass": 0.0,
        "n_fisher_states": 0,
    }

    capture = capture_resource(offline_window_steps, offline_capture_tau, offline_capture_max)
    diag["capture_resource"] = round(capture, 6)
    # Cheap early-out that costs no RNG: no interval or no budget -> the unconstrained control.
    if capture <= 0.0 or float(offline_ewc_max_coef) <= 0.0:
        return diag

    # Everything below may consume RNG (policy forward on collected states, novelty rollouts);
    # snapshot and restore all three streams so the window changes ONLY the returned anchor.
    _torch_rng = torch.get_rng_state()
    _np_rng = np.random.get_state()
    _py_rng = random.getstate()
    try:
        # Novelty gate (Moncada behavioural tagging).
        if pairing == "none":
            novelty_factor = 1.0
        elif pairing == "unpaired":
            novelty_factor = float(tag_leak)
        else:  # paired
            measured = _measure_novelty(
                rep, env_kwargs, seed, steps=int(steps),
                exposure_episodes=int(novelty_exposure_episodes),
            )
            diag["measured_novelty"] = round(measured, 6)
            novelty_factor = float(
                min(1.0, max(float(tag_leak), measured / float(novelty_ref)))
            )
        diag["novelty_factor"] = round(novelty_factor, 6)

        eff_coef = float(offline_ewc_max_coef) * capture * novelty_factor
        diag["effective_ewc_coef"] = round(eff_coef, 6)
        if not (eff_coef > 0.0):
            return diag  # fully-leaked tag -> no anchor -> control

        # Tag pattern: estimate the Fisher diagonal over replayed on-expert states.
        zs = _collect_expert_states(
            rep, demo, env_kwargs, seed, steps=int(steps), n_states=int(offline_fisher_samples)
        )
        diag["n_fisher_states"] = int(len(zs))
        if not zs:
            return diag
        fisher = _estimate_fisher_diag(rep, zs)
        anchor = OfflineEWCAnchor(rep.params(), fisher, eff_coef)
        diag["anchor"] = anchor
        diag["fisher_mass"] = round(anchor.fisher_mass, 6)
        return diag
    finally:
        torch.set_rng_state(_torch_rng)
        np.random.set_state(_np_rng)
        random.setstate(_py_rng)
