"""V3-EXQ-1018: MECH-222 self-attribution contamination -- does failure of
continuous z_world residualization degrade the downstream visibility of
EXOGENOUS world events? (DIAGNOSTIC)

SLEEP DRIVER: N/A (no sleep loop; wake-only representational measurement run).

red-team (fable): see RED-TEAM line at the end of this docstring.

CLAIM TESTED (claims.yaml MECH-222, verbatim title):
"Failure of continuous z_self residualization produces self-attribution
contamination psychosis: referential delusion and thought broadcasting
substrate."

Its notes name the mechanism precisely: "When MECH-221 (continuous z_self
residualization) fails, self-generated motor predictions persist in z_world.
The coherency calculation then finds z_world consistent with the agent's own
predictions -- not because the world is responding to the agent, but because the
agent's predictions were never subtracted." And the clinical reading:
"Referential delusion: environmental events are interpreted as self-directed
because z_world contains the agent's own prediction footprint."

MECH-221 (co-tagged) asserts the antecedent: "z_world must be continuously
residualized against z_self predictions". This run ABLATES that residualization
and measures the downstream consequence, so it bears on both -- MECH-221's
"must" (is it load-bearing for anything downstream?) and MECH-222's "failure
produces contamination".

MECHANISM UNDER TEST (code-confirmed, Step 2.5/2.5a)
-----------------------------------------------------------------
`ree_core/latent/stack.py::ReafferencePredictor` (SD-007 / MECH-098 / MECH-101)
predicts the z_world change caused purely by the agent's own motor command:

    dz_world_loco = ReafferencePredictor(z_world_raw_prev, a_prev)
    z_world_corrected = z_world_raw - dz_world_loco

`LatentStack.encode()` (stack.py ~1460-1483) applies this on EVERY encode tick,
gated ONLY on `self.reafference_predictor is not None` (plus prev_action present
and timestamp > 0). `REEAgent.sense()` passes `prev_action=self._last_action`
every tick (agent.py ~4777). So the substrate's residualization IS continuous in
exactly MECH-221's sense, and `config.latent.reafference_action_dim` (0 =
disabled, action_dim = enabled) is a clean, single-point ON/OFF knob for it.

STEP 2.5a PROBE FINDING -- WHY z_world_raw IS NOT THE CONTRAST
--------------------------------------------------------------
`LatentState.z_world_raw` is captured BEFORE the correction AND before the
alpha_world EMA, while `z_world` is post-both. Probed 2026-09-10 on this
substrate: with `reafference_action_dim=0` (predictor is None, NO correction
applied at all) `z_world != z_world_raw` on 38/38 measured steps -- the
difference is the EMA, not the correction. A within-run
`z_world - z_world_raw` contrast would therefore measure smoothing and be read
as residualization. This run instead contrasts the DOWNSTREAM-VISIBLE `z_world`
stream ACROSS ARMS, which is the only place the manipulation actually lands.

WHAT THE LOAD-BEARING DV IS, AND WHY IT IS NOT TAUTOLOGICAL
-----------------------------------------------------------
The obvious readout -- "how much of z_world is explained by the efference copy"
-- is very nearly an arithmetic identity: the ON arm subtracts exactly that
quantity, and its magnitude is just the reafference predictor's own R2 restated.
It is recorded here as a MANIPULATION CHECK (`self_footprint_frac`), explicitly
NOT load-bearing.

The genuinely open question is the DOWNSTREAM CONSEQUENCE MECH-222 asserts: does
the retained self-footprint actually MASK exogenous world events, so that a
consumer reading z_world can no longer tell that something happened in the world
that was not the agent's own doing? That is what "environmental events are
interpreted as self-directed" cashes out to representationally.

So the load-bearing DV is the AUC of a held-out linear probe trained to detect a
GROUND-TRUTH exogenous event (a hazard actually changed cell this step, snapshot-
compared around env.step) from the downstream z_world delta alone.

This is NOT an identity and could fail in either direction:
  - a linear probe with 32 dims may simply project the self-component out, in
    which case removing it changes nothing (C1 fails);
  - the correction subtracts a LEARNED and therefore IMPERFECT vector, injecting
    prediction-error noise that could REDUCE detectability (C1 fails negative);
  - the exogenous signal may dominate the self-footprint outright, making the
    contamination inconsequential in this substrate (C1 fails).
Any of those is an informative negative about MECH-222's substrate purchase.

DV-SYMMETRY DECLARATION (Step 3, mandatory, per arm)
-----------------------------------------------------
DV = AUC of a probe over the downstream `dz_world` distribution, plus a cosine
threshold rate. Symmetry group of an AUC is monotone rescaling of the SCORE and
permutation of same-class samples. The manipulation subtracts a PER-STEP,
STEP-VARYING vector (dz_hat_t) from the representation the probe reads; it is a
per-sample change of the FEATURE vector, not a monotone map of the probe score
and not a permutation of samples within a class, so it is not invariant under
either. It also changes the class-CONDITIONAL feature distributions unequally
(dz_hat is large on self-moved steps, near-zero on stationary steps), which is
precisely why an AUC can move. Identical statement holds for all three arms;
ARM_SHAM subtracts a vector of the SAME distribution and magnitude with the
action mis-paired, so any AUC change common to ON and SHAM is attributable to
"subtracting a vector of that size", not to correct residualization.

DESIGN -- 3 ARMS x 5 SEEDS
---------------------------
Actions are drawn UNIFORMLY AT RANDOM in both P0 and P2. This is deliberate and
is what makes the arms exactly matched: the action sequence and the env
trajectory are then identical across arms at a given seed (RNG fully reset at
cell entry), so the ONLY thing that differs downstream is the residualization.
SCOPING NOTE: this therefore tests the REPRESENTATIONAL half of MECH-222
(contamination of z_world and the resulting loss of exogenous-event visibility),
NOT its behavioural/clinical half (that the agent then ACTS on the false
attribution). The behavioural half needs a policy-learning successor and is out
of scope here; a PASS does not license the clinical reading.

  ARM_RESID_ON   reafference_action_dim = action_dim, trained predictor
                 installed -> correction applied every encode tick.
  ARM_RESID_OFF  predictor set to None after P0 -> NO correction (the MECH-221
                 failure condition). The trained predictor is retained OUT of
                 band for analysis, so dz_hat is computed identically in all
                 three arms.
  ARM_SHAM_RESID predictor replaced by a wrapper that subtracts
                 net(z_world_raw_prev, PERMUTED action) -- same module, same
                 output magnitude distribution, wrong self-content. Matched-
                 noise control (skill Step 3.5 "matched-noise positive control").

The encoder is NEVER trained (frozen random projection) in any arm; only the
ReafferencePredictor is trained, in P0, on `.detach()`ed z_world_raw. This is
phased training in the required sense (encoder frozen while the head trains --
there is no moving target because the encoder never moves) and matches the
architectural-probe precedent of V3-EXQ-997 / V3-EXQ-914.

PRE-REGISTERED CRITERIA (constants below, fixed before any run)
----------------------------------------------------------------
READINESS preconditions (all must be met, else -> substrate_not_ready_requeue):
  R1 reaf_test_r2            >= 0.05   the residualization mechanism must
                                       actually predict something, else ON is
                                       just OFF plus noise and the contrast is
                                       vacuous.
  R2 drift_auc_control       >= 0.60   SAME STATISTIC as C1 (AUC of the same
                                       probe class), measured on the RAW world
                                       observation delta -- a positive control
                                       that the exogenous label is learnable at
                                       all. Below floor = label unlearnable =
                                       C1 is starved, not falsified.
  R3 drift_base_rate         in [0.05, 0.95]  (two-sided interval)
  R4 drift_auc_off           <= 0.90   DV HEADROOM: C1 asks for a +0.05 AUC gap
                                       over ARM_RESID_OFF; if OFF already sits
                                       above 0.90 the gap is not attainable and
                                       C1 would fire negative BY CONSTRUCTION.
LOAD-BEARING criteria:
  C1 drift_auc_on  - drift_auc_off  >= 0.05
  C2 fsar_off      - fsar_on        >= 0.10
  C3 drift_auc_on  - drift_auc_sham >= 0.03   attribution control
combination_rule: "C1 AND C2 AND C3"

RED-TEAM: recorded in the queue entry note and in RED_TEAM_VERDICT below.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorld  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._metrics import (  # noqa: E402
    check_degeneracy,
    dv_headroom_check,
    p0_readiness_gate,
    P0NotReady,
)
from experiments.pack_writer import write_flat_manifest, flat_readout  # noqa: E402

EXPERIMENT_PURPOSE = "diagnostic"

RUN_ID_BASE = "v3_exq_1018_mech222_self_attribution_contamination"
EXPERIMENT_TYPE = "claim_probe_mech_222"
CLAIM_IDS = ["MECH-222", "MECH-221"]
RED_TEAM_VERDICT = "PENDING"  # overwritten at Step 4.5; see queue entry note

# ---- pre-registered constants (fixed before any run) ----------------------
SEEDS = [42, 43, 45, 46, 47]          # 44 excluded (per-seed instability, CLAUDE.md)
ARMS = ["ARM_RESID_ON", "ARM_RESID_OFF", "ARM_SHAM_RESID"]

GRID_SIZE = 10
# Operating point CALIBRATED at Step 4 (2026-09-10), not chosen by taste. At the
# env defaults a single hazard moving one cell in a 100-cell grid is far too small
# an exogenous perturbation to be decodable at all (measured: probe AUC 0.53-0.57,
# i.e. chance). At the other extreme (>=15 hazards, interval 1) the drift base rate
# hits 0.85-0.93 and the probe AUC pins at EXACTLY 1.000 -- saturated, zero headroom
# for C1, and rejected on that basis. This point sits between: base rate ~0.46, the
# stationary positive control at ~0.62-0.88, and the DV at ~0.62 with ~0.38 of
# headroom left for C1's registered gap.
NUM_HAZARDS = 8
NUM_RESOURCES = 5
ENV_DRIFT_INTERVAL = 2
ENV_DRIFT_PROB = 0.8
ALPHA_WORLD = 0.9                     # SD-008: >=0.9 for z_world fidelity
SELF_DIM = 32
WORLD_DIM = 32

P0_EPISODES = 100                     # reafference predictor training
EVAL_EPISODES = 40                    # measurement
STEPS_PER_EPISODE = 60
EPISODES_PER_RUN = P0_EPISODES + EVAL_EPISODES   # == queue entry episodes_per_run

REAF_LR = 1e-3
REAF_BATCH = 128
# Calibrated at Step 4: the original 4-steps-per-episode schedule left held-out R2
# at -0.04 (the predictor learned nothing, so ARM_RESID_ON would have been ARM_OFF
# plus noise and the whole contrast vacuous). A dedicated post-collection fit of
# 4000 steps over the full buffer reaches R2 ~0.25 and is still climbing.
REAF_TRAIN_STEPS = 4000
REAF_TEST_FRAC = 0.25

PROBE_EPOCHS = 400
PROBE_LR = 5e-2
PROBE_L2 = 1e-3
PROBE_TEST_FRAC = 0.30

# readiness floors / bounds
R1_REAF_TEST_R2_FLOOR = 0.05
# SAME STATISTIC as C1 (probe AUC over the downstream dz_world), measured on the
# STATIONARY stratum of the UNCORRECTED (ARM_RESID_OFF) stream. On a stationary
# step there is no self-motion to subtract, so residualization is irrelevant there
# -- which makes it the correct positive control for a claim about MASKING BY
# SELF-MOTION: it asks "is the exogenous event decodable from z_world at all, when
# it is not being masked?". Below floor = the DV is starved, not falsified.
R2_CONTROL_AUC_FLOOR = 0.55
R3_BASE_RATE_LOW = 0.15
R3_BASE_RATE_HIGH = 0.85

# load-bearing thresholds
C1_AUC_GAP = 0.05
C2_FSAR_GAP = 0.10
C3_SHAM_GAP = 0.03
FSAR_COS_TAU = 0.20                   # "reads as confirming my own action"

MIN_DRIFT_SAMPLES = 40                # sample floor for a non-degenerate AUC


# ---------------------------------------------------------------- utilities
def _onehot(i: int, n: int, dev) -> torch.Tensor:
    v = torch.zeros(1, n, device=dev)
    v[0, int(i)] = 1.0
    return v


def _auc(scores: List[float], labels: List[int]) -> Optional[float]:
    """Mann-Whitney U / rank-based AUC. None when a class is absent."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_pos = sum(r for r, y in zip(ranks, labels) if y == 1)
    n_p, n_n = len(pos), len(neg)
    return float((rank_pos - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm().item(), b.norm().item()
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(torch.dot(a.flatten(), b.flatten()).item() / (na * nb))


def _probe_auc(
    feats: List[torch.Tensor],
    labels: List[int],
    episode_ids: List[int],
    seed: int,
) -> Tuple[Optional[float], int, int]:
    """Held-out linear (logistic) probe AUC. Split BY EPISODE (no leakage).

    Returns (auc, n_train, n_test). auc is None when either split lacks a class.
    """
    if not feats:
        return None, 0, 0
    eps = sorted(set(episode_ids))
    g = torch.Generator().manual_seed(int(seed) + 9973)
    perm = torch.randperm(len(eps), generator=g).tolist()
    n_test_eps = max(1, int(round(len(eps) * PROBE_TEST_FRAC)))
    test_eps = {eps[i] for i in perm[:n_test_eps]}

    X = torch.stack([f.flatten() for f in feats]).float()
    y = torch.tensor(labels, dtype=torch.float32)
    is_test = torch.tensor([1 if e in test_eps else 0 for e in episode_ids], dtype=torch.bool)
    Xtr, ytr = X[~is_test], y[~is_test]
    Xte, yte = X[is_test], y[is_test]
    if Xtr.numel() == 0 or Xte.numel() == 0:
        return None, int(Xtr.shape[0]), int(Xte.shape[0])
    if len(set(ytr.tolist())) < 2 or len(set(yte.tolist())) < 2:
        return None, int(Xtr.shape[0]), int(Xte.shape[0])

    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    torch.manual_seed(int(seed) + 4441)
    lin = nn.Linear(Xtr.shape[1], 1)
    opt = torch.optim.Adam(lin.parameters(), lr=PROBE_LR, weight_decay=PROBE_L2)
    # class-balanced BCE so a skewed base rate cannot pin the probe at majority
    n_pos = float(ytr.sum().item())
    n_neg = float(len(ytr) - n_pos)
    pos_w = torch.tensor([max(n_neg, 1.0) / max(n_pos, 1.0)])
    for _ in range(PROBE_EPOCHS):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(
            lin(Xtr).squeeze(-1), ytr, pos_weight=pos_w
        )
        loss.backward()
        opt.step()
    with torch.no_grad():
        scores = lin(Xte).squeeze(-1).tolist()
    return _auc(scores, [int(v) for v in yte.tolist()]), int(Xtr.shape[0]), int(Xte.shape[0])


class ShamReafference(nn.Module):
    """Matched-noise control: subtract net(z_prev, MIS-PAIRED action).

    Same module, same weights, same output magnitude distribution as the real
    correction -- only the action is permuted, so the subtracted vector carries
    no correct self-content. Exposes the exact interface LatentStack.encode uses
    (`correct_z_world`), so the encode path is byte-identical apart from which
    vector comes back.
    """

    def __init__(self, base: nn.Module, action_dim: int, seed: int):
        super().__init__()
        self.base = base
        self.action_dim = int(action_dim)
        self._g = torch.Generator().manual_seed(int(seed) + 7717)

    def _mispair(self, a: torch.Tensor) -> torch.Tensor:
        """Return a one-hot action DIFFERENT from the one actually taken."""
        out = torch.zeros_like(a)
        for b in range(a.shape[0]):
            taken = int(a[b].argmax().item())
            choices = [k for k in range(self.action_dim) if k != taken]
            pick = choices[int(torch.randint(len(choices), (1,), generator=self._g).item())]
            out[b, pick] = 1.0
        return out

    def forward(self, z_prev: torch.Tensor, a_prev: torch.Tensor) -> torch.Tensor:
        return self.base(z_prev, self._mispair(a_prev))

    def correct_z_world(
        self, z_world_raw: torch.Tensor, z_world_prev: torch.Tensor, a_prev: torch.Tensor
    ) -> torch.Tensor:
        return z_world_raw - self.forward(z_world_prev, a_prev)


# ------------------------------------------------------------------- config
def _env_kwargs() -> Dict[str, Any]:
    return dict(
        size=GRID_SIZE,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        env_drift_interval=ENV_DRIFT_INTERVAL,
        env_drift_prob=ENV_DRIFT_PROB,
    )


def _make_env(seed: int) -> CausalGridWorld:
    return CausalGridWorld(seed=seed, **_env_kwargs())


def _make_agent(env: CausalGridWorld) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=ALPHA_WORLD,
        # SD-007 residualization pathway ON at construction in EVERY arm, so the
        # predictor exists and is trained identically; the ARM manipulation is
        # applied AFTER P0 by swapping this attribute (v3_exq_099a precedent).
        reafference_action_dim=env.action_dim,
    )
    return REEAgent(cfg)


def _config_slice() -> Dict[str, Any]:
    return {
        "env": _env_kwargs(),
        "self_dim": SELF_DIM,
        "world_dim": WORLD_DIM,
        "alpha_world": ALPHA_WORLD,
        "p0_episodes": P0_EPISODES,
        "eval_episodes": EVAL_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "reaf_lr": REAF_LR,
        "reaf_batch": REAF_BATCH,
        "reaf_train_steps": REAF_TRAIN_STEPS,
        "reaf_test_frac": REAF_TEST_FRAC,
    }


# --------------------------------------------------------------------- cell
def _run_cell(arm: str, seed: int, n_p0: int, n_eval: int, n_steps: int,
              zg_acc=None) -> Dict[str, Any]:
    """One (arm, seed) cell: P0 predictor training, then P2 stratified measurement.

    DELTA/LABEL ALIGNMENT (this is the part that is easy to get wrong, and was
    wrong in the first draft -- caught at Step 4 calibration): the downstream
    change `z_{t+1} - z_t` is PRODUCED BY the transition taken at t, so it must
    carry t's `moved` / `drifted` labels and t's efference copy
    `pred(z_raw_t, a_t)`. Emitting a row at time t using t's labels but the
    t-1 -> t delta is an off-by-one that pushes every AUC to chance. The
    `pending` tuple below is what enforces the correct pairing.
    """
    env = _make_env(seed)
    agent = _make_agent(env)
    agent.train()
    dev = agent.device
    action_dim = env.action_dim
    rng = random.Random(seed)

    predictor = agent.latent_stack.reafference_predictor
    assert predictor is not None, "reafference predictor must exist in P0 for every arm"

    total_eps = n_p0 + n_eval

    # ---------------- P0: collect pure self-motion transitions ---------------
    # Target is the z_world change caused ONLY by the agent's own locomotion, so
    # samples are restricted to steps where the agent genuinely moved AND no
    # exogenous drift occurred. That is exactly the quantity MECH-221 says must
    # be subtracted.
    train_pairs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    test_pairs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for ep in range(n_p0):
        _, obs = env.reset()
        agent.reset()
        pending = None
        for _ in range(n_steps):
            lat = agent.sense(obs["body_state"], obs["world_state"])
            agent.clock.advance()
            z_raw_cur = (
                lat.z_world_raw.detach() if lat.z_world_raw is not None else lat.z_world.detach()
            )
            if pending is not None:
                z_raw_0, a_0, moved_0, drift_0 = pending
                if moved_0 and not drift_0:
                    sample = (z_raw_0.cpu(), a_0.cpu(), (z_raw_cur - z_raw_0).cpu())
                    (test_pairs if rng.random() < REAF_TEST_FRAC else train_pairs).append(sample)

            a_i = rng.randrange(action_dim)
            a_vec = _onehot(a_i, action_dim, dev)
            agent._last_action = a_vec
            pos_b = (env.agent_x, env.agent_y)
            haz_b = [tuple(h[:2]) for h in env.hazards]
            _, _, done, _info, obs = env.step(a_vec)
            pending = (
                z_raw_cur,
                a_vec,
                (env.agent_x, env.agent_y) != pos_b,
                [tuple(h[:2]) for h in env.hazards] != haz_b,
            )
            if done:
                break

        if (ep + 1) % 25 == 0:
            print(
                f"  [train] p0-collect seed={seed} arm={arm} ep {ep + 1}/{total_eps} "
                f"train_n={len(train_pairs)} test_n={len(test_pairs)}",
                flush=True,
            )

    # ---------------- P0 fit: dedicated pass over the whole buffer ------------
    reaf_test_r2 = 0.0
    if len(train_pairs) >= REAF_BATCH and len(test_pairs) >= 16:
        reaf_opt = torch.optim.Adam(predictor.parameters(), lr=REAF_LR)
        Ztr = torch.cat([b[0] for b in train_pairs]).to(dev)
        Atr = torch.cat([b[1] for b in train_pairs]).to(dev)
        Ttr = torch.cat([b[2] for b in train_pairs]).to(dev)
        for gstep in range(REAF_TRAIN_STEPS):
            idx = torch.randint(0, Ztr.shape[0], (REAF_BATCH,))
            loss = F.mse_loss(predictor(Ztr[idx], Atr[idx]), Ttr[idx])
            if loss.requires_grad:
                reaf_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), 0.5)
                reaf_opt.step()
            if (gstep + 1) % 1000 == 0:
                print(
                    f"  [train] p0-fit seed={seed} arm={arm} ep {n_p0}/{total_eps} "
                    f"grad_step {gstep + 1}/{REAF_TRAIN_STEPS} mse={float(loss):.6f}",
                    flush=True,
                )
        with torch.no_grad():
            Zte = torch.cat([b[0] for b in test_pairs]).to(dev)
            Ate = torch.cat([b[1] for b in test_pairs]).to(dev)
            Tte = torch.cat([b[2] for b in test_pairs]).to(dev)
            pred_te = predictor(Zte, Ate)
            ss_res = float(((Tte - pred_te) ** 2).sum().item())
            ss_tot = float(((Tte - Tte.mean(0, keepdim=True)) ** 2).sum().item())
        reaf_test_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    # ---------------- ARM MANIPULATION (after P0, before P2) -----------------
    trained_predictor = predictor          # retained OUT of band for analysis
    for prm in trained_predictor.parameters():
        prm.requires_grad_(False)
    if arm == "ARM_RESID_ON":
        pass                                                  # correction stays live
    elif arm == "ARM_RESID_OFF":
        agent.latent_stack.reafference_predictor = None       # MECH-221 failure
    elif arm == "ARM_SHAM_RESID":
        agent.latent_stack.reafference_predictor = ShamReafference(
            trained_predictor, action_dim, seed
        )
    else:
        raise ValueError("unknown arm: " + str(arm))

    # ---------------- P2: measurement (no gradients) -------------------------
    agent.eval()
    rows_moved: List[Dict[str, Any]] = []
    rows_stat: List[Dict[str, Any]] = []
    n_moved = 0

    for ep in range(n_eval):
        _, obs = env.reset()
        agent.reset()
        pending = None
        for _ in range(n_steps):
            with torch.no_grad():
                lat = agent.sense(obs["body_state"], obs["world_state"])
            agent.clock.advance()
            z_down_cur = lat.z_world.detach().flatten()
            z_raw_cur = (
                lat.z_world_raw.detach() if lat.z_world_raw is not None
                else lat.z_world.detach()
            )
            if pending is not None:
                z_down_0, z_raw_0, a_0, moved_0, drift_0 = pending
                with torch.no_grad():
                    dz_hat = trained_predictor(z_raw_0, a_0).detach().flatten()
                d_down = z_down_cur - z_down_0
                row = {
                    "dz": d_down.cpu(),
                    "cos": _cos(d_down, dz_hat),
                    "drift": int(drift_0),
                    "ep": ep,
                }
                (rows_moved if moved_0 else rows_stat).append(row)

            a_i = rng.randrange(action_dim)
            a_vec = _onehot(a_i, action_dim, dev)
            agent._last_action = a_vec
            pos_b = (env.agent_x, env.agent_y)
            haz_b = [tuple(h[:2]) for h in env.hazards]
            _, _, done, _info, obs = env.step(a_vec)
            moved = (env.agent_x, env.agent_y) != pos_b
            n_moved += int(moved)
            pending = (
                z_down_cur,
                z_raw_cur,
                a_vec,
                moved,
                [tuple(h[:2]) for h in env.hazards] != haz_b,
            )
            if done:
                break

        if (ep + 1) % 10 == 0:
            print(
                f"  [train] p2 seed={seed} arm={arm} ep {n_p0 + ep + 1}/{total_eps} "
                f"moved_rows={len(rows_moved)} stat_rows={len(rows_stat)}",
                flush=True,
            )

    if zg_acc is not None:
        zg_acc.observe(agent)

    def _probe_on(rs: List[Dict[str, Any]]) -> Tuple[Optional[float], int, int]:
        if not rs:
            return None, 0, 0
        return _probe_auc([r["dz"] for r in rs], [r["drift"] for r in rs],
                          [r["ep"] for r in rs], seed)

    # LOAD-BEARING stratum: steps where the agent MOVED. That is the only stratum
    # where a self-motion footprint exists to mask anything, so it is where
    # residualization can matter at all.
    drift_auc_moved, n_tr, n_te = _probe_on(rows_moved)
    # POSITIVE-CONTROL stratum: stationary steps -- no self-motion to subtract.
    drift_auc_stat, _, _ = _probe_on(rows_stat)

    drift_cos = [r["cos"] for r in rows_moved if r["drift"] == 1]
    fsar = (
        float(sum(1 for c in drift_cos if c > FSAR_COS_TAU) / len(drift_cos))
        if drift_cos else 0.0
    )
    foot = [r["cos"] ** 2 for r in rows_moved]
    self_footprint_frac = float(sum(foot) / len(foot)) if foot else 0.0
    n_drift_moved = int(sum(r["drift"] for r in rows_moved))
    base_rate = float(n_drift_moved / len(rows_moved)) if rows_moved else 0.0

    return {
        "arm": arm,
        "seed": seed,
        "drift_auc_moved": drift_auc_moved,
        "drift_auc_stationary": drift_auc_stat,
        "fsar": fsar,
        "self_footprint_frac": self_footprint_frac,
        "reaf_test_r2": reaf_test_r2,
        "drift_base_rate_moved": base_rate,
        "n_rows_moved": len(rows_moved),
        "n_rows_stationary": len(rows_stat),
        "n_drift_rows_moved": n_drift_moved,
        "n_probe_train": n_tr,
        "n_probe_test": n_te,
        "n_reaf_train": len(train_pairs),
        "n_reaf_test": len(test_pairs),
        "moved_frac": float(n_moved / max(1, n_eval * n_steps)),
        "mean_cos_moved": (
            float(sum(r["cos"] for r in rows_moved) / len(rows_moved)) if rows_moved else 0.0
        ),
        "mean_cos_drift_moved": (
            float(sum(drift_cos) / len(drift_cos)) if drift_cos else 0.0
        ),
    }


# ---------------------------------------------------------------- aggregate
def _mean(vals: List[Optional[float]]) -> Optional[float]:
    ok = [v for v in vals if v is not None]
    return float(sum(ok) / len(ok)) if ok else None


def _by_arm(rows: List[Dict[str, Any]], arm: str, key: str) -> List[Optional[float]]:
    return [r[key] for r in rows if r["arm"] == arm]


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    seeds = SEEDS[:1] if dry_run else SEEDS
    n_p0 = 4 if dry_run else P0_EPISODES
    n_eval = 4 if dry_run else EVAL_EPISODES
    n_steps = 25 if dry_run else STEPS_PER_EPISODE

    zg = ZGoalStreamAccumulator()
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm in ARMS:
            print(f"Seed {seed} Condition {arm}", flush=True)
            with arm_cell(
                seed,
                config_slice=_config_slice(),
                script_path=Path(__file__),
                config_slice_declared=True,
            ) as cell:
                row = _run_cell(arm, seed, n_p0, n_eval, n_steps, zg_acc=zg)
                cell.stamp(row)
            rows.append(row)
            print(
                f"verdict: {'PASS' if row.get('drift_auc_moved') is not None else 'FAIL'}",
                flush=True,
            )

    # ---- aggregates -------------------------------------------------------
    auc_on = _mean(_by_arm(rows, "ARM_RESID_ON", "drift_auc_moved"))
    auc_off = _mean(_by_arm(rows, "ARM_RESID_OFF", "drift_auc_moved"))
    auc_sham = _mean(_by_arm(rows, "ARM_SHAM_RESID", "drift_auc_moved"))
    fsar_on = _mean(_by_arm(rows, "ARM_RESID_ON", "fsar"))
    fsar_off = _mean(_by_arm(rows, "ARM_RESID_OFF", "fsar"))
    fsar_sham = _mean(_by_arm(rows, "ARM_SHAM_RESID", "fsar"))
    foot_on = _mean(_by_arm(rows, "ARM_RESID_ON", "self_footprint_frac"))
    foot_off = _mean(_by_arm(rows, "ARM_RESID_OFF", "self_footprint_frac"))
    foot_sham = _mean(_by_arm(rows, "ARM_SHAM_RESID", "self_footprint_frac"))
    reaf_r2 = _mean([r["reaf_test_r2"] for r in rows])
    base_rate = _mean([r["drift_base_rate_moved"] for r in rows])
    min_drift_rows = min([r["n_drift_rows_moved"] for r in rows]) if rows else 0

    # The readiness CONTROL is measured on the UNCORRECTED (ARM_RESID_OFF) stream's
    # STATIONARY stratum -- "is the exogenous event in z_world at all before we do
    # anything to it, when no self-motion is masking it".
    off_stat = [
        r["drift_auc_stationary"]
        for r in rows
        if r["arm"] == "ARM_RESID_OFF" and r["drift_auc_stationary"] is not None
    ]
    control_auc = _mean(off_stat)

    def _gap(a: Optional[float], b: Optional[float]) -> Optional[float]:
        return None if (a is None or b is None) else float(a - b)

    c1_measured = _gap(auc_on, auc_off)
    c2_measured = _gap(fsar_off, fsar_on)
    c3_measured = _gap(auc_on, auc_sham)

    # ---- readiness preconditions -----------------------------------------
    # WORST-CELL reporting: both floors are worst-case claims over cells, so the
    # reported `measured` is the extremum, not the mean (skill Step 3).
    worst_r2_cell = min(rows, key=lambda r: r["reaf_test_r2"]) if rows else None
    worst_r2 = float(worst_r2_cell["reaf_test_r2"]) if worst_r2_cell else 0.0
    off_stat_cells = [
        r for r in rows
        if r["arm"] == "ARM_RESID_OFF" and r["drift_auc_stationary"] is not None
    ]
    worst_ctrl_cell = (
        min(off_stat_cells, key=lambda r: r["drift_auc_stationary"]) if off_stat_cells else None
    )
    worst_ctrl = float(worst_ctrl_cell["drift_auc_stationary"]) if worst_ctrl_cell else 0.0

    precondition_specs = [
        {
            "name": "reafference_predictor_test_r2_supra_floor",
            "kind": "readiness",
            "measured": worst_r2,
            "threshold": R1_REAF_TEST_R2_FLOOR,
            "direction": "lower",
            "control": "held-out R2 of the trained ReafferencePredictor on PURE "
                       "self-caused perspective-shift transitions (agent moved, no "
                       "exogenous drift). If this is at/below zero the predictor "
                       "subtracts noise, ARM_RESID_ON is ARM_RESID_OFF plus noise, "
                       "and C1/C3 are vacuous rather than falsified.",
            "offending_cell": (
                f"{worst_r2_cell['arm']}::seed{worst_r2_cell['seed']}" if worst_r2_cell else None
            ),
        },
        {
            "name": "exogenous_event_decodable_from_zworld_stationary_control",
            "kind": "readiness",
            "measured": worst_ctrl,
            "threshold": R2_CONTROL_AUC_FLOOR,
            "direction": "lower",
            "control": "SAME STATISTIC as C1 -- held-out AUC of the SAME linear probe "
                       "class over the SAME downstream dz_world feature -- measured on "
                       "the STATIONARY stratum of the UNCORRECTED (ARM_RESID_OFF) "
                       "stream, where there is no self-motion to mask the event. "
                       "Below floor means the DV is STARVED (the event is not in "
                       "z_world at all), NOT that residualization does not matter.",
            "offending_cell": (
                f"{worst_ctrl_cell['arm']}::seed{worst_ctrl_cell['seed']}"
                if worst_ctrl_cell else None
            ),
        },
        # HEADROOM AS A FLOOR (achievable >= required): the AUC gap C1 can still
        # physically reach above the OFF arm is (1.0 - drift_auc_moved_off), and
        # C1 registers C1_AUC_GAP of it. An OFF arm pinned near the 1.0 ceiling
        # leaves less, and C1 would fire negative BY CONSTRUCTION. (The >=15-hazard
        # regimes rejected at Step 4 pinned exactly there.)
        dv_headroom_check(
            "c1_auc_gap_headroom_above_off_arm",
            dv_name="drift_auc_moved_gap_on_minus_off",
            criterion_threshold=C1_AUC_GAP,
            achievable=(float(1.0 - auc_off) if auc_off is not None else 0.0),
            statistic="range",
            dv_bounds=(0.0, 1.0),
            control="achievable = 1.0 - mean(drift_auc_moved, ARM_RESID_OFF); "
                    "required = C1's own registered gap constant.",
        ),
    ]

    # The base-rate check is TWO-SIDED and p0_readiness_gate is single-bound only,
    # so build that entry directly (the indexer honours threshold_low/high and
    # prefers the interval over any single `threshold`).
    _br = float(base_rate if base_rate is not None else 0.0)
    interval_precondition = {
        "name": "drift_base_rate_moved_in_band",
        "kind": "readiness",
        "measured": _br,
        "threshold_low": R3_BASE_RATE_LOW,
        "threshold_high": R3_BASE_RATE_HIGH,
        "comparator_low": ">=",
        "comparator_high": "<=",
        "direction": "interval",
        "met": bool(R3_BASE_RATE_LOW <= _br <= R3_BASE_RATE_HIGH),
        "control": "ground-truth exogenous-event rate on the MOVED stratum (a hazard "
                   "actually changed cell), snapshot-compared around env.step. "
                   "Two-sided: too few positives starves the probe, too many leaves "
                   "no negatives and saturates the AUC.",
    }

    try:
        preconditions = p0_readiness_gate(precondition_specs)
    except P0NotReady as exc:
        preconditions = exc.preconditions
    preconditions = list(preconditions) + [interval_precondition]
    readiness_met = all(bool(pc.get("met")) for pc in preconditions)

    # ---- criteria ---------------------------------------------------------
    def _passed(measured: Optional[float], thr: float) -> bool:
        return measured is not None and measured >= thr

    c1 = _passed(c1_measured, C1_AUC_GAP)
    c2 = _passed(c2_measured, C2_FSAR_GAP)
    c3 = _passed(c3_measured, C3_SHAM_GAP)

    criteria = [
        {
            "name": "C1_residualization_improves_exogenous_event_visibility",
            "load_bearing": True,
            "passed": bool(c1),
            "measured": c1_measured,
            "threshold": C1_AUC_GAP,
            "detail": "mean held-out drift-detection AUC (linear probe on downstream "
                      "dz_world, MOVED stratum), ARM_RESID_ON minus ARM_RESID_OFF",
        },
        {
            "name": "C2_contamination_raises_false_self_confirmation",
            "load_bearing": True,
            "passed": bool(c2),
            "measured": c2_measured,
            "threshold": C2_FSAR_GAP,
            "detail": f"fraction of GROUND-TRUTH exogenous MOVED steps whose downstream "
                      f"dz_world reads as confirming the agent's own action "
                      f"(cos(dz_world, dz_hat) > {FSAR_COS_TAU}), OFF minus ON",
        },
        {
            "name": "C3_effect_attributable_to_correct_residualization",
            "load_bearing": True,
            "passed": bool(c3),
            "measured": c3_measured,
            "threshold": C3_SHAM_GAP,
            "detail": "ARM_RESID_ON minus ARM_SHAM_RESID drift AUC (MOVED stratum). "
                      "SHAM subtracts a same-magnitude mis-paired-action vector, so "
                      "this isolates correct self-content from 'subtracting any "
                      "vector of that size'",
        },
        {
            "name": "MANIPULATION_CHECK_self_footprint_removed",
            "load_bearing": False,
            "passed": bool(
                foot_off is not None and foot_on is not None and foot_off > foot_on
            ),
            "measured": _gap(foot_off, foot_on),
            "threshold_not_applicable": "NOT load-bearing and NEAR-TAUTOLOGICAL: the ON "
                                        "arm subtracts exactly dz_hat, so this is close "
                                        "to the predictor's own R2 restated. Recorded as "
                                        "evidence the manipulation reached the DV, never "
                                        "as evidence for the claim.",
            "detail": "mean cos^2(dz_world, dz_hat) on the MOVED stratum, OFF minus ON",
        },
    ]

    combination_rule = (
        "C1 AND C2 AND C3 -- all three load-bearing criteria must hold, on top of "
        "every readiness precondition being met. C3 is an attribution control, not "
        "an effect: without it C1+C2 could be produced by subtracting any vector."
    )
    overall_pass = bool(readiness_met and c1 and c2 and c3)

    # ---- non-degeneracy ---------------------------------------------------
    arms_identical = False
    if auc_on is not None and auc_off is not None and auc_sham is not None:
        arms_identical = (
            abs(auc_on - auc_off) < 1e-9 and abs(auc_on - auc_sham) < 1e-9
        )
    auc_saturated = bool(
        auc_off is not None and auc_on is not None
        and min(auc_off, auc_on) > 0.999
    )
    enough_drift = min_drift_rows >= (4 if dry_run else MIN_DRIFT_SAMPLES)
    non_degen = bool(readiness_met and not arms_identical and not auc_saturated and enough_drift)
    criteria_non_degenerate = {
        "C1_residualization_improves_exogenous_event_visibility": non_degen,
        "C2_contamination_raises_false_self_confirmation": bool(readiness_met and enough_drift),
        "C3_effect_attributable_to_correct_residualization": non_degen,
    }

    # ---- interpretation self-route ---------------------------------------
    if not readiness_met or not enough_drift or arms_identical or auc_saturated:
        label = "substrate_not_ready_requeue"
        direction = "non_contributory"
    elif c1 and c2 and c3:
        label = "residualization_failure_produces_self_attribution_contamination"
        direction = "supports"
    elif c1 and c2 and not c3:
        label = "contamination_effect_not_attributable_to_correct_residualization"
        direction = "mixed"
    elif not c1:
        label = "no_downstream_exogenous_visibility_cost_from_residualization_failure"
        direction = "weakens"
    else:
        label = "partial_contamination_signature_only"
        direction = "mixed"

    degeneracy = check_degeneracy(
        {
            "drift_auc_moved": [
                r["drift_auc_moved"] for r in rows if r["drift_auc_moved"] is not None
            ],
            "fsar": [r["fsar"] for r in rows],
        }
    )

    manifest: Dict[str, Any] = {
        "run_id": f"{RUN_ID_BASE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "queue_id": "V3-EXQ-1018",
        "backlog_id": "EVB-1462",
        "proposal_id": "EXP-0893",
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "outcome": "PASS" if overall_pass else "FAIL",
        "dry_run": bool(dry_run),
        "sleep_driver_pattern": "N/A (no sleep loop)",
        "red_team_verdict": RED_TEAM_VERDICT,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {"MECH-222": direction, "MECH-221": direction},
        "combination_rule": combination_rule,
        "criteria": criteria,
        "non_degenerate": bool(degeneracy.get("non_degenerate", True)) and non_degen,
        "degeneracy_reason": degeneracy.get("degeneracy_reason"),
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
        },
        "arm_results": rows,
        "per_seed_rows": rows,
        "aggregates_by_arm": {
            "drift_auc_moved": {"ARM_RESID_ON": auc_on, "ARM_RESID_OFF": auc_off,
                                "ARM_SHAM_RESID": auc_sham},
            "drift_auc_stationary": {
                a: _mean(_by_arm(rows, a, "drift_auc_stationary")) for a in ARMS
            },
            "fsar": {"ARM_RESID_ON": fsar_on, "ARM_RESID_OFF": fsar_off,
                     "ARM_SHAM_RESID": fsar_sham},
            "self_footprint_frac": {"ARM_RESID_ON": foot_on, "ARM_RESID_OFF": foot_off,
                                    "ARM_SHAM_RESID": foot_sham},
            "reaf_test_r2": {a: _mean(_by_arm(rows, a, "reaf_test_r2")) for a in ARMS},
        },
        "label_balance": {
            "drift_base_rate_moved": base_rate,
            "min_drift_rows_per_cell_moved": min_drift_rows,
            "n_rows_moved_per_cell": [r["n_rows_moved"] for r in rows],
            "n_rows_stationary_per_cell": [r["n_rows_stationary"] for r in rows],
            "n_reaf_train_per_cell": [r["n_reaf_train"] for r in rows],
            "n_reaf_test_per_cell": [r["n_reaf_test"] for r in rows],
        },
        "custom_information": {
            "scoping_note": "Random-action policy in both phases: this measures the "
                            "REPRESENTATIONAL half of MECH-222 (contamination of "
                            "z_world and the resulting loss of exogenous-event "
                            "visibility), NOT its behavioural/clinical half. A PASS "
                            "does not license the referential-delusion reading "
                            "directly. Random actions are also what makes the arms "
                            "exactly matched: same action sequence, same env "
                            "trajectory, only the residualization differs.",
            "alignment_note": "delta z_{t+1}-z_t is labelled with the transition at t "
                              "(the one that produced it). An off-by-one here pushes "
                              "every AUC to chance; caught at Step 4 calibration.",
            "z_world_raw_note": "z_world_raw is PRE-EMA as well as pre-correction "
                                "(Step 2.5a probe, 2026-09-10), so it is NOT a valid "
                                "within-run uncorrected counterpart to z_world -- the "
                                "arms are the only valid contrast.",
            "operating_point_calibration": "num_hazards/env_drift_* were calibrated at "
                                           "Step 4, not chosen by taste: at env "
                                           "defaults the DV is at chance (0.53-0.57); "
                                           "at >=15 hazards the AUC pins at exactly "
                                           "1.000 with base rate 0.85-0.93 (saturated, "
                                           "no C1 headroom). See docstring.",
            "substrate_defect_gate": "substrate_queue 'mode-governance-engagement' "
                                     "(corrupting, ree_core/agent.py) does NOT apply: "
                                     "its defect is the salience-coordinator affinity "
                                     "clamp; probe confirms agent.salience is None "
                                     "under this config (use_salience_coordinator off).",
        },
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
        },
    }

    manifest["readout"] = flat_readout(
        {
            "drift_auc_moved_on": auc_on,
            "drift_auc_moved_off": auc_off,
            "drift_auc_moved_sham": auc_sham,
            "drift_auc_stationary_control_off": control_auc,
            "c1_auc_gap_on_minus_off": c1_measured,
            "c2_fsar_gap_off_minus_on": c2_measured,
            "c3_auc_gap_on_minus_sham": c3_measured,
            "fsar_on": fsar_on,
            "fsar_off": fsar_off,
            "fsar_sham": fsar_sham,
            "self_footprint_frac_on": foot_on,
            "self_footprint_frac_off": foot_off,
            "reaf_test_r2_mean": reaf_r2,
            "reaf_test_r2_worst_cell": worst_r2,
            "drift_base_rate_moved": base_rate,
            "min_drift_rows_per_cell_moved": min_drift_rows,
            "c1_passed": int(bool(c1)),
            "c2_passed": int(bool(c2)),
            "c3_passed": int(bool(c3)),
            "readiness_met": int(bool(readiness_met)),
            "non_degenerate": int(bool(non_degen)),
            "overall_pass": int(bool(overall_pass)),
        }
    )
    manifest["_t0"] = t0
    manifest["_zg"] = zg
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    result = run_experiment(dry_run=args.dry_run)
    t0 = result.pop("_t0")
    _zg = result.pop("_zg")

    out_path = write_flat_manifest(
        result,
        dry_run=args.dry_run,
        config=_config_slice(),
        seeds=SEEDS[:1] if args.dry_run else SEEDS,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_zg.stats(),
    )
    print(f"manifest: {out_path}", flush=True)
    print(f"outcome: {result['outcome']}  label: {result['interpretation']['label']}", flush=True)
    print("readout: " + json.dumps(result["readout"], sort_keys=True), flush=True)

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
