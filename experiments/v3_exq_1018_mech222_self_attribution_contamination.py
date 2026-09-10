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
from experiments._metrics import check_degeneracy, p0_readiness_gate, P0NotReady  # noqa: E402
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
NUM_HAZARDS = 3
NUM_RESOURCES = 5
ENV_DRIFT_INTERVAL = 3                # tightened from default 5 (Step 2.5a: raises
ENV_DRIFT_PROB = 0.6                  # the real-movement base rate ~0.13 -> ~0.31)
ALPHA_WORLD = 0.9                     # SD-008: >=0.9 for z_world fidelity
SELF_DIM = 32
WORLD_DIM = 32

P0_EPISODES = 40                      # reafference predictor training
EVAL_EPISODES = 30                    # measurement
STEPS_PER_EPISODE = 60
EPISODES_PER_RUN = P0_EPISODES + EVAL_EPISODES   # == queue entry episodes_per_run

REAF_LR = 1e-3
REAF_BATCH = 64
REAF_STEPS_PER_EP = 4
REAF_TEST_FRAC = 0.25

PROBE_EPOCHS = 400
PROBE_LR = 5e-2
PROBE_L2 = 1e-3
PROBE_TEST_FRAC = 0.30

# readiness floors / bounds
R1_REAF_TEST_R2_FLOOR = 0.05
R2_CONTROL_AUC_FLOOR = 0.60
R3_BASE_RATE_LOW = 0.05
R3_BASE_RATE_HIGH = 0.95
R4_OFF_AUC_CEILING = 0.90

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
        "reaf_steps_per_ep": REAF_STEPS_PER_EP,
        "reaf_test_frac": REAF_TEST_FRAC,
    }


# --------------------------------------------------------------------- cell
def _run_cell(arm: str, seed: int, n_p0: int, n_eval: int, n_steps: int) -> Dict[str, Any]:
    """One (arm, seed) cell: P0 predictor training, then P2 measurement."""
    dev_env = _make_env(seed)
    agent = _make_agent(dev_env)
    agent.train()
    dev = agent.device
    action_dim = dev_env.action_dim
    rng = random.Random(seed)

    predictor = agent.latent_stack.reafference_predictor
    assert predictor is not None, "reafference predictor must exist in P0 for every arm"
    reaf_opt = torch.optim.Adam(predictor.parameters(), lr=REAF_LR)

    # ---------------- P0: train the reafference predictor (encoder frozen) ---
    # Samples are collected ONLY on steps where the agent genuinely moved and no
    # exogenous drift occurred, so the target is pure self-caused perspective
    # shift -- the quantity MECH-221 says must be subtracted.
    train_pairs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    test_pairs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    total_eps = n_p0 + n_eval

    for ep in range(n_p0):
        _, obs = dev_env.reset()
        agent.reset()
        z_raw_prev: Optional[torch.Tensor] = None
        a_prev: Optional[torch.Tensor] = None
        for _ in range(n_steps):
            lat = agent.sense(obs["body_state"], obs["world_state"])
            agent.clock.advance()
            z_raw_cur = lat.z_world_raw.detach() if lat.z_world_raw is not None else lat.z_world.detach()

            a_i = rng.randrange(action_dim)
            a_vec = _onehot(a_i, action_dim, dev)
            agent._last_action = a_vec

            pos_before = (dev_env.agent_x, dev_env.agent_y)
            haz_before = [tuple(h[:2]) for h in dev_env.hazards]
            _, _, done, _info, obs = dev_env.step(a_vec)
            moved = (dev_env.agent_x, dev_env.agent_y) != pos_before
            drifted = [tuple(h[:2]) for h in dev_env.hazards] != haz_before

            if z_raw_prev is not None and a_prev is not None and moved and not drifted:
                sample = (z_raw_prev.cpu(), a_prev.cpu(), (z_raw_cur - z_raw_prev).cpu())
                (test_pairs if rng.random() < REAF_TEST_FRAC else train_pairs).append(sample)

            z_raw_prev, a_prev = z_raw_cur, a_vec
            if done:
                break

        # gradient steps on the accumulated buffer
        if len(train_pairs) >= REAF_BATCH:
            for _ in range(REAF_STEPS_PER_EP):
                idx = [rng.randrange(len(train_pairs)) for _ in range(REAF_BATCH)]
                zb = torch.cat([train_pairs[i][0] for i in idx]).to(dev)
                ab = torch.cat([train_pairs[i][1] for i in idx]).to(dev)
                tb = torch.cat([train_pairs[i][2] for i in idx]).to(dev)
                loss = F.mse_loss(predictor(zb, ab), tb)
                if loss.requires_grad:
                    reaf_opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(predictor.parameters(), 0.5)
                    reaf_opt.step()

        if (ep + 1) % 10 == 0:
            print(
                f"  [train] p0 seed={seed} arm={arm} ep {ep + 1}/{total_eps} "
                f"train_n={len(train_pairs)} test_n={len(test_pairs)}",
                flush=True,
            )

    # held-out R2 of the residualization mechanism (readiness R1)
    reaf_test_r2 = 0.0
    if len(test_pairs) >= 16:
        with torch.no_grad():
            zb = torch.cat([p[0] for p in test_pairs]).to(dev)
            ab = torch.cat([p[1] for p in test_pairs]).to(dev)
            tb = torch.cat([p[2] for p in test_pairs]).to(dev)
            pred = predictor(zb, ab)
            ss_res = float(((tb - pred) ** 2).sum().item())
            ss_tot = float(((tb - tb.mean(0, keepdim=True)) ** 2).sum().item())
        reaf_test_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    # ---------------- ARM MANIPULATION (applied after P0, before P2) --------
    trained_predictor = predictor          # retained OUT of band for analysis
    for p in trained_predictor.parameters():
        p.requires_grad_(False)
    if arm == "ARM_RESID_ON":
        pass                                # correction stays live
    elif arm == "ARM_RESID_OFF":
        agent.latent_stack.reafference_predictor = None      # MECH-221 failure
    elif arm == "ARM_SHAM_RESID":
        agent.latent_stack.reafference_predictor = ShamReafference(
            trained_predictor, action_dim, seed
        )
    else:
        raise ValueError("unknown arm: " + str(arm))

    # ---------------- P2: measurement (no gradients anywhere) ---------------
    agent.eval()
    dz_down: List[torch.Tensor] = []
    dz_obs: List[torch.Tensor] = []
    drift_lbl: List[int] = []
    ep_ids: List[int] = []
    cos_vals: List[float] = []
    foot_vals: List[float] = []
    n_moved = 0
    n_obs_steps = 0

    for ep in range(n_eval):
        _, obs = dev_env.reset()
        agent.reset()
        z_down_prev: Optional[torch.Tensor] = None
        z_raw_prev = None
        a_prev = None
        obs_prev: Optional[torch.Tensor] = None

        for _ in range(n_steps):
            with torch.no_grad():
                lat = agent.sense(obs["body_state"], obs["world_state"])
            agent.clock.advance()
            z_down_cur = lat.z_world.detach()
            z_raw_cur = lat.z_world_raw.detach() if lat.z_world_raw is not None else z_down_cur
            obs_cur = torch.as_tensor(obs["world_state"]).detach().float().flatten()

            a_i = rng.randrange(action_dim)
            a_vec = _onehot(a_i, action_dim, dev)
            agent._last_action = a_vec

            pos_before = (dev_env.agent_x, dev_env.agent_y)
            haz_before = [tuple(h[:2]) for h in dev_env.hazards]
            _, _, done, _info, obs = dev_env.step(a_vec)
            moved = (dev_env.agent_x, dev_env.agent_y) != pos_before
            drifted = [tuple(h[:2]) for h in dev_env.hazards] != haz_before
            n_moved += int(moved)

            # A measurement row needs the PREVIOUS step's raw latent + action to
            # form the efference copy dz_hat, and the previous downstream latent
            # to form dz_down. Both available from step 2 of each episode on.
            if z_down_prev is not None and z_raw_prev is not None and a_prev is not None:
                with torch.no_grad():
                    dz_hat = trained_predictor(z_raw_prev, a_prev).detach()
                d_down = (z_down_cur - z_down_prev)
                c = _cos(d_down, dz_hat)
                dz_down.append(d_down.cpu())
                dz_obs.append((obs_cur - obs_prev).cpu())
                drift_lbl.append(int(drifted))
                ep_ids.append(ep)
                cos_vals.append(c)
                foot_vals.append(c * c)          # share of ||dz_down||^2 on dz_hat
                n_obs_steps += 1

            z_down_prev, z_raw_prev, a_prev, obs_prev = z_down_cur, z_raw_cur, a_vec, obs_cur
            if done:
                break

        if (ep + 1) % 10 == 0:
            print(
                f"  [train] p2 seed={seed} arm={arm} ep {n_p0 + ep + 1}/{total_eps} "
                f"rows={n_obs_steps} drift={sum(drift_lbl)}",
                flush=True,
            )

    n_drift = int(sum(drift_lbl))
    base_rate = float(n_drift / len(drift_lbl)) if drift_lbl else 0.0

    drift_auc, n_tr, n_te = _probe_auc(dz_down, drift_lbl, ep_ids, seed)
    control_auc, _, _ = _probe_auc(dz_obs, drift_lbl, ep_ids, seed)

    drift_cos = [c for c, y in zip(cos_vals, drift_lbl) if y == 1]
    fsar = float(sum(1 for c in drift_cos if c > FSAR_COS_TAU) / len(drift_cos)) if drift_cos else 0.0
    self_footprint_frac = float(sum(foot_vals) / len(foot_vals)) if foot_vals else 0.0

    row: Dict[str, Any] = {
        "arm": arm,
        "seed": seed,
        "drift_auc": drift_auc,
        "drift_auc_control": control_auc,
        "fsar": fsar,
        "self_footprint_frac": self_footprint_frac,
        "reaf_test_r2": reaf_test_r2,
        "drift_base_rate": base_rate,
        "n_rows": len(drift_lbl),
        "n_drift_rows": n_drift,
        "n_probe_train": n_tr,
        "n_probe_test": n_te,
        "n_reaf_train": len(train_pairs),
        "n_reaf_test": len(test_pairs),
        "moved_frac": float(n_moved / max(1, n_eval * n_steps)),
        "mean_cos_all": float(sum(cos_vals) / len(cos_vals)) if cos_vals else 0.0,
        "mean_cos_drift": float(sum(drift_cos) / len(drift_cos)) if drift_cos else 0.0,
    }
    return row


# ---------------------------------------------------------------- aggregate
def _mean(vals: List[Optional[float]]) -> Optional[float]:
    ok = [v for v in vals if v is not None]
    return float(sum(ok) / len(ok)) if ok else None


def _by_arm(rows: List[Dict[str, Any]], arm: str, key: str) -> List[Optional[float]]:
    return [r[key] for r in rows if r["arm"] == arm]


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    seeds = SEEDS[:1] if dry_run else SEEDS
    n_p0 = 3 if dry_run else P0_EPISODES
    n_eval = 3 if dry_run else EVAL_EPISODES
    n_steps = 20 if dry_run else STEPS_PER_EPISODE

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
                row = _run_cell(arm, seed, n_p0, n_eval, n_steps)
                cell.stamp(row)
            rows.append(row)
            print(f"verdict: {'PASS' if row.get('drift_auc') is not None else 'FAIL'}", flush=True)

    # ---- aggregates -------------------------------------------------------
    auc_on = _mean(_by_arm(rows, "ARM_RESID_ON", "drift_auc"))
    auc_off = _mean(_by_arm(rows, "ARM_RESID_OFF", "drift_auc"))
    auc_sham = _mean(_by_arm(rows, "ARM_SHAM_RESID", "drift_auc"))
    fsar_on = _mean(_by_arm(rows, "ARM_RESID_ON", "fsar"))
    fsar_off = _mean(_by_arm(rows, "ARM_RESID_OFF", "fsar"))
    foot_on = _mean(_by_arm(rows, "ARM_RESID_ON", "self_footprint_frac"))
    foot_off = _mean(_by_arm(rows, "ARM_RESID_OFF", "self_footprint_frac"))
    foot_sham = _mean(_by_arm(rows, "ARM_SHAM_RESID", "self_footprint_frac"))
    reaf_r2 = _mean([r["reaf_test_r2"] for r in rows])
    control_auc = _mean([r["drift_auc_control"] for r in rows])
    base_rate = _mean([r["drift_base_rate"] for r in rows])
    min_drift_rows = min([r["n_drift_rows"] for r in rows]) if rows else 0

    def _gap(a: Optional[float], b: Optional[float]) -> Optional[float]:
        return None if (a is None or b is None) else float(a - b)

    c1_measured = _gap(auc_on, auc_off)
    c2_measured = _gap(fsar_off, fsar_on)
    c3_measured = _gap(auc_on, auc_sham)

    # ---- readiness preconditions -----------------------------------------
    # WORST-CELL reporting where met is a worst-case claim (skill Step 3):
    # reaf_test_r2 and control AUC are floors that must hold for EVERY cell.
    worst_r2 = min([r["reaf_test_r2"] for r in rows]) if rows else 0.0
    worst_r2_cell = min(rows, key=lambda r: r["reaf_test_r2"]) if rows else None
    ctrl_vals = [r["drift_auc_control"] for r in rows if r["drift_auc_control"] is not None]
    worst_ctrl = min(ctrl_vals) if ctrl_vals else 0.0
    worst_ctrl_cell = (
        min([r for r in rows if r["drift_auc_control"] is not None],
            key=lambda r: r["drift_auc_control"])
        if ctrl_vals else None
    )

    precondition_specs = [
        {
            "name": "reafference_predictor_test_r2_supra_floor",
            "kind": "readiness",
            "measured": float(worst_r2),
            "threshold": R1_REAF_TEST_R2_FLOOR,
            "direction": "lower",
            "control": "held-out R2 of the trained ReafferencePredictor on pure "
                       "self-caused perspective-shift transitions (agent moved, no "
                       "exogenous drift) -- the positive control that the "
                       "residualization mechanism predicts anything at all",
            "offending_cell": (
                f"{worst_r2_cell['arm']}::seed{worst_r2_cell['seed']}" if worst_r2_cell else None
            ),
        },
        {
            "name": "drift_label_detectable_from_raw_obs_control_auc",
            "kind": "readiness",
            "measured": float(worst_ctrl),
            "threshold": R2_CONTROL_AUC_FLOOR,
            "direction": "lower",
            "control": "SAME STATISTIC as C1 (AUC of the same held-out linear probe "
                       "class) measured on the RAW world-observation delta, where the "
                       "exogenous event definitionally is. Below floor means the label "
                       "is unlearnable and C1 is STARVED, not falsified.",
            "offending_cell": (
                f"{worst_ctrl_cell['arm']}::seed{worst_ctrl_cell['seed']}" if worst_ctrl_cell else None
            ),
        },
        {
            "name": "drift_base_rate_in_band",
            "kind": "readiness",
            "measured": float(base_rate if base_rate is not None else 0.0),
            "threshold_low": R3_BASE_RATE_LOW,
            "threshold_high": R3_BASE_RATE_HIGH,
            "direction": "interval",
            "control": "ground-truth exogenous-event rate (a hazard actually changed "
                       "cell), snapshot-compared around env.step",
        },
        {
            "name": "drift_auc_off_headroom_for_c1",
            "kind": "dv_headroom",
            "measured": float(auc_off if auc_off is not None else 0.0),
            "threshold": R4_OFF_AUC_CEILING,
            "direction": "upper",
            "control": "C1 requires drift_auc_on - drift_auc_off >= 0.05 and AUC is "
                       "bounded above by 1.0; an OFF arm above 0.90 leaves under 0.10 "
                       "of headroom and C1 could fire negative BY CONSTRUCTION",
        },
    ]

    try:
        preconditions = p0_readiness_gate(precondition_specs)
        readiness_met = all(bool(p.get("met")) for p in preconditions)
    except P0NotReady as exc:
        preconditions = exc.preconditions
        readiness_met = False

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
            "detail": "mean drift-detection AUC (held-out linear probe on downstream "
                      "dz_world), ARM_RESID_ON minus ARM_RESID_OFF",
        },
        {
            "name": "C2_contamination_raises_false_self_confirmation",
            "load_bearing": True,
            "passed": bool(c2),
            "measured": c2_measured,
            "threshold": C2_FSAR_GAP,
            "detail": f"fraction of GROUND-TRUTH exogenous steps whose downstream "
                      f"dz_world reads as confirming the agent's own action "
                      f"(cos(dz_world, dz_hat) > {FSAR_COS_TAU}), OFF minus ON",
        },
        {
            "name": "C3_effect_attributable_to_correct_residualization",
            "load_bearing": True,
            "passed": bool(c3),
            "measured": c3_measured,
            "threshold": C3_SHAM_GAP,
            "detail": "ARM_RESID_ON minus ARM_SHAM_RESID drift AUC. SHAM subtracts a "
                      "same-magnitude mis-paired-action vector, so this isolates "
                      "correct self-content from 'subtracting any vector of that size'",
        },
        {
            "name": "MANIPULATION_CHECK_self_footprint_removed",
            "load_bearing": False,
            "passed": bool(
                foot_off is not None and foot_on is not None and foot_off > foot_on
            ),
            "measured": _gap(foot_off, foot_on),
            "threshold": 0.0,
            "threshold_not_applicable": "NOT load-bearing and NEAR-TAUTOLOGICAL: the ON "
                                        "arm subtracts exactly dz_hat, so this is close "
                                        "to the predictor's own R2 restated. Recorded as "
                                        "evidence the manipulation reached the DV, never "
                                        "as evidence for the claim.",
            "detail": "mean cos^2(dz_world, dz_hat), OFF minus ON",
        },
    ]

    combination_rule = "C1 AND C2 AND C3 (all three load-bearing criteria must hold)"
    overall_pass = bool(readiness_met and c1 and c2 and c3)

    # ---- non-degeneracy ---------------------------------------------------
    arms_identical = False
    if auc_on is not None and auc_off is not None and auc_sham is not None:
        arms_identical = (
            abs(auc_on - auc_off) < 1e-9 and abs(auc_on - auc_sham) < 1e-9
        )
    enough_drift = min_drift_rows >= (4 if dry_run else MIN_DRIFT_SAMPLES)
    criteria_non_degenerate = {
        "C1_residualization_improves_exogenous_event_visibility": bool(
            readiness_met and not arms_identical and enough_drift
        ),
        "C2_contamination_raises_false_self_confirmation": bool(
            readiness_met and enough_drift
        ),
        "C3_effect_attributable_to_correct_residualization": bool(
            readiness_met and not arms_identical and enough_drift
        ),
    }

    # ---- interpretation self-route ---------------------------------------
    if not readiness_met:
        label = "substrate_not_ready_requeue"
        direction = "non_contributory"
    elif not enough_drift or arms_identical:
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
            "drift_auc": [r["drift_auc"] for r in rows if r["drift_auc"] is not None],
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
        "non_degenerate": bool(degeneracy.get("non_degenerate", True))
        and not arms_identical
        and enough_drift,
        "degeneracy_reason": degeneracy.get("degeneracy_reason"),
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
        },
        "arm_results": rows,
        "aggregates_by_arm": {
            "drift_auc": {"ARM_RESID_ON": auc_on, "ARM_RESID_OFF": auc_off,
                          "ARM_SHAM_RESID": auc_sham},
            "fsar": {"ARM_RESID_ON": fsar_on, "ARM_RESID_OFF": fsar_off,
                     "ARM_SHAM_RESID": _mean(_by_arm(rows, "ARM_SHAM_RESID", "fsar"))},
            "self_footprint_frac": {"ARM_RESID_ON": foot_on, "ARM_RESID_OFF": foot_off,
                                    "ARM_SHAM_RESID": foot_sham},
        },
        "per_seed_rows": rows,
        "label_balance": {
            "drift_base_rate": base_rate,
            "min_drift_rows_per_cell": min_drift_rows,
            "n_rows_per_cell": [r["n_rows"] for r in rows],
        },
        "custom_information": {
            "scoping_note": "Random-action policy in both phases: this measures the "
                            "REPRESENTATIONAL half of MECH-222 (contamination of "
                            "z_world and loss of exogenous-event visibility), not the "
                            "behavioural/clinical half. A PASS does not license the "
                            "referential-delusion reading directly.",
            "z_world_raw_note": "z_world_raw is PRE-EMA as well as pre-correction "
                                "(Step 2.5a probe, 2026-09-10), so it is not a valid "
                                "within-run uncorrected counterpart to z_world.",
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
            "drift_auc_on": auc_on,
            "drift_auc_off": auc_off,
            "drift_auc_sham": auc_sham,
            "drift_auc_control": control_auc,
            "c1_auc_gap_on_minus_off": c1_measured,
            "c2_fsar_gap_off_minus_on": c2_measured,
            "c3_auc_gap_on_minus_sham": c3_measured,
            "fsar_on": fsar_on,
            "fsar_off": fsar_off,
            "self_footprint_frac_on": foot_on,
            "self_footprint_frac_off": foot_off,
            "reaf_test_r2": reaf_r2,
            "drift_base_rate": base_rate,
            "min_drift_rows_per_cell": min_drift_rows,
            "c1_passed": int(bool(c1)),
            "c2_passed": int(bool(c2)),
            "c3_passed": int(bool(c3)),
            "readiness_met": int(bool(readiness_met)),
            "overall_pass": int(bool(overall_pass)),
        }
    )
    manifest["_elapsed"] = time.perf_counter() - t0
    manifest["_t0"] = t0
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    result = run_experiment(dry_run=args.dry_run)
    t0 = result.pop("_t0")
    result.pop("_elapsed", None)

    out_path = write_flat_manifest(
        result,
        dry_run=args.dry_run,
        config=_config_slice(),
        seeds=SEEDS[:1] if args.dry_run else SEEDS,
        script_path=Path(__file__),
        started_at=t0,
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
