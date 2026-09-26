"""V3-EXQ-1108: probe N2 at the W3 dose (post 1200) on the fleet -- does W3's retained replay keep the
L2R bar while W6a trains the world encoder through the read path? ree_core PINNED to 9b322d5.

red-team (fable): CONTESTED -- F1 leg (e) fails on W6a-ON twins too (secondary keeps_bar_excluding_e + reading rule
added, (e) still gates); F2 missing twin now INCOMPLETE (fixed); F3 (b) sign flip if pre < B0 (recorded + precondition).
Amendment A1 pre-registered in the N2 record, REE_assembly fc5c0795221.

WHAT THIS IS
============
The coupled-loop-repair campaign's probe N2 (REE_assembly/evidence/planning/
coupled_loop_repair_campaign_plan.md sec 4 N2 row; P8; Q4c), re-run under a PRE-REGISTERED
AMENDMENT of its own record, REE_assembly/evidence/planning/n2_replay_encoder_probe_20260925.md
(pre-registration 025fc6a5cfe; first result bbccb5d1416 = CANNOT_DETERMINE because the pre-registered
dose cut to post 600 removed the second half of W3's on-policy phase, where the control arm banks its
gain). The amendment (sec 4 of that record, committed BEFORE this entry was queued) changes exactly:
  dose   post 600 -> post 1200 (= the W3 member-gate dose; 9600 W3 updates, 1200 W6a updates)
  seeds  611-615 -> 811-815 (fresh: not 106-110, 531-535, 611-615, 721-725)
  where  Mac probe -> this fleet driver (cross-machine-class: numbers are not bit-comparable to the
         Mac runs because torch.multinomial differs linux vs darwin; see CLAUDE.md)
Everything else -- arms, twins, readouts, gate legs (a)-(e), the per-arm keep-the-bar rule, the decision
rule and its evaluation order, the secondary readouts S1-S4 -- is the pre-registration, unchanged.

ARMS (configuration only, one pinned sha):
  frozen   : W6a OFF                                   (= the W3 member-gate configuration; CONTROL)
  reencode : W6a ON  + W3 replay_latent "reencode"     (raw obs re-encoded at replay)
  stored   : W6a ON  + W3 replay_latent "stored"       (the z sensed at record time)
TWIN per arm: shuf = retained babbling actions relabelled by W3's FIXED class permutation [1,2,3,4,0].
B0 per seed: the babbling probe's own on-policy-data head (12 native episodes, 3000 head updates).

PER-RUN GATE LEGS (read in the CURRENT, end-of-run encoder space):
  (a) disc4_h1 >= 0.47 and k == 10     (b) retention (post_cur - B0)/(pre - B0) >= 0.5
  (c) the arm's shuf twin does NOT meet (a)     (d) guard PASS on the W3 group (real and shuf)
  (e) bounded rollout: max late (steps 21-30) per-step norm growth < 1.2 and median t30/t0 < 5
ARM KEEPS THE BAR iff (a) >= 4/5, (b) >= 4/5, twin meets (a) <= 1/5, (d) 5/5, (e) >= 4/5.
DECISION (evaluated in order): CANNOT_DETERMINE if the control misses (a) on >= 2 seeds;
HOLDS-REENCODE if reencode keeps the bar (HOLDS-BOTH if stored also does); HOLDS-STORED-ONLY if only
stored does; NEITHER otherwise. (INCOMPLETE if a run crashed and the control alone does not decide.)
OUTCOME: PASS iff the verdict is HOLDS-*; FAIL otherwise. EXPERIMENT_PURPOSE = diagnostic, no claim
pressed (it gates the campaign's W6 preset buffer policy for W3 / harm_eval / codec / E1).
Evidence domain D1 (a member-trained head's discrimination in the space it is consumed in); no E3
consumer or behaviour reading.

SUBSTRATE PIN (experiments/_lib/substrate_pin.py)
=================================================
ree_core/ is executed from 9b322d5c8e (tag archive/coupled-loop-repair-9b322d5, the W6a commit on
integration/coupled-loop-repair), extracted read-only with git archive; experiments/** (the harness,
_lib, pack_writer, infant_curriculum) comes from the LIVE checkout (main). experiments/_harness.py,
experiments/_lib/** and experiments/infant_curriculum.py are byte-identical between 9b322d5 and the
main this driver was authored on (7a37d1d), and main's ree_core is unchanged since the branch's
merge-base 07b5fe6, so the pinned ree_core + live experiments/** is exactly the tree the Mac probe ran.
Resolution: the full sha is resolved locally (workers fetch +refs/heads/*, and the tag auto-follows);
if it does not resolve, the driver fetches that one tag from origin into refs/tags and retries; if it
still does not resolve, SubstratePinError -> ERROR (never a silent run on main). verify_pin() checks
ree_core.__file__ is under the pin dir and that ree_core.utils.waking_trainer_world_encoder
.WorldEncoderMember exists (it does not exist on main). Every run is a separate child process that
re-pins, exactly as the Mac probe ran one process per (seed, arm, twin). Pinned cells are
reuse-INELIGIBLE (pin_fingerprint_kwargs).

PROTOCOL (verbatim from the Mac probe REE_assembly/evidence/planning/probes/n2/n2_probe.py, whose
helpers came from probes/babble/babble_probe.py, probes/rollout/rollout_fidelity_probe.py and
probes/rollout/balanced_replay_probe.py; vendored below function-for-function):
babbling probe env (CausalGridWorldV2 size 12, Phase-0 kwargs, new env per 200-step episode); build_B
agent (world_dim 32 = deployed, alpha_world 0.3 = from_dims default); held-out test set 3000 uniform
random {0..3} steps (k = 120..134); W2a StructuredBabbler 2400 steps into the FROZEN retained set;
3000 W3 member updates (pre); post phase 1200 native closed-loop StepHarness steps (6 episodes,
k = 50..55), 8 W3 updates per step at the 25% retained mix, W6a (from_dims defaults, 1 update per step,
registered after W3) trains only in the post phase. Canary (dry-run only, darwin only): seed 106 frozen
p1200 must reproduce W3's published s106 row (pre 0.4333/k9, post 0.5233/k10, retention 1.711;
B0 0.3067) -- that is the vendoring check.

PROGRESS: one "Seed S Condition <cond>" + "[train] ... ep 1/1" + "verdict:" triple per run as each run
finishes (35 runs = 5 seeds x 7 conditions: B0 + 3 arms x 2 twins). "verdict: PASS" on a run line means
the run completed and wrote its record, not that it met a gate. Runs of different seeds execute in
parallel child processes when the box has the cores (N2_JOBS, default min(4, cpu_count // 2)); each
child sets torch threads = 2, so parallelism does not change any number.

No sleep is enabled (no SLEEP DRIVER line needed). ASCII output only.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# --- SUBSTRATE PIN -- MUST run before the first `import ree_core` ----------------------------------
from experiments._lib.substrate_pin import (  # noqa: E402
    SubstratePinError, pin_ree_core, verify_pin, pin_fingerprint_kwargs, pin_manifest_block,
)

SUBSTRATE_PIN_REF = "9b322d5c8e3f4efaf1a4530ebae2e7f3ef2e3055"
SUBSTRATE_PIN_TAG = "archive/coupled-loop-repair-9b322d5"
SUBSTRATE_PIN_MARKER_MODULE = "ree_core.utils.waking_trainer_world_encoder"
SUBSTRATE_PIN_MARKER_ATTR = "WorldEncoderMember"
SUBSTRATE_PIN_MARKER_EXPECTED_PRESENT = True


def _ensure_pin_resolvable() -> Dict[str, Any]:
    """Resolve the pin sha locally; if absent, fetch ONLY its archive tag from origin, then re-check."""
    def _rp() -> bool:
        r = subprocess.run(["git", "-C", str(_REPO_ROOT), "rev-parse", "--verify", "-q",
                            SUBSTRATE_PIN_REF + "^{commit}"], capture_output=True, text=True, timeout=60)
        return r.returncode == 0
    info = {"resolved_locally": _rp(), "fetched_tag": False, "fetch_rc": None}
    if not info["resolved_locally"]:
        f = subprocess.run(["git", "-C", str(_REPO_ROOT), "fetch", "-q", "origin",
                            "+refs/tags/%s:refs/tags/%s" % (SUBSTRATE_PIN_TAG, SUBSTRATE_PIN_TAG)],
                           capture_output=True, text=True, timeout=300)
        info["fetched_tag"] = True
        info["fetch_rc"] = f.returncode
        if not _rp():
            raise SubstratePinError("pin %s unresolvable even after fetching tag %s (rc %s): %s" % (
                SUBSTRATE_PIN_REF, SUBSTRATE_PIN_TAG, f.returncode, (f.stderr or "").strip()))
    return info


_PIN_FETCH = _ensure_pin_resolvable()
_PIN = pin_ree_core(SUBSTRATE_PIN_REF)
verify_pin(_PIN, marker_module=SUBSTRATE_PIN_MARKER_MODULE, marker_attr=SUBSTRATE_PIN_MARKER_ATTR,
           marker_expected_present=SUBSTRATE_PIN_MARKER_EXPECTED_PRESENT)
_PIN["fetch_fallback"] = _PIN_FETCH

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.run_id import make_run_id  # noqa: E402
from experiments._lib.arm_fingerprint import compute_arm_fingerprint, reset_all_rng  # noqa: E402
from experiments._harness import StepHarness  # noqa: E402
from experiments.infant_curriculum import InfantCurriculumScheduler  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.utils import waking_trainer as WT  # noqa: E402
from ree_core.utils import waking_trainer_world_encoder as WE  # noqa: E402
from ree_core.developmental.structured_babbling import StructuredBabbler  # noqa: E402

for _mod in (WT, WE):
    assert str(Path(_mod.__file__).resolve()).startswith(str(Path(_PIN["pin_dir"]).resolve()) + os.sep), \
        "%s did not resolve from the pin dir" % _mod.__file__

torch.set_num_threads(2)

EXPERIMENT_TYPE = "v3_exq_1108_n2_replay_encoder_full_dose_pinned"
QUEUE_ID = "V3-EXQ-1108"
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS: List[str] = []

# ---- pre-registered constants (N2 record secs 2 + 4 amendment) -----------------------------------
SEEDS = [811, 812, 813, 814, 815]
POST = 1200                      # amendment: 600 -> 1200 (the W3 member-gate dose)
UPS = 8                          # W3 updates per post-phase waking step
W6A_UPS = 1
W6A_BATCH = 64
PRE_UPDATES = 3000
B0_N_EPS = 12                    # the published babbling runs used --n-eps 12 (N2 record P4)
ARMS = ["frozen", "reencode", "stored"]
TWINS = ["real", "shuf"]
PERM = [1, 2, 3, 4, 0]
A_DISC = 0.47
A_K = 10
RET_MIN = 0.5
E_LATE_MAX = 1.2
E_T30_MAX = 5.0
KEEP_A = 4
KEEP_B = 4
KEEP_TWIN_A_MAX = 1
KEEP_D = 5
KEEP_E = 4
CTRL_MISS_CD = 2
N_SEEDS_REQ = 5
# vendoring canary (dry-run, darwin only): W3's published seed-106 row, reproduced on the Mac in P3
CANARY = {"seed": 106, "post": 1200, "pre_disc4": 0.4333333333333333, "pre_k": 9,
          "post_disc4": 0.5233333333333333, "post_k": 10, "retention": 1.711, "B0": 0.30666666666666664}
DRY = {"seeds": [811], "post": 200}

# ================================ vendored probe helpers ===========================================
# babble_probe.py (REE_assembly probes/babble, bt0925 babble): constants + make_env ... fresh_agent
GRID = 12                        # v3_exq_591 GRID_SIZE
EP_STEPS = 200
SEED_STRIDE = 160
A_ENV = 5
CLASSES = [0, 1, 2, 3]
PH0_KW = InfantCurriculumScheduler(grid_size=GRID).env_kwargs(0)


def seed_all(s):                 # rollout_fidelity_probe.seed_all
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def build_B(seed, clamp):        # rollout_fidelity_probe.build_B
    env = CausalGridWorldV2(size=8, num_hazards=2, num_resources=3, max_episode_steps=200, seed=seed)
    kw = dict(body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
              action_dim=env.action_dim, use_sleep_aggregation_cluster=True,
              use_cross_module_consolidation=True, use_sleep_residue_integration=True,
              use_offline_integration_gradient_step=True, sleep_loop_episodes_K=10_000_000)
    if clamp:
        kw["e2_rollout_output_norm_clamp_enabled"] = True
        kw["e2_rollout_output_norm_clamp_ratio"] = 2.0
    cfg = REEConfig.from_dims(**kw)
    agent = REEAgent(cfg).to(torch.device("cpu"))
    agent.eval()
    return env, agent, cfg


def head_params(agent):          # balanced_replay_probe.head_params
    return list(agent.e2.world_transition.parameters()) + list(agent.e2.world_action_encoder.parameters())


def get_head(agent):             # balanced_replay_probe.get_head
    return {"t": copy.deepcopy(agent.e2.world_transition.state_dict()),
            "a": copy.deepcopy(agent.e2.world_action_encoder.state_dict())}


def set_head(agent, h):          # balanced_replay_probe.set_head
    agent.e2.world_transition.load_state_dict(h["t"])
    agent.e2.world_action_encoder.load_state_dict(h["a"])


def make_env(seed, k):
    return CausalGridWorldV2(size=GRID, seed=seed * SEED_STRIDE + k, resource_respawn_on_consume=True,
                             pos_telemetry_enabled=True, traj_telemetry_enabled=True, **PH0_KW)


def snap_obs(od):
    keep = {}
    for key in ("body_state", "world_state", "harm_obs", "harm_obs_a", "harm_history"):
        v = od.get(key)
        if v is not None:
            keep[key] = torch.as_tensor(v).detach().clone().float()
    return keep


def gen_policy(seed, n_eps, k0, policy):
    segs, early, rew = [], 0, []
    for ep in range(n_eps):
        env = make_env(seed, k0 + ep)
        _f, od = env.reset()
        seg = {"obs": [snap_obs(od)], "a": []}
        for _s in range(EP_STEPS):
            ai = policy()
            _o, harm, done, info, od = env.step(ai)
            rew.append(float(harm))
            seg["a"].append(ai); seg["obs"].append(snap_obs(od))
            if done:
                early += 1
                segs.append(seg)
                _f, od = env.reset()
                seg = {"obs": [snap_obs(od)], "a": []}
        segs.append(seg)
    return segs, {"early_terminations": early}


def pol_uniform(seed):
    g = np.random.default_rng(seed)
    return lambda: int(g.integers(0, 4))


def gen_POL(agent, seed, n_eps, k0):
    segs, early, rew = [], 0, []
    for ep in range(n_eps):
        env = make_env(seed, k0 + ep)
        h = StepHarness(agent, env, train_mode=False, seed=seed * 1000 + k0 + ep)
        _f, od = env.reset(); agent.reset(); h.reset()
        seg = {"obs": [snap_obs(od)], "a": [], "z": []}
        for _s in range(EP_STEPS):
            r = h.step(od)
            rew.append(float(r.harm_signal))
            seg["a"].append(int(r.action.detach().reshape(-1).argmax()))
            od = r.next_obs_dict
            seg["obs"].append(snap_obs(od))
            if r.done:
                early += 1
                segs.append(seg)
                _f, od = env.reset(); agent.reset(); h.reset()
                seg = {"obs": [snap_obs(od)], "a": [], "z": []}
        segs.append(seg)
    rew = np.asarray(rew)
    n = len(rew)
    return segs, {"early_terminations": early, "early_per_1000": early * 1000.0 / n,
                  "harm_events_per_100": float((rew < 0).sum() * 100.0 / n),
                  "benefit_events_per_100": float((rew > 0).sum() * 100.0 / n),
                  "reward_per_100": float(rew.sum() * 100.0 / n)}


@torch.no_grad()
def encode_segs(ref, segs):
    out = []
    for s in segs:
        if len(s["a"]) < 1:
            continue
        ref.reset()
        zs = []
        for o in s["obs"]:
            lat = ref.sense(o["body_state"], o["world_state"], obs_harm=o.get("harm_obs"),
                            obs_harm_a=o.get("harm_obs_a"), obs_harm_history=o.get("harm_history"))
            zs.append(lat.z_world.detach().reshape(-1).clone())
        raw = torch.stack([o["world_state"].reshape(-1) for o in s["obs"]])
        out.append({"z": torch.stack(zs), "raw": raw, "a": torch.tensor(s["a"])})
    return out


def to_trans(eps, key):
    X0_, X1_, A_ = [], [], []
    for e in eps:
        x = e[key]
        X0_.append(x[:-1]); X1_.append(x[1:]); A_.append(e["a"])
    return torch.cat(X0_), torch.cat(X1_), torch.cat(A_)


def train_head(ref, init, trans, steps, seed):
    set_head(ref, init)
    torch.manual_seed(seed)
    x0, x1, a = trans
    oh = F.one_hot(a, A_ENV).float()
    params = head_params(ref)
    opt = torch.optim.Adam(params, lr=3e-4)
    before = [p.detach().clone() for p in params]
    losses, grad_ok = [], None
    n = x0.shape[0]
    with torch.enable_grad():
        for i in range(steps):
            idx = torch.randint(0, n, (32,))
            zb, ab, yb = x0[idx], oh[idx], x1[idx]
            loss = F.mse_loss(ref.e2.world_forward(zb, ab), yb)
            opt.zero_grad(); loss.backward()
            if i == 0:
                grad_ok = all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in params)
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            losses.append(float(loss.detach()))
    delta = float(torch.sqrt(sum(((p.detach() - b) ** 2).sum() for p, b in zip(params, before))))
    return get_head(ref), {"loss_first50": float(np.mean(losses[:50])), "loss_last200": float(np.mean(losses[-200:])),
                           "identity_mse": float(F.mse_loss(x0, x1)), "grad_nonnull_step1": grad_ok,
                           "param_delta": delta, "n_trans": int(n)}


@torch.no_grad()
def evaluate(ref, head, te, key, seed, H=10, max_starts=300):
    set_head(ref, head)
    e2 = ref.e2
    g = np.random.default_rng(seed + 77)
    starts = [(i, t) for i, e in enumerate(te) for t in range(e["a"].shape[0] - H + 1)]
    if len(starts) > max_starts:
        starts = [starts[j] for j in sorted(g.choice(len(starts), max_starts, replace=False))]
    zs = torch.zeros(1, 32)
    rows = {h: {"err": [], "pers": []} for h in range(1, H + 1)}
    disc4 = {1: [], 3: [], 5: []}
    disc5 = {1: []}
    for i, t in starts:
        x = te[i][key]
        acts = F.one_hot(te[i]["a"][t:t + H], A_ENV).float().unsqueeze(0)
        x0 = x[t:t + 1]
        tr = e2.rollout_with_world(zs, x0, acts, compute_action_objects=False)
        for h in range(1, H + 1):
            y = x[t + h:t + h + 1]
            rows[h]["err"].append(float((tr.world_states[h] - y).norm()))
            rows[h]["pers"].append(float((x0 - y).norm()))
        ex = int(te[i]["a"][t])
        for h in (1, 3, 5):
            y = x[t + h:t + h + 1]
            errs = []
            for c in range(A_ENV):
                a2 = acts[:, :h].clone(); a2[0, 0] = 0.0; a2[0, 0, c] = 1.0
                errs.append(float((e2.rollout_with_world(zs, x0, a2, compute_action_objects=False).world_states[h] - y).norm()))
            e4 = [errs[c] for c in CLASSES]
            disc4[h].append(int(np.argmin(e4)) == CLASSES.index(ex))
            if h == 1:
                disc5[1].append(int(np.argmin(errs)) == ex)
    k = 0
    eop = {}
    for h in range(1, H + 1):
        me, mp = float(np.median(rows[h]["err"])), float(np.median(rows[h]["pers"]))
        eop[h] = me / mp if mp > 0 else None
        if mp > 0 and me < mp and k == h - 1:
            k = h
    return {"n_starts": len(starts), "disc4_h1": float(np.mean(disc4[1])), "disc4_h3": float(np.mean(disc4[3])),
            "disc4_h5": float(np.mean(disc4[5])), "disc5_h1": float(np.mean(disc5[1])), "k": k,
            "err_over_pers_h1": eop[1], "err_over_pers_h5": eop[5]}


def fresh_agent(seed, ref_enc):
    seed_all(seed)
    _e, agent, _c = build_B(seed, False)
    enc = agent.latent_stack.state_dict()
    for kk, v in ref_enc.items():
        assert torch.equal(enc[kk], v), "encoder mismatch vs reference"
    agent.eval()
    return agent


# n2_probe.py helpers --------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_sd(ref, head, te, key, seed, sd=None, H=10, max_starts=300):
    """evaluate() with an optional per-dimension standardisation of the error metric
    (pred and target both divided by sd).  sd=None reproduces evaluate() exactly."""
    set_head(ref, head)
    e2 = ref.e2
    g = np.random.default_rng(seed + 77)
    starts = [(i, t) for i, e in enumerate(te) for t in range(e["a"].shape[0] - H + 1)]
    if len(starts) > max_starts:
        starts = [starts[j] for j in sorted(g.choice(len(starts), max_starts, replace=False))]
    zs = torch.zeros(1, 32)
    w = torch.ones(1, 32) if sd is None else (1.0 / sd.clamp_min(1e-6)).reshape(1, -1)
    rows = {h: {"err": [], "pers": []} for h in range(1, H + 1)}
    disc4, disc5 = {1: [], 3: [], 5: []}, {1: []}
    for i, t in starts:
        x = te[i][key]
        acts = F.one_hot(te[i]["a"][t:t + H], A_ENV).float().unsqueeze(0)
        x0 = x[t:t + 1]
        tr = e2.rollout_with_world(zs, x0, acts, compute_action_objects=False)
        for h in range(1, H + 1):
            y = x[t + h:t + h + 1]
            rows[h]["err"].append(float(((tr.world_states[h] - y) * w).norm()))
            rows[h]["pers"].append(float(((x0 - y) * w).norm()))
        ex = int(te[i]["a"][t])
        for h in (1, 3, 5):
            y = x[t + h:t + h + 1]
            errs = []
            for c in range(A_ENV):
                a2 = acts[:, :h].clone(); a2[0, 0] = 0.0; a2[0, 0, c] = 1.0
                errs.append(float(((e2.rollout_with_world(zs, x0, a2, compute_action_objects=False)
                                    .world_states[h] - y) * w).norm()))
            e4 = [errs[c] for c in CLASSES]
            disc4[h].append(int(np.argmin(e4)) == CLASSES.index(ex))
            if h == 1:
                disc5[1].append(int(np.argmin(errs)) == ex)
    k, eop = 0, {}
    for h in range(1, H + 1):
        me, mp = float(np.median(rows[h]["err"])), float(np.median(rows[h]["pers"]))
        eop[h] = me / mp if mp > 0 else None
        if mp > 0 and me < mp and k == h - 1:
            k = h
    return {"n_starts": len(starts), "disc4_h1": float(np.mean(disc4[1])), "disc4_h3": float(np.mean(disc4[3])),
            "disc4_h5": float(np.mean(disc4[5])), "disc5_h1": float(np.mean(disc5[1])), "k": k,
            "err_over_pers_h1": eop[1], "err_over_pers_h5": eop[5]}


def rollout_e(ref, head, TE, seed):
    set_head(ref, head)
    g = np.random.default_rng(seed + 31)
    t30, late = [], []
    with torch.no_grad():
        for _i in range(40):
            ep = TE[int(g.integers(0, len(TE)))]
            t = int(g.integers(0, ep["z"].shape[0]))
            x0 = ep["z"][t:t + 1]
            acts = F.one_hot(torch.tensor(g.integers(0, 5, 30)), 5).float().unsqueeze(0)
            ws = ref.e2.rollout_with_world(torch.zeros(1, 32), x0, acts, compute_action_objects=False).world_states
            n = [float(w.norm()) for w in ws]
            t30.append(n[30] / max(n[0], 1e-9))
            late.append(float(np.mean([n[j] / max(n[j - 1], 1e-9) for j in range(21, 31)])))
    return {"t30_over_t0_median": float(np.median(t30)), "late_growth_median": float(np.median(late)),
            "late_growth_max": float(np.max(late)),
            "gate_e": bool(float(np.max(late)) < E_LATE_MAX and float(np.median(t30)) < E_T30_MAX)}


def pr_and_norm(TE):
    Z = torch.cat([e["z"] for e in TE])
    Zc = Z - Z.mean(0)
    ev = torch.linalg.eigvalsh(Zc.T @ Zc / max(1, Z.shape[0] - 1)).clamp_min(0)
    return {"pr": float(ev.sum() ** 2 / (ev ** 2).sum().clamp_min(1e-30)),
            "norm_median": float(Z.norm(dim=-1).median()), "sd": Z.std(0)}


# ================================ child: one (seed, arm, twin) run ====================================

def child_run(S: int, arm: Optional[str], twin: str, post: int, out: str, b0_only: bool,
              b0_cache: Optional[str]) -> None:
    """One invocation = one seed x arm x twin (or B0), exactly the Mac probe's n2_probe.py body."""
    t0 = time.time()

    def log(m):
        print("[n2 s%d %s/%s t=%4.0fs] %s" % (S, arm, twin, time.time() - t0, m), flush=True)

    # per-cell RNG reset (each run is also its own process); seed_all(S) below re-seeds identically
    reset_all_rng(S)
    # reference build + test set
    seed_all(S)
    _e, ref, _c = build_B(S, False)
    ref.eval()
    ref_enc = copy.deepcopy(ref.latent_stack.state_dict())
    init = get_head(ref)
    te_segs, _ = gen_policy(S, 3000 // EP_STEPS, 120, pol_uniform(S * 7 + 3))
    TE = encode_segs(ref, te_segs)
    ev_init = evaluate(ref, init, TE, "z", S)

    if b0_only:
        pol_agent = fresh_agent(S, ref_enc)
        seed_all(S + 300)
        pol_segs, pol_info = gen_POL(pol_agent, S, B0_N_EPS, 25)
        trapped = bool(pol_info["early_per_1000"] >= 3.0 or pol_info["harm_events_per_100"] >= 10.0)
        E_POL = encode_segs(ref, pol_segs)
        hd, tinfo = train_head(ref, init, to_trans(E_POL, "z"), PRE_UPDATES, S)
        ev_b0 = evaluate(ref, hd, TE, "z", S)
        res = {"kind": "B0", "seed": S, "hazard_class": "hazard-trapped" if trapped else "benign",
               "pol_run": pol_info, "B0": ev_b0, "B0_train": tinfo, "init": ev_init, "t_s": time.time() - t0}
        json.dump(res, open(out, "w"), indent=1, default=str)
        log("B0 disc4 %.4f k %d INIT %.4f class %s" % (ev_b0["disc4_h1"], ev_b0["k"], ev_init["disc4_h1"],
                                                       res["hazard_class"]))
        return

    b0 = json.load(open(b0_cache))
    assert abs(b0["init"]["disc4_h1"] - ev_init["disc4_h1"]) < 1e-12, "INIT mismatch vs B0 cache"
    B0 = b0["B0"]["disc4_h1"]
    log("INIT disc4 %.4f B0 %.4f (%s)" % (ev_init["disc4_h1"], B0, b0["hazard_class"]))

    agent = fresh_agent(S, ref_enc)
    cfg = agent.config
    cfg.waking_trainer_guard_min_steps = 8
    latent_mode = "stored" if arm == "stored" else "reencode"
    member = WT.E2WorldMember(agent, lr=3e-4, batch_size=32, buffer_max=2000, retained_max=5000,
                              replay_frac=0.25, reencode_window=0, replay_latent=latent_mode,
                              objective="mse", grad_clip=1.0, updates_per_step=1)
    members = [member]
    wem = None
    if arm in ("reencode", "stored"):
        wem = WE.WorldEncoderMember(agent, lr=1e-3, batch_size=W6A_BATCH, buffer_max=2000, window=0,
                                    grad_clip=1.0, updates_per_step=W6A_UPS, seed=S)
        members.append(wem)
    tr = WT.WakingTrainer(agent, cfg, members=members)
    agent.waking_trainer = tr
    woe0 = agent.world_obs_encoder[0].weight.detach().clone()
    log("members %s W3 W=%d alpha_world=%.2f latent=%s" % (list(tr.members), member.reencode_window,
                                                          cfg.latent.alpha_world, member.replay_latent))

    # babbling epoch (W6a only records here)
    bab = StructuredBabbler(n_classes=5, max_run=4, seed=S * 13 + 1)
    tr.set_e2_world_source("babble")
    tr.every_k = 10 ** 9
    counts = [0] * 5
    with torch.no_grad():
        for k in range(12):
            env = make_env(S, k)
            _f, od = env.reset(); agent.reset(); bab.reset()
            for _s in range(EP_STEPS):
                agent.sense(od["body_state"], od["world_state"], obs_harm=od.get("harm_obs"),
                            obs_harm_a=od.get("harm_obs_a"), obs_harm_history=od.get("harm_history"))
                act = bab.next_action()
                c = int(act.argmax()); counts[c] += 1
                agent.record_executed_action(act)
                _f, h, done, _i, od = env.step(c)
                tr.on_waking_step(float(h))
                if done:
                    _f, od = env.reset(); agent.reset(); bab.reset()
    retained_n = len(member._retained)
    if twin == "shuf":
        for r in member._retained:
            r["a"] = F.one_hot(torch.tensor([PERM[int(r["a"].argmax())]]), 5).float()
    log("babble done: retained %d classes %s" % (retained_n, counts))
    snap = [(r["step"], r["a"].clone(), r["obs"][-1][1].clone()) for r in member._retained]
    for _u in range(PRE_UPDATES):
        tr._update("e2_world", member)
    assert torch.equal(agent.world_obs_encoder[0].weight, woe0), "encoder moved before the post phase"
    ev_pre = evaluate(ref, get_head(agent), TE, "z", S)
    log("pre disc4 %.4f k %d" % (ev_pre["disc4_h1"], ev_pre["k"]))

    # post phase: native closed loop; W3 on-policy + 25% retained; W6a trains the encoder (arms 2/3)
    tr.set_e2_world_source("on_policy")
    tr.every_k = 1
    member.updates_per_step = UPS
    agent.reset()
    seed_all(S + 500)
    rew, acts_all = [], []
    tpost = time.time()
    for ep in range(post // EP_STEPS):
        env = make_env(S, 50 + ep)
        hh = StepHarness(agent, env, train_mode=False, seed=S * 1000 + 50 + ep)
        _f, od = env.reset(); agent.reset(); hh.reset()
        for _s in range(EP_STEPS):
            r = hh.step(od)
            rew.append(float(r.harm_signal)); acts_all.append(int(r.action.detach().reshape(-1).argmax()))
            od = r.next_obs_dict
            if r.done:
                _f, od = env.reset(); agent.reset(); hh.reset()
        log("post ep %d done (%.0fs) steps %s" % (ep, time.time() - tpost, dict(tr.steps)))
    head_post = get_head(agent)
    frozen_ok = len(snap) == len(member._retained) and all(
        s == r["step"] and torch.equal(aa, r["a"]) and torch.equal(o, r["obs"][-1][1])
        for (s, aa, o), r in zip(snap, member._retained))

    # readouts: detach the trainer and encode the test streams through the end-of-run read path
    saved_tr = agent.waking_trainer
    agent.waking_trainer = None
    TE_cur = encode_segs(agent, te_segs)
    agent.waking_trainer = saved_tr
    st_ref, st_cur = pr_and_norm(TE), pr_and_norm(TE_cur)
    ev_cur = evaluate_sd(ref, head_post, TE_cur, "z", S)                          # PRIMARY
    ev_cur_std = evaluate_sd(ref, head_post, TE_cur, "z", S, sd=st_cur["sd"])     # S1
    ev_refspace = evaluate_sd(ref, head_post, TE, "z", S)                          # S2
    ev_refspace_bb = evaluate(ref, head_post, TE, "z", S)
    assert ev_refspace_bb == ev_refspace, "evaluate_sd(sd=None) does not reproduce evaluate"
    roll_cur = rollout_e(ref, head_post, TE_cur, S)
    roll_ref = rollout_e(ref, head_post, TE, S)

    g = torch.Generator().manual_seed(S + 5)
    stale = {}
    for nm, buf in (("retained", member._retained), ("on_policy", list(member._on_policy))):
        idx = torch.randperm(len(buf), generator=g)[:256].tolist()
        recs = [buf[i] for i in idx if buf[i].get("z_live") is not None]
        if not recs:
            continue
        zs0 = torch.cat([r["z_live"][0] for r in recs])
        zr0, _zr1 = member._reencode(recs)
        stale[nm] = {"n": len(recs),
                     "rel_diff_median": float(((zs0 - zr0).norm(dim=-1) / zr0.norm(dim=-1).clamp_min(1e-9)).median()),
                     "norm_ratio_current_over_stored_median": float((zr0.norm(dim=-1) / zs0.norm(dim=-1).clamp_min(1e-9)).median())}

    woe_rel = float((agent.world_obs_encoder[0].weight.detach() - woe0).norm() / woe0.norm())
    res = {
        "kind": "run", "seed": S, "arm": arm, "twin": twin, "hazard_class": b0["hazard_class"],
        "config": {"post": post, "ups": UPS, "w6a_ups": W6A_UPS, "w6a_batch": W6A_BATCH,
                   "replay_latent": member.replay_latent, "w3_window": member.reencode_window,
                   "w6a_window": None if wem is None else wem.window},
        "init": ev_init, "B0": B0, "pre": ev_pre,
        "post_current": ev_cur, "post_current_std": ev_cur_std, "post_refspace": ev_refspace,
        "retention": (ev_cur["disc4_h1"] - B0) / (ev_pre["disc4_h1"] - B0) if ev_pre["disc4_h1"] != B0 else None,
        # red-team F3: the pre-registered ratio flips sign when pre < B0; recorded, the formula is unchanged
        "retention_denominator": ev_pre["disc4_h1"] - B0,
        "retention_denominator_nonpositive": bool(ev_pre["disc4_h1"] - B0 <= 0),
        "gate_a": bool(ev_cur["disc4_h1"] >= A_DISC and ev_cur["k"] == A_K),
        "gate_a_std_secondary": bool(ev_cur_std["disc4_h1"] >= A_DISC and ev_cur_std["k"] == A_K),
        "gate_a_refspace_secondary": bool(ev_refspace["disc4_h1"] >= A_DISC and ev_refspace["k"] == A_K),
        "rollout_current": roll_cur, "rollout_refspace": roll_ref,
        "guard": {k: v.status for k, v in tr.guard_results.items()},
        "frozen_retained_unchanged_after_post": frozen_ok,
        "retained_n": retained_n, "babble_class_counts": counts, "onpol_n": len(member._on_policy),
        "updates": dict(tr.steps), "drawn_retained": member.n_drawn_retained,
        "drawn_on_policy": member.n_drawn_on_policy, "n_reencoded": member.n_reencoded,
        "n_cache_hits": member.n_cache_hits,
        "latent_ref": {"pr": st_ref["pr"], "norm_median": st_ref["norm_median"]},
        "latent_current": {"pr": st_cur["pr"], "norm_median": st_cur["norm_median"]},
        "norm_ratio_current_over_ref": st_cur["norm_median"] / max(st_ref["norm_median"], 1e-9),
        "world_obs_encoder_rel_change": woe_rel,
        "w6a_last_terms": None if wem is None else wem.last_terms,
        "staleness": stale,
        "post_reward_per_100": float(np.sum(rew) * 100.0 / len(rew)),
        "post_action_counts": {str(c): acts_all.count(c) for c in range(5)},
        "t_post_s": time.time() - tpost, "t_total_s": time.time() - t0,
    }
    json.dump(res, open(out, "w"), indent=1, default=str)
    log("RESULT cur disc4 %.4f k %d disc5 %.4f gate_a %s ret %s | std %.4f k %d | refspace %.4f k %d | "
        "e %s t30 %.2f late %.3f/%.3f | norm x%.2f PR %.2f->%.2f woe %.3f | guard %s frozen %s" % (
            ev_cur["disc4_h1"], ev_cur["k"], ev_cur["disc5_h1"], res["gate_a"],
            None if res["retention"] is None else round(res["retention"], 3),
            ev_cur_std["disc4_h1"], ev_cur_std["k"], ev_refspace["disc4_h1"], ev_refspace["k"],
            roll_cur["gate_e"], roll_cur["t30_over_t0_median"], roll_cur["late_growth_median"],
            roll_cur["late_growth_max"], res["norm_ratio_current_over_ref"], st_ref["pr"], st_cur["pr"],
            woe_rel, res["guard"], frozen_ok))


# ================================ parent: orchestration, verdict, manifest ============================

_PRINT_LOCK = threading.Lock()


def _spawn(work: Path, tag: str, extra: List[str]) -> Dict[str, Any]:
    out = work / ("%s.json" % tag)
    lg = work / ("%s.log" % tag)
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--child", "--out", str(out)] + extra
    t = time.time()
    with open(lg, "w") as fh:
        rc = subprocess.call(cmd, cwd=str(_REPO_ROOT), stdout=fh, stderr=subprocess.STDOUT)
    rec = None
    if rc == 0 and out.exists():
        try:
            rec = json.load(open(out))
        except Exception:  # noqa: BLE001
            rec = None
    try:
        tail = open(lg).read().splitlines()[-3:]
    except Exception:  # noqa: BLE001
        tail = []
    return {"tag": tag, "rc": rc, "wall_s": time.time() - t, "record": rec, "log_tail": tail}


def _announce(seed: int, cond: str, r: Dict[str, Any], i_done: List[int], n_total: int) -> None:
    ok = r["rc"] == 0 and r["record"] is not None
    with _PRINT_LOCK:
        i_done[0] += 1
        print("Seed %d Condition %s" % (seed, cond), flush=True)
        print("  [train] n2 seed=%d cond=%s ep 1/1 rc=%d wall=%.0fs (%d/%d runs)" % (
            seed, cond, r["rc"], r["wall_s"], i_done[0], n_total), flush=True)
        if ok and r["record"].get("kind") == "run":
            rr = r["record"]
            print("  [n2] s%d %s: post disc4 %.4f k %d gate_a %s ret %s e %s guard %s" % (
                seed, cond, rr["post_current"]["disc4_h1"], rr["post_current"]["k"], rr["gate_a"],
                None if rr["retention"] is None else round(rr["retention"], 3),
                rr["rollout_current"]["gate_e"], rr["guard"]), flush=True)
        print("verdict: %s" % ("PASS" if ok else "FAIL"), flush=True)


def _seed_pipeline(seed: int, post: int, work: Path, i_done: List[int], n_total: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    b0 = _spawn(work, "B0_s%d" % seed, ["--seed", str(seed), "--b0-only"])
    _announce(seed, "B0", b0, i_done, n_total)
    out["B0"] = b0
    b0_path = work / ("B0_s%d.json" % seed)
    for arm in ARMS:
        for twin in TWINS:
            cond = "%s_%s" % (arm, twin)
            if b0["record"] is None:
                r = {"tag": cond, "rc": -1, "wall_s": 0.0, "record": None, "log_tail": ["skipped: B0 failed"]}
            else:
                r = _spawn(work, "N2_%s_s%d" % (cond, seed),
                           ["--seed", str(seed), "--arm", arm, "--twin", twin, "--post", str(post),
                            "--b0-cache", str(b0_path)])
            _announce(seed, cond, r, i_done, n_total)
            out[cond] = r
    return out


def score(runs: Dict[int, Dict[str, Any]], seeds: List[int], n_req: int) -> Dict[str, Any]:
    """The pre-registered keep-the-bar + decision rule (summarize_n2.py, verbatim logic)."""
    def rec(s, arm, twin):
        r = runs.get(s, {}).get("%s_%s" % (arm, twin))
        return None if r is None else r.get("record")
    summ: Dict[str, Dict[str, Any]] = {}
    for arm in ARMS:
        A = {"a": 0, "b": 0, "c_twin_a": 0, "d": 0, "e": 0, "n": 0, "a_std": 0, "a_ref": 0,
             "n_twin": 0, "e_fail_twin_also_fails_e": 0}
        for s in seeds:
            r, t = rec(s, arm, "real"), rec(s, arm, "shuf")
            if t is not None:
                A["n_twin"] += 1
            if r is None:
                continue
            A["n"] += 1
            A["a"] += int(r["gate_a"])
            A["b"] += int(r["retention"] is not None and r["retention"] >= RET_MIN)
            A["d"] += int(all(v == "PASS" for k, v in r["guard"].items() if k == "e2_world") and
                          (t is None or all(v == "PASS" for k, v in t["guard"].items() if k == "e2_world")))
            A["e"] += int(r["rollout_current"]["gate_e"])
            A["a_std"] += int(r["gate_a_std_secondary"])
            A["a_ref"] += int(r["gate_a_refspace_secondary"])
            if t is not None:
                A["c_twin_a"] += int(t["gate_a"])
                # red-team F1 (secondary, not gating): an (e) failure the label-permuted twin shares is a
                # property of the (shared) encoder state, not of what the replay policy retained
                A["e_fail_twin_also_fails_e"] += int((not r["rollout_current"]["gate_e"])
                                                     and (not t["rollout_current"]["gate_e"]))
        # pre-registered counts at n = 5; a dry-run (n < 5) requires every seed (smoke only, never evidence)
        need = (lambda k: k) if n_req == N_SEEDS_REQ else (lambda k: n_req)
        A["keeps_bar"] = bool(A["n"] == n_req and A["a"] >= need(KEEP_A) and A["b"] >= need(KEEP_B)
                              and A["c_twin_a"] <= KEEP_TWIN_A_MAX and A["d"] == n_req and A["e"] >= need(KEEP_E))
        # secondary, NOT gating (amendment A1, red-team F1): the same bar with leg (e) removed
        A["keeps_bar_excluding_e_secondary"] = bool(
            A["n"] == n_req and A["a"] >= need(KEEP_A) and A["b"] >= need(KEEP_B)
            and A["c_twin_a"] <= KEEP_TWIN_A_MAX and A["d"] == n_req)
        summ[arm] = A
    ctrl_miss = summ["frozen"]["n"] - summ["frozen"]["a"]
    if ctrl_miss >= CTRL_MISS_CD:
        v = "CANNOT_DETERMINE"
    elif any(summ[a]["n"] < n_req or summ[a]["n_twin"] < n_req for a in ARMS):
        # a missing twin (the (c) control) is INCOMPLETE too, never a vacuous (c)/(d) pass (red-team F2)
        v = "INCOMPLETE"
    elif summ["reencode"]["keeps_bar"]:
        v = "HOLDS-BOTH" if summ["stored"]["keeps_bar"] else "HOLDS-REENCODE"
    elif summ["stored"]["keeps_bar"]:
        v = "HOLDS-STORED-ONLY"
    else:
        v = "NEITHER"
    return {"per_arm": summ, "control_misses_a": ctrl_miss, "verdict": v}


def _scoring_selftest() -> Dict[str, Any]:
    """Exercise every verdict branch of score() on synthetic records before any compute."""
    def mk(a, ret, e, guard="PASS"):
        return {"record": {"gate_a": a, "retention": ret, "guard": {"e2_world": guard},
                           "rollout_current": {"gate_e": e}, "gate_a_std_secondary": a,
                           "gate_a_refspace_secondary": a}}

    def grid(ctrl_a, re_a, st_a, twin_a=False):
        runs = {}
        for i, s in enumerate(SEEDS):
            runs[s] = {"frozen_real": mk(ctrl_a[i], 1.0, True), "frozen_shuf": mk(twin_a, 0.0, True),
                       "reencode_real": mk(re_a[i], 1.0, True), "reencode_shuf": mk(twin_a, 0.0, True),
                       "stored_real": mk(st_a[i], 1.0, True), "stored_shuf": mk(twin_a, 0.0, True)}
        return runs
    T, Fa = [True] * 5, [False] * 5
    cases = {
        "CANNOT_DETERMINE": grid([True, True, True, False, False], T, T),
        "HOLDS-BOTH": grid(T, T, T),
        "HOLDS-REENCODE": grid(T, T, Fa),
        "HOLDS-STORED-ONLY": grid(T, Fa, T),
        "NEITHER": grid(T, Fa, Fa),
        "NEITHER_twin": grid(T, T, T, twin_a=True),
    }
    got = {k: score(v, SEEDS, 5)["verdict"] for k, v in cases.items()}
    inc = grid(T, T, T)
    del inc[SEEDS[0]]["stored_real"]
    got["INCOMPLETE"] = score(inc, SEEDS, 5)["verdict"]
    inc2 = grid(T, T, T)
    del inc2[SEEDS[0]]["reencode_shuf"]
    got["INCOMPLETE_twin"] = score(inc2, SEEDS, 5)["verdict"]
    want = {"CANNOT_DETERMINE": "CANNOT_DETERMINE", "HOLDS-BOTH": "HOLDS-BOTH", "HOLDS-REENCODE": "HOLDS-REENCODE",
            "HOLDS-STORED-ONLY": "HOLDS-STORED-ONLY", "NEITHER": "NEITHER", "NEITHER_twin": "NEITHER",
            "INCOMPLETE": "INCOMPLETE", "INCOMPLETE_twin": "INCOMPLETE"}
    assert got == want, "scoring selftest failed: %s" % got
    return {"cases": sorted(got), "ok": True}


def _flat(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return float(int(v))
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return None


def main(dry_run: bool = False) -> Dict[str, Any]:
    started_at = time.perf_counter()
    selftest = _scoring_selftest()
    seeds = DRY["seeds"] if dry_run else SEEDS
    post = DRY["post"] if dry_run else POST
    n_req = len(seeds)
    jobs = int(os.environ.get("N2_JOBS", max(1, min(4, (os.cpu_count() or 2) // 2))))
    work = Path(tempfile.mkdtemp(prefix="exq1108_n2_"))
    print("V3-EXQ-1108 N2 full dose: dry_run=%s seeds=%s post=%d jobs=%d pin=%s work=%s" % (
        dry_run, seeds, post, jobs, _PIN["resolved_sha"][:10], work), flush=True)

    canary: Dict[str, Any] = {"ran": False}
    if dry_run:
        # vendoring canary: seed 106, frozen/real at post 1200, against W3's published Mac row
        cb = _spawn(work, "canary_B0_s106", ["--seed", "106", "--b0-only"])
        cr = _spawn(work, "canary_frozen_s106", ["--seed", "106", "--arm", "frozen", "--twin", "real",
                                                 "--post", str(CANARY["post"]),
                                                 "--b0-cache", str(work / "canary_B0_s106.json")])
        canary = {"ran": True, "B0": cb, "frozen": cr}
        rb, rr = cb.get("record"), cr.get("record")
        if rb and rr:
            obs = {"B0": rb["B0"]["disc4_h1"], "pre_disc4": rr["pre"]["disc4_h1"], "pre_k": rr["pre"]["k"],
                   "post_disc4": rr["post_current"]["disc4_h1"], "post_k": rr["post_current"]["k"],
                   "retention": round(rr["retention"], 3)}
            canary["observed"] = obs
            canary["matches_mac_reference"] = bool(
                abs(obs["B0"] - CANARY["B0"]) < 1e-9 and abs(obs["pre_disc4"] - CANARY["pre_disc4"]) < 1e-9
                and obs["pre_k"] == CANARY["pre_k"] and abs(obs["post_disc4"] - CANARY["post_disc4"]) < 1e-9
                and obs["post_k"] == CANARY["post_k"] and abs(obs["retention"] - CANARY["retention"]) < 1e-9)
        print("  [canary] s106 frozen p1200: %s matches_mac_reference=%s" % (
            canary.get("observed"), canary.get("matches_mac_reference")), flush=True)
        if sys.platform == "darwin":
            assert canary.get("matches_mac_reference"), "vendoring canary FAILED on darwin: %s" % canary.get("observed")

    n_total = n_req * (1 + len(ARMS) * len(TWINS))
    i_done = [0]
    runs: Dict[int, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {s: ex.submit(_seed_pipeline, s, post, work, i_done, n_total) for s in seeds}
        for s in seeds:
            runs[s] = futs[s].result()

    sc = score(runs, seeds, n_req)
    verdict = sc["verdict"]
    outcome = "PASS" if verdict.startswith("HOLDS") else "FAIL"

    # ---- per-seed table, integrity checks, arm_results with pinned fingerprints ----------------------
    per_seed: List[Dict[str, Any]] = []
    arm_results: List[Dict[str, Any]] = []
    crashed: List[str] = []
    pre_equal_max_absdiff = 0.0
    frozen_ok_all = True
    pin_fp = pin_fingerprint_kwargs(_PIN)
    for s in seeds:
        b0r = runs[s]["B0"]["record"]
        row: Dict[str, Any] = {"seed": s, "hazard_class": None if b0r is None else b0r["hazard_class"],
                               "B0_disc4": None if b0r is None else b0r["B0"]["disc4_h1"],
                               "B0_k": None if b0r is None else b0r["B0"]["k"], "arms": {}}
        if b0r is None:
            crashed.append("B0_s%d" % s)
        pres = []
        for arm in ARMS:
            for twin in TWINS:
                cond = "%s_%s" % (arm, twin)
                r = runs[s][cond]
                rr = r["record"]
                if rr is None:
                    crashed.append("%s_s%d" % (cond, s))
                    row["arms"][cond] = None
                    continue
                if twin == "real":
                    pres.append(rr["pre"]["disc4_h1"])
                frozen_ok_all = frozen_ok_all and bool(rr["frozen_retained_unchanged_after_post"])
                st = rr.get("staleness", {})
                row["arms"][cond] = {
                    "pre_disc4": rr["pre"]["disc4_h1"], "pre_k": rr["pre"]["k"],
                    "post_disc4": rr["post_current"]["disc4_h1"], "post_k": rr["post_current"]["k"],
                    "post_disc5": rr["post_current"]["disc5_h1"], "retention": rr["retention"],
                    "retention_denominator_nonpositive": rr.get("retention_denominator_nonpositive"),
                    "gate_a": rr["gate_a"], "gate_e": rr["rollout_current"]["gate_e"],
                    "t30_over_t0_median": rr["rollout_current"]["t30_over_t0_median"],
                    "late_growth_max": rr["rollout_current"]["late_growth_max"],
                    "S1_std_disc4": rr["post_current_std"]["disc4_h1"], "S1_std_k": rr["post_current_std"]["k"],
                    "S2_ref_disc4": rr["post_refspace"]["disc4_h1"], "S2_ref_k": rr["post_refspace"]["k"],
                    "norm_ratio": rr["norm_ratio_current_over_ref"], "pr_ref": rr["latent_ref"]["pr"],
                    "pr_cur": rr["latent_current"]["pr"], "woe_rel_change": rr["world_obs_encoder_rel_change"],
                    "stale_retained": (st.get("retained") or {}).get("rel_diff_median"),
                    "stale_on_policy": (st.get("on_policy") or {}).get("rel_diff_median"),
                    "guard": rr["guard"], "post_reward_per_100": rr["post_reward_per_100"],
                }
                cfg_slice = {"seed": s, "arm": arm, "twin": twin, "post": post, "ups": UPS, "w6a_ups": W6A_UPS,
                             "w6a_batch": W6A_BATCH, "pre_updates": PRE_UPDATES, "b0_n_eps": B0_N_EPS,
                             "pin": _PIN["resolved_sha"]}
                cell = {"seed": s, "arm": arm, "twin": twin, "condition": cond,
                        "post_disc4": rr["post_current"]["disc4_h1"], "post_k": rr["post_current"]["k"],
                        "gate_a": rr["gate_a"], "retention": rr["retention"],
                        "gate_e": rr["rollout_current"]["gate_e"], "wall_s": r["wall_s"]}
                cell["arm_fingerprint"] = compute_arm_fingerprint(
                    config_slice=cfg_slice, seed=s, script_path=Path(__file__), rng_fully_reset=True,
                    config_slice_declared=True, **pin_fp)
                arm_results.append(cell)
        if len(pres) >= 2:
            pre_equal_max_absdiff = max(pre_equal_max_absdiff, float(max(pres) - min(pres)))
        per_seed.append(row)

    ctrl = sc["per_arm"]["frozen"]
    preconditions = [
        {"name": "control_arm_meets_gate_a",
         "description": "positive control: the frozen-encoder arm (= the W3 member-gate config) must meet (a) on "
                        ">= n-1 seeds, else the decision rule returns CANNOT_DETERMINE (pre-registered rule 1)",
         "measured": float(ctrl["a"]), "threshold": float(n_req - (CTRL_MISS_CD - 1)),
         "control": "frozen arm: no encoder training, the configuration W3 passed at this dose",
         "direction": "lower",
         "met": bool(ctrl["n"] - ctrl["a"] < CTRL_MISS_CD)},
        {"name": "pre_phase_identical_across_arms",
         "description": "babbling + pre phase are byte-identical across arms within a seed (W6a trains only post)",
         "measured": pre_equal_max_absdiff, "threshold": 0.0, "direction": "upper",
         "met": bool(pre_equal_max_absdiff == 0.0)},
        {"name": "retained_babbling_set_frozen",
         "description": "the W3 retained babbling set is unchanged after the post phase in every run",
         "measured": float(int(frozen_ok_all)), "threshold": 1.0, "direction": "lower",
         "met": bool(frozen_ok_all)},
        {"name": "retention_denominator_positive",
         "description": "(b)'s pre-registered ratio (post - B0)/(pre - B0) is only interpretable when pre > B0; "
                        "count of real runs with pre <= B0 (red-team F3; the formula itself is unchanged)",
         "measured": float(sum(1 for row in per_seed for c, x in row["arms"].items()
                               if x and c.endswith("_real") and x.get("retention_denominator_nonpositive"))),
         "threshold": 0.0, "direction": "upper",
         "met": bool(not any(x and c.endswith("_real") and x.get("retention_denominator_nonpositive")
                             for row in per_seed for c, x in row["arms"].items()))},
        {"name": "all_runs_completed",
         "description": "every B0 and (seed, arm, twin) run wrote its record",
         "measured": float(len(crashed)), "threshold": 0.0, "direction": "upper", "met": bool(not crashed)},
    ]

    def _nd(arm: str) -> bool:
        vals, twins = [], []
        for s in seeds:
            a = (runs[s].get("%s_real" % arm) or {}).get("record")
            t = (runs[s].get("%s_shuf" % arm) or {}).get("record")
            if a is not None:
                vals.append(a["post_current"]["disc4_h1"])
            if a is not None and t is not None:
                twins.append(a["post_current"]["disc4_h1"] != t["post_current"]["disc4_h1"])
        return bool(len(vals) == n_req and any(twins))
    crit_nd = {"%s_keeps_bar" % arm: _nd(arm) for arm in ARMS}
    criteria = []
    for arm in ARMS:
        A = sc["per_arm"][arm]
        criteria.append({"name": "%s_keeps_bar" % arm, "passed": A["keeps_bar"],
                         "load_bearing": arm in ("reencode", "stored"),
                         "a_count": A["a"], "a_threshold": KEEP_A, "b_count": A["b"], "b_threshold": KEEP_B,
                         "twin_a_count": A["c_twin_a"], "twin_a_max": KEEP_TWIN_A_MAX,
                         "d_count": A["d"], "d_threshold": KEEP_D, "e_count": A["e"], "e_threshold": KEEP_E,
                         "n_runs": A["n"]})
    combination_rule = ("CANNOT_DETERMINE if frozen misses (a) on >= %d seeds; else INCOMPLETE if any arm has < %d "
                        "seeds; else HOLDS-REENCODE if reencode keeps the bar (HOLDS-BOTH if stored also keeps it); "
                        "else HOLDS-STORED-ONLY if stored keeps it; else NEITHER. outcome PASS iff HOLDS-*." % (
                            CTRL_MISS_CD, n_req))
    raw = {"verdict_holds": verdict.startswith("HOLDS"), "control_misses_a": sc["control_misses_a"]}
    for arm in ARMS:
        A = sc["per_arm"][arm]
        for k in ("a", "b", "c_twin_a", "d", "e", "n", "a_std", "a_ref", "n_twin", "e_fail_twin_also_fails_e"):
            raw["%s_%s_count" % (arm, k)] = A[k]
        raw["%s_keeps_bar" % arm] = A["keeps_bar"]
        raw["%s_keeps_bar_excluding_e_secondary" % arm] = A["keeps_bar_excluding_e_secondary"]
        pd = [row["arms"].get("%s_real" % arm) for row in per_seed]
        pd = [x for x in pd if x]
        if pd:
            raw["%s_post_disc4_median" % arm] = float(np.median([x["post_disc4"] for x in pd]))
            raw["%s_retention_median" % arm] = float(np.median([x["retention"] for x in pd
                                                                 if x["retention"] is not None] or [float("nan")]))
            raw["%s_late_growth_max_max" % arm] = float(max(x["late_growth_max"] for x in pd))
            raw["%s_norm_ratio_median" % arm] = float(np.median([x["norm_ratio"] for x in pd]))
    readout = {k: fv for k, fv in ((k, _flat(v)) for k, v in raw.items()) if fv is not None}

    label = verdict.lower().replace("-", "_")
    summary = ("N2 at post %d, seeds %s: verdict %s. frozen (a) %d/%d, reencode keeps bar %s, stored keeps bar %s." % (
        post, seeds, verdict, ctrl["a"], ctrl["n"], sc["per_arm"]["reencode"]["keeps_bar"],
        sc["per_arm"]["stored"]["keeps_bar"]))
    manifest: Dict[str, Any] = {
        "run_id": make_run_id(EXPERIMENT_TYPE, queue_id=QUEUE_ID),
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "non_contributory",
        "evidence_direction_note": ("campaign design probe (coupled_loop_repair N2): gates the W6 preset buffer "
                                    "policy; presses no claim"),
        "outcome": outcome,
        "n2_verdict": verdict,
        "timestamp_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "dry_run": bool(dry_run),
        "substrate_pin": pin_manifest_block(_PIN),
        "scoring_selftest": selftest,
        "canary_s106_vendoring": {k: v for k, v in canary.items() if k not in ("B0", "frozen")} if canary.get("ran")
        else canary,
        "per_seed": per_seed,
        "per_arm": sc["per_arm"],
        "arm_results": arm_results,
        "per_run_records": {str(s): {c: (runs[s][c] or {}).get("record") for c in runs[s]} for s in seeds},
        "per_run_wall_s": {str(s): {c: (runs[s][c] or {}).get("wall_s") for c in runs[s]} for s in seeds},
        "per_run_log_tail": {str(s): {c: (runs[s][c] or {}).get("log_tail") for c in runs[s]} for s in seeds},
        "crashed_runs": crashed,
        "criteria": criteria,
        "combination_rule": combination_rule,
        "readout": readout,
        "interpretation": {
            "label": label,
            "summary": summary,
            "preconditions": preconditions,
            "criteria_non_degenerate": crit_nd,
            "what_a_null_does_not_mean": (
                "CANNOT_DETERMINE says the frozen control did not reach the W3 bar at this dose on these seeds, so no "
                "replay policy can be judged; it says nothing about re-encode vs stored. NEITHER says neither W3 "
                "replay policy keeps the L2R bar while W6a trains the encoder at 1 update per waking step for 1200 "
                "steps; it does not say a phased schedule (encoder frozen while W3 trains) fails. D1 only: no E3 "
                "consumer or behaviour is read. LEG (e) (amendment A1, red-team F1, stated before the run): in every "
                "W6a-ON run seen so far the rollout late-growth max exceeded 1.2 in BOTH the real arm and its "
                "label-permuted twin, while every frozen run passed; an (e) failure the twin shares reflects the "
                "trained encoder's latent space, which both W6a arms share, not the replay policy. So a NEITHER whose "
                "arms fail ONLY (e) (per_arm keeps_bar_excluding_e_secondary true, e_fail_twin_also_fails_e high) "
                "is a rollout-stability finding about the W6a encoder, not evidence that the buffer policy loses "
                "the L2R bar; the pre-registered verdict is still reported as computed."),
            "dv_symmetry": (
                "disc4/k are invariant to a uniform rescaling of the latent space, and W6a rescales z_world ~9x; the "
                "manipulation still reaches the DV through the head's TRAINING (stored z go stale, re-encoded z do "
                "not), not through the readout metric, and S1 (per-dimension standardised) reports whether any break "
                "is metric anisotropy. The shuf twin permutes action labels, which argmin-over-classes disc4 is NOT "
                "invariant to (it scores the true class)."),
        },
        "non_degenerate": bool(all(crit_nd.values()) and not crashed),
        "degeneracy_reason": ("" if (all(crit_nd.values()) and not crashed) else
                              "a run crashed, or an arm's real and shuf heads read identically on every seed"),
        "pre_registered_thresholds": {
            "SEEDS": SEEDS, "POST": POST, "UPS": UPS, "W6A_UPS": W6A_UPS, "W6A_BATCH": W6A_BATCH,
            "PRE_UPDATES": PRE_UPDATES, "B0_N_EPS": B0_N_EPS, "A_DISC": A_DISC, "A_K": A_K, "RET_MIN": RET_MIN,
            "E_LATE_MAX": E_LATE_MAX, "E_T30_MAX": E_T30_MAX, "KEEP_A": KEEP_A, "KEEP_B": KEEP_B,
            "KEEP_TWIN_A_MAX": KEEP_TWIN_A_MAX, "KEEP_D": KEEP_D, "KEEP_E": KEEP_E, "CTRL_MISS_CD": CTRL_MISS_CD,
            "PERM": PERM,
        },
        "ethics_preflight": {
            "involves_negative_valence": False, "involves_suffering_like_state": False,
            "involves_self_model": False, "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False, "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False, "decision": "allow",
        },
        "custom_information": {
            "lineage": {"prereg": "REE_assembly 025fc6a5cfe", "first_result_cd": "REE_assembly bbccb5d1416",
                        "record": "REE_assembly/evidence/planning/n2_replay_encoder_probe_20260925.md",
                        "amendment": "that record sec 4 (dose 600 -> 1200, seeds 811-815, fleet driver)",
                        "mac_probe": "REE_assembly/evidence/planning/probes/n2/n2_probe.py"},
            "cross_machine_class": ("torch.multinomial differs linux vs darwin on the native action path, so the "
                                    "post-phase trajectories are not bit-comparable to the Mac N2/W3 runs"),
            "z_goal_stream": "not recorded: agents live in per-run child processes; this harness does not enable z_goal",
            "jobs": jobs, "work_dir": str(work),
        },
    }
    out_path = write_flat_manifest(
        manifest,
        dry_run=dry_run,
        config={"post": post, "ups": UPS, "w6a_ups": W6A_UPS, "w6a_batch": W6A_BATCH, "pre_updates": PRE_UPDATES,
                "b0_n_eps": B0_N_EPS, "arms": ARMS, "twins": TWINS, "perm": PERM, "grid": GRID,
                "ep_steps": EP_STEPS, "world_dim": 32, "alpha_world": 0.3, "torch_threads": 2,
                "substrate_pin": _PIN["resolved_sha"], "thresholds": manifest["pre_registered_thresholds"]},
        seeds=list(seeds),
        script_path=Path(__file__),
        started_at=started_at,
        env=make_env(seeds[0], 50),
    )
    manifest["_out_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="V3-EXQ-1108 N2 full dose, ree_core pinned to 9b322d5")
    ap.add_argument("--dry-run", action="store_true",
                    help="vendoring canary (s106 frozen p1200) + seed 811 at post 200; manifest relocated")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--arm", choices=ARMS, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--twin", choices=TWINS, default="real", help=argparse.SUPPRESS)
    ap.add_argument("--post", type=int, default=POST, help=argparse.SUPPRESS)
    ap.add_argument("--out", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--b0-only", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--b0-cache", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        child_run(args.seed, args.arm, args.twin, args.post, args.out, args.b0_only, args.b0_cache)
        sys.exit(0)
    result = main(dry_run=args.dry_run)
    out_path = result["_out_path"]
    print()
    print("=== V3-EXQ-1108 N2 full dose (pinned 9b322d5) ===")
    print("n2 verdict: %s" % result["n2_verdict"])
    print("outcome: %s" % result["outcome"])
    print("summary: %s" % result["interpretation"]["summary"])
    print("manifest: %s" % out_path)
    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
