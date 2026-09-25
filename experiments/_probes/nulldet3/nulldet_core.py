"""V3-EXQ-1105 port of the grounded-valuation null-detector probe harness (nulldet v1/v2).

Source: REE_assembly/evidence/planning/probes/nulldet/nulldet_probe.py (frozen with the v1
pre-registration commit 789b61f6958; byte-identical to the copy v2 ran, md5 5f2c126fb21e18de9ad4ae5af68ff1b3).
Everything from `from __future__` down to the end of `run_arm` is copied VERBATIM, except the three
sys.path lines (they pointed at the probe's private ree-v3-wt worktree; they now point at this
repo's root). `main()` is replaced by `preamble()` below, which performs the identical calls in the
identical order with the identical budgets (PREAMBLE_BUDGETS_FULL); the only additions are a budget
parameter (smaller values are used ONLY under --dry-run) and a guard that the grid-world tie-break
patch is applied exactly once per process. Nothing here lands in ree_core. ASCII-only output.
"""
from __future__ import annotations

import argparse
import copy
import functools
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[2] / "experiments"))
sys.path.insert(0, str(HERE))

from experiments._harness import StepHarness  # noqa: E402
import rollout_fidelity_probe as R  # noqa: E402
import encoding_vs_objective_probe as E  # noqa: E402
import balanced_replay_probe as BP  # noqa: E402
import partitioned_repair_probe as PR  # noqa: E402
import evaluation_edge_probe as EP  # noqa: E402

torch.set_num_threads(2)
DEPTH = PR.DEPTH
CH = ["F", "harm", "residue", "benefit"]
FIELDS = ["f_weight", "lambda_ethical", "rho_residue", "benefit_weight"]
DKEYS = ["f_weighted", "harm_weighted", "residue_weighted", "benefit_weighted"]
SIGN = np.array([1.0, 1.0, 1.0, -1.0])
LN4 = math.log(4.0)
THR = 0.1
STRATUM_CONTACTS_600 = 15  # pre-registered: M0 true harm contacts in steps [0,600) >= 15 -> hazard_trapped
MAXHACK_THETA = [0.0, -LN4, -LN4, LN4]


class Rule:
    """Copied from valuation_smoke2_probe.Rule (M0 / M4 / M1RAW unchanged); FIXED added; M4 noise
    seed offset made a parameter so K null replicates are independent (rep 0 offset differs from
    the smoke's single M4 by construction)."""

    def __init__(self, kind, seed, m4_sd=None, noise_offset=9001, fixed=None):
        self.kind = kind; self.theta = np.zeros(4) if fixed is None else np.array(fixed, float); self.hist = []
        self.absv = []; self.A = []; self.V = []; self.R = []
        self.rng = np.random.default_rng(seed + noise_offset); self.m4_sd = m4_sd

    def update(self, closed):
        old = self.theta.copy()
        if self.kind in ("M0", "FIXED") or closed is None:
            pass
        elif self.kind == "M4":
            self.theta = self.theta + self.rng.normal(0.0, 1.0, 4) * self.m4_sd
        elif closed["committed"]:
            v = closed["v"]
            if self.kind == "M1RAW":
                self.absv.append(np.abs(v))
                if closed["R"] != 0:
                    med = np.median(np.asarray(self.absv), axis=0)
                    self.theta = self.theta + 0.05 * np.sign(closed["R"]) * np.sign(v) * (np.abs(v) > med)
        self.theta = np.clip(self.theta, -LN4, LN4)
        return self.theta - old


def run_arm(seed, enc_state, headD, evT, gate_n, steps, rule):
    R.seed_all(seed)
    env, agent, cfg = R.build_B(seed, False)
    agent.latent_stack.load_state_dict(enc_state)
    BP.set_head(agent, headD)
    agent.hippocampal.config.use_action_class_scaffold_candidates = True
    agent.e3._score_depth_limit = DEPTH
    EP.configure_eval(agent, evT, True, gate_n)
    agent.eval()
    e3 = agent.e3
    e3.e3_score_decomp_enabled = True
    w0 = np.array([float(getattr(e3.config, f)) for f in FIELDS])
    rew = []; ticks = []; st = {"step": -1, "pending": None, "demo": None}
    orig = e3.select

    def write_weights():
        w = w0 * np.exp(rule.theta)
        for f, val in zip(FIELDS, w):
            setattr(e3.config, f, float(val))
        return w

    def wsel(*a, **k):
        i = st["step"]
        closed = None
        if st["pending"] is not None:
            p = st["pending"]
            rr = np.asarray(rew[p["step"]:i], float)
            closed = dict(p, R=float(rr[np.abs(rr) > THR].sum()) if len(rr) else 0.0)
        dth = rule.update(closed)
        w = write_weights()
        # (kept from the smoke for construction identity: first-weight-change rescoring demo, RNG restored)
        if st["demo"] is None and np.any(np.abs(rule.theta) > 1e-9):
            rs = torch.get_rng_state(); nps = np.random.get_state()
            e3c = copy.deepcopy(e3)
            for f, val in zip(FIELDS, w0):
                setattr(e3c.config, f, float(val))
            e3c.e3_score_decomp_enabled = True
            dflt = []
            with torch.no_grad():
                for c in a[0]:
                    e3c.score_trajectory(c)
                    dflt.append([e3c._last_traj_components[kk] for kk in DKEYS])
            torch.set_rng_state(rs); np.random.set_state(nps)
            st["demo"] = {"step": i, "theta": rule.theta.tolist()}
        res = orig(*a, **k)
        pc = (e3.last_score_decomp or {}).get("per_candidate") or []
        X = np.asarray([[d.get(kk, 0.0) for kk in DKEYS] for d in pc], float)
        wsafe = np.where(w > 0, w, 1.0)
        sel = int(res.selected_index)
        flip = None
        if X.shape[0] >= 2 and sel < X.shape[0]:
            raw = X / wsafe
            v = -SIGN * (raw[sel] - np.delete(raw, sel, axis=0).mean(0))
            # 4-channel pick-flip diagnostic (informational): argmin of the signed 4-channel sum under
            # the CURRENT weights vs under DEFAULT weights, same candidates. Ignores non-channel terms.
            s_now = (raw * w * SIGN).sum(1); s_def = (raw * w0 * SIGN).sum(1)
            flip = bool(int(np.argmin(s_now)) != int(np.argmin(s_def)))
        else:
            v = np.zeros(4)
        acls = int(res.selected_action.detach().reshape(-1).argmax())
        st["pending"] = {"step": i, "v": v, "a": acls, "committed": bool(res.committed)}
        ticks.append({"step": i, "theta": rule.theta.tolist(), "dtheta": dth.tolist(), "committed": bool(res.committed),
                      "flip4": flip})
        return res
    e3.select = wsel
    h = StepHarness(agent, env, train_mode=False, seed=seed)
    _f, obs = env.reset(); agent.reset(); h.reset()
    tts, ep, ends = [], 0, []
    for i in range(steps):
        st["step"] = i
        r = h.step(obs)
        rew.append(float(r.harm_signal))
        info = r.info if isinstance(r.info, dict) else {}
        tts.append(info.get("transition_type", "none"))
        obs = r.next_obs_dict
        if r.done:
            ends.append([i, str(info.get("done_cause", ""))])
            _f, obs = env.reset(); agent.reset(); h.reset(); ep += 1

    def counts(lo, hi):
        tc = Counter(tts[lo:hi]); rw = np.asarray(rew[lo:hi])
        return {"reward_per_100": float(rw.sum() * 100 / max(hi - lo, 1)),
                "harm_contacts": int(sum(v for k, v in tc.items() if k in EP.CONTACT)),
                "hazard_approach": int(tc.get("hazard_approach", 0)), "benefit_approach": int(tc.get("benefit_approach", 0)),
                "benefit_contacts": int(sum(v for k, v in tc.items() if k in EP.BCONTACT)),
                "health_depleted_ends": int(sum(1 for s_, c in ends if lo <= s_ < hi and c == "health_depleted"))}
    return {"first600": counts(0, 600), "all": counts(0, steps), "half1": counts(0, steps // 2),
            "half2": counts(steps // 2, steps), "episodes_ended": ep, "ends": ends, "ticks": ticks,
            "w0": w0.tolist(), "tts": tts}



# ---------------------------------------------------------------------------------------------
# Port additions (V3-EXQ-1105). Not part of the v1/v2 probe text above.
# ---------------------------------------------------------------------------------------------

# v1/v2 main() budgets, verbatim: PR.p0 (20 episodes x 50 steps), collect_replay 1500,
# train_arm 3000 (x2), collect_native 1500, collect_random 1500, train_evaluators 1500.
PREAMBLE_BUDGETS_FULL = {"p0_episodes": 20, "replay": 1500, "head_train": 3000,
                         "native": 1500, "random": 1500, "eval_train": 1500}
PREAMBLE_BUDGETS_DRY = {"p0_episodes": 2, "replay": 120, "head_train": 60,
                        "native": 150, "random": 150, "eval_train": 60}

_TIEBREAK_PATCHED = {"done": False}


def apply_tiebreak_patch_once():
    """v1 main(): R.CausalGridWorldV2 = functools.partial(..., proximity_approach_magnitude_tiebreak=True).
    Applied once per process (each seed runs in its own worker process, as in v1/v2)."""
    if not _TIEBREAK_PATCHED["done"]:
        R.CausalGridWorldV2 = functools.partial(R.CausalGridWorldV2, proximity_approach_magnitude_tiebreak=True)
        _TIEBREAK_PATCHED["done"] = True


def _p0(agent, seed, episodes):
    """Identical to PR.p0 except that the episode count is a parameter (PR.p0 hard-codes 20)."""
    if episodes == 20:
        return PR.p0(agent, seed)
    from experiments._lib.zworld_p0_warmup import run_zworld_p0
    from experiments._lib.capability_eval import RandomPolicy
    from ree_core.latent.zworld_p0 import ZWorldP0Config
    return run_zworld_p0(agent, R.build_B(seed, False)[0], seed=seed, episodes=episodes, steps_per_episode=50,
                         policy=RandomPolicy(seed), label="bt0924d", dry_run=False,
                         config=ZWorldP0Config(preservation_weight=1000.0))


def preamble(seed, budgets=None):
    """v1 main() lines from `R.seed_all(a.seed)` to the PREAMBLE print, as a function.
    Returns (enc_state, headD, evT, gate_n, agent, p0_info)."""
    b = dict(PREAMBLE_BUDGETS_FULL if budgets is None else budgets)
    apply_tiebreak_patch_once()
    R.seed_all(seed)
    env, agent, cfg = R.build_B(seed, False)
    A = env.action_dim
    p0_info = _p0(agent, seed, b["p0_episodes"])
    agent.eval()
    enc_state = copy.deepcopy(agent.latent_stack.state_dict())
    init = BP.get_head(agent)
    trans = BP.collect_replay(agent, env, b["replay"], seed)
    rnd = E.collect(R.build_B(seed + 11, False)[0], agent, len(trans) + 200, seed + 3)
    trans_d = []
    for e in rnd:
        for t in range(e["a"].shape[0] - 1):
            trans_d.append((e["z"][t:t + 1], int(e["a"][t]), e["z"][t + 1:t + 2]))
    trans_d = trans_d[:len(trans)]
    headA, _ = BP.train_arm(agent, init, trans, A, np.ones(len(trans)), b["head_train"], seed)
    headD, _ = BP.train_arm(agent, init, trans_d, A, np.ones(len(trans_d)), b["head_train"], seed)
    init_ev = EP.eval_heads(agent)
    BP.set_head(agent, headA)
    nat = EP.collect_native(agent, R.build_B(seed + 21, False)[0], b["native"], seed + 21)
    ran = EP.collect_random(agent, R.build_B(seed + 23, False)[0], b["random"], seed + 23)
    agent.reset()
    evT, infoT = EP.train_evaluators(agent, init_ev, nat + ran, b["eval_train"], seed, False)
    gate_n = infoT["n_benefit_pos_train"]
    return enc_state, headD, evT, gate_n, agent, p0_info
