"""Probe 6: with SD-055 coupled ON, does the SD-061 temperature lever move
anything at the proposal layer -- and specifically does it move the registered
DV (candidate first-action-class entropy)?

Compares, at one seed, coupled vs uncoupled, and (within coupled) a lifted vs
un-lifted stuck_score:
  - the raw action-object tensors of the proposed candidate set
  - the raw actions tensors
  - the first-action-class multiset (the registered DV's input)
"""
from __future__ import annotations
import sys, random, math
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np, torch
from collections import Counter
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig

DIMS = dict(body_obs_dim=4, world_obs_dim=8, action_dim=4, self_dim=8,
            world_dim=8, alpha_world=0.9, use_sleep_loop=False,
            sws_enabled=False, rem_enabled=False,
            use_sleep_aggregation_cluster=False)


def ent(cls):
    n = len(cls)
    if not n:
        return 0.0
    e = 0.0
    for k in Counter(cls).values():
        p = k / n
        e -= p * math.log(p)
    return e


def run(enable_cem, forced, steps=12, seed=0, widen=8, gain=1.0):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    cfg = REEConfig.from_dims(use_difficulty_gated_proposal_entropy=True,
                              dgpe_enable_differentiable_cem=enable_cem,
                              dgpe_candidate_widen_max=widen,
                              dgpe_temperature_gain_max=gain, **DIMS)
    ag = REEAgent(cfg); ag.reset()
    rows = []
    orig = ag.hippocampal.propose_trajectories

    def w(*a, **k):
        out = orig(*a, **k)
        cls, aos, acts = [], [], []
        for t in out:
            try:
                cls.append(int(torch.argmax(t.actions[0, 0, :]).item()))
                acts.append(t.actions.detach().clone())
            except Exception:
                cls.append(-1)
            s = t.get_action_object_sequence()
            if s is not None:
                aos.append(s.detach().clone())
        rows.append({
            "n": len(out), "cls": tuple(cls), "ent": round(ent(cls), 6),
            "ao": torch.stack(aos) if aos else None,
            "act": torch.stack(acts) if acts else None,
        })
        return out

    ag.hippocampal.propose_trajectories = w
    torch.manual_seed(seed)
    for _ in range(steps):
        ag._last_stuck_score = forced
        ag.act_with_split_obs(torch.randn(1, 4), torch.randn(1, 8))
    return rows


def cmp(tag, A, B):
    print("== %s ==" % tag, flush=True)
    for i, (a, b) in enumerate(zip(A, B)):
        ao_same = (a["ao"] is None) == (b["ao"] is None) and (
            a["ao"] is None or (a["ao"].shape == b["ao"].shape
                                and torch.equal(a["ao"], b["ao"])))
        ao_maxdiff = (float((a["ao"] - b["ao"]).abs().max())
                      if a["ao"] is not None and b["ao"] is not None
                      and a["ao"].shape == b["ao"].shape else None)
        act_same = (a["act"] is not None and b["act"] is not None
                    and a["act"].shape == b["act"].shape
                    and torch.equal(a["act"], b["act"]))
        print("  proposal %d: n=%d/%d  ao_identical=%s ao_maxabsdiff=%s "
              "actions_identical=%s  first_action_ent=%.6f/%.6f  cls_identical=%s"
              % (i, a["n"], b["n"], ao_same, ao_maxdiff, act_same,
                 a["ent"], b["ent"], a["cls"] == b["cls"]), flush=True)


cmp("coupled ON vs OFF, forced stuck=1.0", run(True, 1.0), run(False, 1.0))
cmp("coupled ON: stuck=1.0 vs stuck=0.0", run(True, 1.0), run(True, 0.0))
cmp("coupled ON, COUNT LEVER OFF (widen=0): stuck=1.0 vs 0.0",
    run(True, 1.0, widen=0), run(True, 0.0, widen=0))
cmp("uncoupled, COUNT LEVER OFF (widen=0): stuck=1.0 vs 0.0",
    run(False, 1.0, widen=0), run(False, 0.0, widen=0))
