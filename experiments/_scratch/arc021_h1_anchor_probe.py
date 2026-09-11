"""ARC-021 H1 pre-build probe 5: is the vacuity REPAIRABLE, and if so how?

Probe 4 established that unfreezing the encoder collapses the control arm's calibration_gap
by ~90x (0.217 -> 0.002) and its attribution_auc to chance, in BOTH arms -- so the registry
leg's criterion is vacuous as specified. A refusal is more useful if it names what is
actually owed, so this measures the two obvious candidate repairs on the CONTROL arm only:

  A. ANCHORED: continue P0's reconstruction objective through P1, symmetrically in both
     arms, so the latent stays readable while still receiving the channel gradients.
  B. SHORT-UNFREEZE: unfreeze for only the last K episodes of P1, so the encoder receives
     channel gradient but drifts less far from the reconstruction solution.

Neither is queued from here; the point is to say whether an answerable H1 exists at all.
Scratch; not an experiment; writes no manifest.
"""
import sys, statistics, torch
import torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import experiments._scratch.arc021_h1_encoder_unfrozen_drive_axis_instrument as D

SEEDS = [1001, 1008, 1015]
ARM = "SEPARATED"       # the CONTROL arm: if it cannot express the DV, nothing else matters


def run(mode, seed, cond="DENSE"):
    import random as _r
    from experiments._lib.arm_fingerprint import reset_all_rng
    reset_all_rng(seed)
    env = D.SmallViewEnv(size=D.GRID_SIZE, num_hazards=D.NUM_HAZARDS[cond], seed=seed)
    enc, dec = D._make_world_codec(env.world_obs_dim)
    copt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=D.LR_ENCODER)
    D._run_p0(env, enc, dec, copt)
    ch = D._make_channels(ARM, env.action_dim, enc)

    if mode == "anchored":
        # Continue the P0 reconstruction objective through P1, on each channel's own
        # encoder, with that channel's own decoder copy. Symmetric across arms by
        # construction: every encoder in the arm gets exactly one recon term.
        decs = {}
        for name in ("enc_sensory", "enc_forward", "enc_harm"):
            d2 = __import__("copy").deepcopy(dec)
            decs[name] = d2
            opt_name = {"enc_sensory": "opt_sensory", "enc_forward": "opt_forward",
                        "enc_harm": "opt_harm"}[name]
            getattr(ch, opt_name).add_param_group({"params": list(d2.parameters()),
                                                   "lr": D.LR_ENCODER})
        orig_step = ch.train_step
        def anchored_step(w0, a, w1, lbl, _c=ch, _d=decs):
            out = orig_step(w0, a, w1, lbl)
            for name, d2 in _d.items():
                e = getattr(_c, name)
                opt = {"enc_sensory": _c.opt_sensory, "enc_forward": _c.opt_forward,
                       "enc_harm": _c.opt_harm}[name]
                loss = F.mse_loss(d2(e(w0)), w0)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(list(e.parameters()) + list(d2.parameters()), 1.0)
                opt.step()
            return out
        ch.train_step = anchored_step

    unfreeze_from = 0
    if mode == "short":
        unfreeze_from = D.P1_EPISODES - 10        # unfrozen for the LAST 10 of 80 episodes
    if mode in ("short",):
        orig_encode = ch._encode
        state = {"ep": 0}
        def gated_encode(w0, w1, _c=ch, _s=state):
            if _s["ep"] < unfreeze_from:
                with torch.no_grad():
                    return (_c.enc_sensory(w0), _c.enc_forward(w0),
                            _c.enc_forward(w1), _c.enc_harm(w0))
            return orig_encode(w0, w1)
        ch._encode = gated_encode

    ch.train_mode()
    import random as _rr
    for ep in range(D.P1_EPISODES):
        if mode == "short":
            state["ep"] = ep
        for w0, ai, w1, h in D._collect_random_episode(env):
            lbl = torch.ones(1, 1) if h < 0 else torch.zeros(1, 1)
            ch.train_step(w0.unsqueeze(0), D._action_onehot(ai, env.action_dim),
                          w1.unsqueeze(0), lbl)
    ch.eval_mode()
    st = D._eval_probes(env, ch.harm_encoder, ch, env.action_dim)
    return st["calibration_gap"], st["attribution_auc"]


print(f"CONTROL ARM ({ARM}), DENSE, {len(SEEDS)} seeds, full P0=80/P1=80 schedule")
print(f"floor on the control gap: SEPARATED_SIGNAL_FLOOR = {D.SEPARATED_SIGNAL_FLOOR}; "
      f"chance AUC = 0.5, MARGIN_AUC = {D.MARGIN_AUC}\n")
print(f"{'mode':12s} {'mean_gap':>10s} {'mean_auc':>10s}  per-seed gaps")
for mode in ("unfrozen", "anchored", "short"):
    gs, as_ = [], []
    for s in SEEDS:
        g, a = run(mode, s)
        gs.append(g); as_.append(a)
    ok = statistics.fmean(gs) >= D.SEPARATED_SIGNAL_FLOOR
    print(f"{mode:12s} {statistics.fmean(gs):+10.5f} {statistics.fmean(as_):10.5f}  "
          f"{[round(x,4) for x in gs]}   {'CLEARS FLOOR' if ok else 'BELOW FLOOR'}", flush=True)
