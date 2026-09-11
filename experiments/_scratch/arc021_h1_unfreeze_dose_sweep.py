"""ARC-021 H1 SPIKE: the unfreeze-DOSE sweep.

WHY THIS EXISTS. The H1 (drive-axis) leg of the ARC-021 channel-separation portfolio was
REFUSED at design time on 2026-09-11 (record:
`REE_assembly/evidence/planning/arc021_h1_leg_refused_readout_dies_under_unfreeze_20260911.md`,
GFLAG-0264): unfreezing the encoder for all 80 P1 episodes collapses the CONTROL arm's own
`calibration_gap` from +0.2169 (frozen) to +0.0024 and its `attribution_auc` to chance, in
BOTH arms, so a merge-vs-separate contrast measures the death of the readout rather than
channel separation. A plumbing bug was ruled out by re-freezing the same driver.

Probe 5 (`arc021_h1_anchor_probe.py`) then measured only the two ENDS of the dose axis:
80-of-80 episodes unfrozen is dead (+0.00178 / 0.499); 10-of-80 clears the floor (+0.21158 /
0.675). Nothing in between was measured. This sweep fills that in, and -- crucially -- adds
the second half of the question that probe 5 never asked: at a dose where the readout
SURVIVES, is the manipulation still REAL?

DELIVERABLE IS KNOWLEDGE, NOT A BUILD. This is a `complex (probe-gated)` spike. It queues
nothing, writes no manifest, consumes no EXQ number and never enters the evidence record.

===========================================================================================
PRE-REGISTERED DECISION CRITERIA -- FIXED IN THIS FILE BEFORE THE SWEEP WAS RUN
===========================================================================================

THE TRAP THIS SECTION EXISTS TO CLOSE. A dose chosen BECAUSE it keeps the control arm above
its own floor is a TUNED KNOB, not a design choice, and pre-registering an H1 criterion at
such a dose would be tuning the design toward a pass. So both halves of the verdict are
written down here, in constants, before any number was seen; the control-arm signal is
reported at EVERY dose measured, INCLUDING the failing ones; and the merge-vs-separate
contrast is reported but is explicitly NOT an input to dose selection (if that contrast
varies with dose, that is EVIDENCE THE CONTRAST IS A TEST OF THE DOSE -- the exact thing this
spike is asking about -- and using it to pick the dose would destroy the finding).

(S) DOES THE INSTRUMENT SURVIVE. Not invented here: these are the driver's OWN pre-registered
    gates, computed with the driver's OWN constants over the control arm exactly as
    `_build_preconditions` computes them, so the sweep cannot drift away from the science it
    is characterising.
      S1  non-degeneracy, per condition: mean(control calibration_gap) >= SEPARATED_SIGNAL_FLOOR (0.20)
      S2  dv_headroom_margin_room_below_control:      range(control gaps)   >= 2.0 * MARGIN (0.30)
      S3  dv_headroom_control_signal_floor_reachable: max_abs(control gaps) >= 2.0 * SEPARATED_SIGNAL_FLOOR (0.40)
      S4  dv_headroom_auc_room_below_control:         min(control aucs) - 0.5 >= MARGIN_AUC (0.105)
    A dose is SURVIVING iff S1 (both conditions) and S2 and S3 and S4 all hold.

(M) IS THE MANIPULATION STILL REAL. A dose so small the encoder barely moves is not an
    encoder-level merge at all -- it is the frozen-encoder regime V3-EXQ-993a/1011 already
    measured, wearing a different label. Three statistics, all in FUNCTION space on a fixed
    64-observation probe batch and all normalised by ||z_P0||_F so they are comparable across
    seeds and conditions:
      M-A  merged_fn_drift_rel   = ||z_end(MERGED.encoder) - z_P0|| / ||z_P0||
           How far the MERGED arm's shared encoder actually moved from its P0 solution.
      M-B  cross_arm_fn_div_rel  = ||z_end(MERGED.encoder) - z_end(SEPARATED.enc_harm)|| / ||z_P0||
           How different the two arms' HARM-READOUT representations actually are. This is the
           treatment-vs-control difference the whole leg rests on; at dose 0 it is exactly 0.
      M-C  merge_vs_channel_ratio = M-B / interchannel_fn_div_rel(SEPARATED)
           where interchannel = ||z(enc_harm) - z(enc_sensory)|| / ||z_P0|| WITHIN the control
           arm: two encoders from the identical P0 start, trained the same number of steps on
           DIFFERENT channel objectives. M-C is therefore INTERNALLY CALIBRATED -- it asks
           whether merging the encoder moves the harm readout at least as far as ordinary
           channel-specific shaping already does inside the control arm. No arbitrary constant.
    A dose is MANIPULATION-REAL iff  M-C >= MANIP_RATIO_FLOOR (1.0)  AND  M-A >= MANIP_DRIFT_FLOOR (0.05).
    M-A is a sanity floor only (the encoder must have moved at all); M-C is load-bearing.

(X) SUPPORTING, NOT A GATE: the cross-channel gradient actually delivered into the
    harm-readout encoder, decomposed on a fixed diagnostic batch as
    g_other = d(sensory_loss + forward_loss)/d(harm-readout encoder params) versus
    g_harm  = d(harm_loss)/d(same params). In ARM_SEPARATED g_other is STRUCTURALLY ZERO
    (enc_harm is in no other channel's graph) -- which is also this decomposition's own
    positive control: if it is not ~0 for SEPARATED, the measurement is wrong, not the arm.

THE SPIKE'S QUESTION, then, is exactly: does {doses that SURVIVE} intersect
{doses that are MANIPULATION-REAL}? A legitimate and complete answer is NO -- that would mean
H1 is unanswerable in this surrogate and the honest route to ARC-021's necessity half is the
H2 leg (substrate-blocked, GFLAG-0229), which is a real prioritisation finding, not a null.

AND IF THE ANSWER IS YES, THE DOSE IS STILL NOT SELECTED HERE. A surviving-and-real window
licenses a successor design; the dose inside it must be fixed on structural grounds or at a
knee of the dose-response curve, argued in the written record, never at the best-scoring cell.

===========================================================================================
DESIGN NOTES
===========================================================================================

DOSE = the number of TRAILING P1 episodes for which the encoder is unfrozen (0 = the frozen
reference regime, 80 = the leg as specified). Gating is applied by wrapping the arm's own
`_encode` in `torch.no_grad()` during frozen episodes -- generic over arms, and numerically
identical to probe 5's hand-written SEPARATED-only version (Linear/ReLU consume no RNG, and a
no_grad forward leaves encoder `.grad` at None so `clip_grad_norm_`/`Adam.step` skip it).

CELL CONSTRUCTION MIRRORS `_run_cell` EXACTLY -- `reset_all_rng(seed)` (which is precisely what
`arm_cell(do_reset=True)` does on enter), then env, then codec, then P0, then
`_make_channels`, then P1, then `_eval_probes` through `channels.harm_encoder`. The 993a
red-team F3 defect (a probe consuming the torch RNG differently before head construction, so
its numbers were a different head-init draw) is avoided by not subclassing anything and not
constructing any module outside that order. Every extra measurement this file takes is done
either under `torch.no_grad()` or via `torch.autograd.grad` -- neither consumes RNG -- and the
probe/diagnostic batches are built inside an RNG SANDBOX that saves and restores Python,
numpy, torch and the harness fallback RNG, so they perturb the cell's stream not at all.

REPRODUCTION CHECK built in: at DOSE=80, ARM_SEPARATED, the sweep re-measures probe 3's
regime on probe 3's own seeds, so its DENSE/SPARSE control means must land near probe 3's
+0.02144 / +0.01421. At DOSE=0 it must land near probe 4's frozen +0.21685. Both are printed.

DETACH DECISION UNCHANGED (refusal record 3e): sensory target sg(z), forward target under
no_grad, harm BCE differentiable into the encoder. This file does not touch it, so
`arc021_h1_detach_probe.py` does not need re-running.

Scratch; not an experiment; writes no manifest. Run:
  /opt/local/bin/python3 arc021_h1_unfreeze_dose_sweep.py [--pilot] [--shard i/n]
"""
import argparse
import contextlib
import copy
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import experiments._scratch.arc021_h1_encoder_unfrozen_drive_axis_instrument as D
from experiments._lib.arm_fingerprint import reset_all_rng

# ---- sweep grid -------------------------------------------------------------------------
DOSES = [0, 2, 5, 10, 15, 20, 30, 40, 60, 80]
ARMS = ("SEPARATED", "MERGED")
PROBE_SEEDS = [1001 + 7 * i for i in range(8)]   # probe 3's seeds; disjoint from D.SEEDS
assert not (set(PROBE_SEEDS) & set(D.SEEDS)), "sweep seeds must be disjoint from run seeds"

# ---- pre-registered manipulation-reality constants (see docstring) ----------------------
MANIP_RATIO_FLOOR = 1.0      # M-C, load-bearing
MANIP_DRIFT_FLOOR = 0.05     # M-A, sanity floor

PROBE_BATCH = 64
DIAG_MAX_POS = 16


# ---- RNG sandbox -------------------------------------------------------------------------
@contextlib.contextmanager
def _rng_sandbox(seed: int):
    """Save every RNG `reset_all_rng` touches, reseed to `seed`, restore on exit.

    Anything measured inside is a deterministic function of `seed` AND leaves the cell's own
    RNG stream bit-identical to what it would have been had this block not run at all.
    """
    st_py = random.getstate()
    st_torch = torch.get_rng_state()
    st_np = None
    try:
        import numpy as _np
        st_np = _np.random.get_state()
    except Exception:
        pass
    st_h = None
    try:
        from experiments import _harness as _h
        st_h = _h._action_random.getstate()
    except Exception:
        _h = None
    try:
        reset_all_rng(seed)
        yield
    finally:
        random.setstate(st_py)
        torch.set_rng_state(st_torch)
        if st_np is not None:
            try:
                import numpy as _np2
                _np2.random.set_state(st_np)
            except Exception:
                pass
        if st_h is not None:
            try:
                from experiments import _harness as _h2
                _h2._action_random.setstate(st_h)
            except Exception:
                pass


def _make_batches(condition: str, seed: int):
    """A fixed (w0, action_onehot, w1, harm_label) batch, identical for every cell of a given
    (condition, seed), built entirely inside the RNG sandbox.

    Stratified: every harm transition up to DIAG_MAX_POS, then the earliest non-harm ones, so
    the harm BCE gradient in the (X) decomposition is measured on a batch that actually
    contains positives instead of on whatever an unstratified slice happened to hold.
    """
    with _rng_sandbox(seed + 90_000):
        penv = D.SmallViewEnv(size=D.GRID_SIZE, num_hazards=D.NUM_HAZARDS[condition], seed=seed)
        pool: List[Tuple[Any, int, Any, float]] = []
        for _ in range(3):
            pool.extend(D._collect_random_episode(penv))
        action_dim = penv.action_dim
    pos = [t for t in pool if t[3] < 0][:DIAG_MAX_POS]
    neg = [t for t in pool if not (t[3] < 0)][: max(0, PROBE_BATCH - len(pos))]
    sel = pos + neg
    w0 = torch.cat([t[0].unsqueeze(0) for t in sel], dim=0)
    act = torch.cat([D._action_onehot(t[1], action_dim) for t in sel], dim=0)
    w1 = torch.cat([t[2].unsqueeze(0) for t in sel], dim=0)
    lbl = torch.tensor([[1.0 if t[3] < 0 else 0.0] for t in sel])
    return w0, act, w1, lbl, len(pos), len(sel)


# ---- measurement helpers -----------------------------------------------------------------
def _latents(encoder, w0: torch.Tensor) -> torch.Tensor:
    was_training = encoder.training
    encoder.eval()
    with torch.no_grad():
        z = encoder(w0).clone()
    if was_training:
        encoder.train()
    return z


def _rel_fro(a: torch.Tensor, b: torch.Tensor, denom: float) -> float:
    return float(torch.linalg.norm(a - b).item()) / denom if denom > 0 else float("nan")


def _param_drift_rel(mod_end, mod_p0) -> float:
    num = 0.0
    den = 0.0
    for pe, p0 in zip(mod_end.parameters(), mod_p0.parameters()):
        num += float(((pe.detach() - p0.detach()) ** 2).sum().item())
        den += float((p0.detach() ** 2).sum().item())
    return math.sqrt(num) / math.sqrt(den) if den > 0 else float("nan")


def _xgrad(channels, orig_encode, batch) -> Dict[str, Optional[float]]:
    """(X) Decompose the gradient arriving at the HARM-READOUT encoder into the harm channel's
    own contribution and the OTHER two channels' contribution, on a fixed batch.

    Uses `torch.autograd.grad`, so nothing is written to `.grad`, no optimizer state moves and
    no RNG is consumed -- this is a probe of the gradient FIELD, not an instrumentation of the
    training step, which is what keeps it from perturbing the run it measures.
    """
    w0, act, w1, lbl = batch
    z_s, z_f, z_f1, z_h = orig_encode(w0, w1)
    sensory_target = z_s.detach() if D.DETACH_MSE_TARGETS else z_s
    sensory_loss = F.mse_loss(channels.predict_sensory_from_z(z_s), sensory_target)
    forward_loss = F.mse_loss(channels.predict_forward_from_z(z_f, act), z_f1)
    harm_loss = F.binary_cross_entropy_with_logits(channels.harm_logit(z_h, act), lbl)
    params = [p for p in channels.harm_encoder.parameters()]
    g_h = torch.autograd.grad(harm_loss, params, retain_graph=True, allow_unused=True)
    g_o = torch.autograd.grad(sensory_loss + forward_loss, params,
                              retain_graph=False, allow_unused=True)

    def _flat(gs):
        parts = []
        for g, p in zip(gs, params):
            parts.append(torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1))
        return torch.cat(parts)

    fh, fo = _flat(g_h), _flat(g_o)
    nh, no_ = float(torch.linalg.norm(fh).item()), float(torch.linalg.norm(fo).item())
    cos = float((fh @ fo / (nh * no_)).item()) if nh > 0 and no_ > 0 else None
    return {
        "g_harm_norm": nh,
        "g_other_norm": no_,
        "g_other_over_harm": (no_ / nh) if nh > 0 else None,
        "cos_other_harm": cos,
    }


# ---- the dosed P1 ------------------------------------------------------------------------
def _run_p1_dosed(env, channels, action_dim: int, dose_k: int, batch) -> Dict[str, Any]:
    """`_run_p1`'s loop, verbatim in its order of operations and RNG consumption, with the
    encoder gated OFF for the first (P1_EPISODES - dose_k) episodes.

    Statistics are split FROZEN-phase vs UNFROZEN-phase: during a frozen episode the encoder
    parameters carry no grad, so the union `clip_grad_norm_` ranges over heads alone and the
    pooled figure would not be comparable with the refusal record's 18.6% / 44.2%.
    """
    unfreeze_from = D.P1_EPISODES - dose_k
    orig_encode = channels._encode
    state = {"frozen": True}

    def gated_encode(world_obs_t, world_obs_t1, _o=orig_encode, _s=state):
        if _s["frozen"]:
            with torch.no_grad():
                out = _o(world_obs_t, world_obs_t1)
            return tuple(t.detach() for t in out)
        return _o(world_obs_t, world_obs_t1)

    channels._encode = gated_encode
    channels.train_mode()

    total_harm_events = 0
    n_steps = n_unfrozen_steps = 0
    n_clipped_unfrozen = 0.0
    sum_pre_clip_unfrozen = 0.0
    sum_harm_loss = 0.0
    xgrad_onset: Optional[Dict[str, Optional[float]]] = None

    for ep in range(D.P1_EPISODES):
        if ep == unfreeze_from:
            # Measured with the UNGATED encode, at the instant the unfreeze begins: the
            # gradient field the encoder is about to be exposed to, at this dose.
            xgrad_onset = _xgrad(channels, orig_encode, batch)
        state["frozen"] = ep < unfreeze_from
        for world_obs_t, action_idx, world_obs_t1, harm_signal in D._collect_random_episode(env):
            action_onehot = D._action_onehot(action_idx, action_dim)
            is_harm = harm_signal < 0
            if is_harm:
                total_harm_events += 1
            harm_label = torch.ones(1, 1) if is_harm else torch.zeros(1, 1)
            stats = channels.train_step(
                world_obs_t.unsqueeze(0), action_onehot, world_obs_t1.unsqueeze(0), harm_label)
            n_steps += 1
            sum_harm_loss += stats["harm_loss"]
            if not state["frozen"]:
                n_unfrozen_steps += 1
                n_clipped_unfrozen += stats["clipped"]
                sum_pre_clip_unfrozen += stats["pre_clip_norm"]

    channels._encode = orig_encode
    xgrad_end = _xgrad(channels, orig_encode, batch)

    channels.eval_mode()
    with torch.no_grad():
        probe = torch.cat([env.reset()[1]["world_state"].unsqueeze(0) for _ in range(64)], dim=0)
        z = channels.harm_encoder(probe)
        zharm_dispersion = float(z.std(dim=0).mean().item())
    return {
        "p1_harm_events": total_harm_events,
        "p1_steps": n_steps,
        "p1_unfrozen_steps": n_unfrozen_steps,
        "clip_active_frac_unfrozen": (n_clipped_unfrozen / n_unfrozen_steps) if n_unfrozen_steps else None,
        "mean_pre_clip_norm_unfrozen": (sum_pre_clip_unfrozen / n_unfrozen_steps) if n_unfrozen_steps else None,
        "mean_harm_loss": sum_harm_loss / n_steps if n_steps else None,
        "zharm_dispersion": zharm_dispersion,
        "xgrad_at_onset": xgrad_onset,
        "xgrad_at_end": xgrad_end,
    }


def run_cell(condition: str, arm: str, seed: int, dose_k: int) -> Dict[str, Any]:
    reset_all_rng(seed)
    env = D.SmallViewEnv(size=D.GRID_SIZE, num_hazards=D.NUM_HAZARDS[condition], seed=seed)
    action_dim = env.action_dim
    encoder, decoder = D._make_world_codec(env.world_obs_dim)
    codec_opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                                 lr=D.LR_ENCODER)
    with contextlib.redirect_stdout(open("/dev/null", "w")):
        p0_stats = D._run_p0(env, encoder, decoder, codec_opt)
    enc_p0 = copy.deepcopy(encoder)          # consumes no RNG (see _clone_encoder's docstring)

    w0, act, w1, lbl, n_pos, n_batch = _make_batches(condition, seed)
    batch = (w0, act, w1, lbl)
    z_p0 = _latents(enc_p0, w0)
    denom = float(torch.linalg.norm(z_p0).item())

    channels = D._make_channels(arm, action_dim, encoder)
    p1_stats = _run_p1_dosed(env, channels, action_dim, dose_k, batch)

    harm_sensitivity = D._harm_action_sensitivity(channels.harm_encoder, channels, env, action_dim)
    probe_stats = D._eval_probes(env, channels.harm_encoder, channels, action_dim)

    z_end = _latents(channels.harm_encoder, w0)
    inter = None
    if arm == "SEPARATED":
        inter = _rel_fro(z_end, _latents(channels.enc_sensory, w0), denom)

    return {
        "condition": condition, "arm": arm, "seed": seed, "dose_k": dose_k,
        "mean_recon_loss": p0_stats["mean_recon_loss"],
        "harm_action_sensitivity": harm_sensitivity,
        "enc_param_drift_rel": _param_drift_rel(channels.harm_encoder, enc_p0),
        "enc_fn_drift_rel": _rel_fro(z_end, z_p0, denom),
        "interchannel_fn_div_rel": inter,
        "z_denom": denom,
        "diag_batch_n": n_batch, "diag_batch_pos": n_pos,
        "_z_end": [round(float(v), 7) for v in z_end.reshape(-1).tolist()],
        **p1_stats,
        **probe_stats,
    }


# ---- driver ------------------------------------------------------------------------------
def _cells(doses, seeds, conditions):
    out = []
    for dose in doses:
        for cond in conditions:
            for seed in seeds:
                for arm in ARMS:
                    out.append((cond, arm, seed, dose))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true",
                    help="3 doses x 2 arms x DENSE x 3 seeds, for a fast wiring check")
    ap.add_argument("--shard", default="1/1", help="i/n -- run only cells with index %% n == i-1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    doses, seeds, conditions = DOSES, PROBE_SEEDS, D.CONDITIONS
    if args.pilot:
        doses, seeds, conditions = [0, 10, 80], PROBE_SEEDS[:3], ["DENSE"]

    i, n = (int(x) for x in args.shard.split("/"))
    cells = [c for k, c in enumerate(_cells(doses, seeds, conditions)) if k % n == i - 1]
    out_path = Path(args.out) if args.out else Path(__file__).with_suffix(
        f".shard{i}of{n}{'.pilot' if args.pilot else ''}.json")

    print(f"dose sweep: {len(cells)} cells (shard {i}/{n}), doses={doses}, "
          f"seeds={seeds}, conditions={conditions}", flush=True)
    torch.set_num_threads(1)
    t0 = time.time()
    rows = []
    for k, (cond, arm, seed, dose) in enumerate(cells):
        t1 = time.time()
        row = run_cell(cond, arm, seed, dose)
        rows.append(row)
        g = row["calibration_gap"]
        print(f"[{k+1}/{len(cells)}] dose={dose:>2} {cond}_{arm} seed={seed} "
              f"gap={('%+.5f' % g) if g is not None else 'None'} "
              f"auc={('%.4f' % row['attribution_auc']) if row['attribution_auc'] is not None else 'None'} "
              f"fn_drift={row['enc_fn_drift_rel']:.5f} "
              f"clip_unfroz={row['clip_active_frac_unfrozen']} "
              f"({time.time()-t1:.1f}s)", flush=True)
        out_path.write_text(json.dumps(
            {"elapsed_s": time.time() - t0, "doses": doses, "seeds": seeds,
             "conditions": conditions, "shard": args.shard, "rows": rows}, indent=1, default=str))
    print(f"wrote {out_path} -- {len(rows)} cells in {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
