"""V3-EXQ-759 -- MECH-304 cue-specific conditioned safety prediction (representation-level).

WALL-INDEPENDENT CONFIRMING probe for MECH-304 ("Cue-specific conditioned safety
prediction"). MECH-304's ONLY prior touch, V3-EXQ-603h, was a BEHAVIOURAL survival
contrast (G_H hazard-median, INTACT-vs-LESION) that hit the engaged-but-insufficient
competence ceiling. The REPRESENTATION-level question -- does the ConditionedSafetyStore
emit a cue-DISCRIMINATING safety-prediction scalar -- was never run and needs no committed
action, so it passes INDEPENDENT of the competence wall
(behavioral_diversity_isolation:GAP-I; in-flight V3-EXQ-752..756). Precedent: the
functional-signature DVs of V3-EXQ-455/447/448 passed while the behavioural baseline was
monostrategy-locked (failure_autopsy_V3-EXQ-455a).

WHY WALL-INDEPENDENT: the DV is a READ-ONLY prediction scalar
(ConditionedSafetyStore.predict(z_world), conditioned_safety_store.py:116) read off a
FROZEN, action-free agent. Random-policy rollouts are used ONLY as a world-state sampler;
no action is committed, no goal pursued, no policy trained. The store is populated by the
substrate's real .update() path (the MECH-302 relief-completion teaching signal, supplied
here as event_fired=True on the conditioned-safe cue). Nothing in the DV depends on
committed-action diversity, so the competence wall cannot gate it.

MECHANISM UNDER TEST (SD-051 / MECH-304): a discrete predictive structure -- an EMA
prototype of z_world at relief-completion event ticks -- emits a cue-SPECIFIC safety
prediction (cosine similarity -> sigmoid). "Cue-specific" = the prediction is reliably
HIGHER for the cue that co-occurred with relief than for other (unpaired/neutral) cues --
the conditioned-inhibition read: a learned prediction attaches to the SPECIFIC cue, not a
blanket safety signal.

DV IS RANK-BASED (scale-free), NOT absolute magnitude. The store's predict is
sigmoid(gain * cosine); on the frozen z_world the cues share a large common-mode component,
so cosine (hence predict) sits near the sigmoid ceiling and absolute predict differences are
tiny (~1e-3). But the RANK-ORDER -- is the conditioned-safe cue's predict reliably above a
neutral cue's? -- is the substrate's actual cue-specific signal and is scale-free. The DV is
therefore the discrimination AUC = P(predict(safe) > predict(neutral)); AUC=0.5 is chance.
(Same lesson as V3-EXQ-746/746a: a narrow gradient demands scale-free thresholds; and the
V3-EXQ-643 same-statistic rule -- the readiness anchor asserts the SAME statistic, AUC, on a
positive control, NOT a centroid-cosine proxy that under-reports the store's discriminability.)

DESIGN (frozen agent, action-free, read-only DV):
  1. Build a FROZEN REEAgent with use_conditioned_safety_store=True (explicitly armed;
     alpha_world=0.9 for z_world fidelity, SD-008). No training, no action selection driving
     learning. z_world is obtained from the substrate's real encoder
     (agent.latent_stack.encode) statelessly per observation -- each observation is a
     discrete "cue presentation".
  2. Sample world states via a seeded random-policy rollout; partition by grid REGION.
     Each region = a distinct CUE CLASS (distinct local content). Split each region's
     instances into disjoint PAIRING / PROBE halves by index parity.
  3. Pavlovian pairing: designate the most-populated region the CONDITIONED-SAFE cue; feed
     its PAIRING-half z_world through the real store.update(z, event_fired=True) -- the cue
     co-occurs with the relief-completion teaching signal. Neutral cues are NOT reinforced.
  4. Probe (READ-ONLY): for the conditioned-safe cue and the pooled neutral cues, feed the
     held-out PROBE-half z_world through store.predict(z). DV = per-seed discrimination AUC.

ROUTING (preconditions met):
  PASS  (supports)   : mean AUC >= AUC_CONFIRM AND >= DIRECTIONAL_FRAC of seeds above chance
                       -- a genuine, cue-SPECIFIC safety-prediction on a frozen agent.
  FAIL  (weakens)    : reliably ANTI-discriminative (mean AUC <= AUC_ANTI AND >=
                       DIRECTIONAL_FRAC of seeds BELOW chance) -- the store ranks neutral
                       above the conditioned-safe cue, a genuine contradiction.
  FAIL  (non_contrib): otherwise (AUC hovering at chance / mixed) -- the store finds no
                       reliable cue structure to discriminate; requeue after z_world
                       enrichment. non_degenerate=False (excluded from scoring).

NON-VACUITY / READINESS GATE (self-route substrate_not_ready_requeue -> non_contributory,
NEVER a false weakens; lesson from V3-EXQ-688's vacuous null -- arm the flag + populate via
the real update path):
  - positive_control_auc (SAME statistic as C1): on KNOWN-discriminable inputs (a
    prototype-aligned class vs an orthogonal class) the store must achieve a high
    discrimination AUC. Below floor => store cannot discriminate at all => not ready.
  - store_prototype_populated_median: median post-pairing prototype L2 norm must exceed the
    store's min_norm (else predict()==0 for all cues).
  - predict_spread_nondegenerate: pooled predict() values must have non-trivial spread (a
    constant predict => degenerate range => not ready).
  - n_valid_seeds: enough seeds yielded >= 2 qualifying cue-class regions.

claim_ids = [MECH-304]; experiment_purpose = evidence; supersedes NOTHING; PROMOTES/DEMOTES
NOTHING (a PASS moves MECH-304 candidate/v3_pending toward provisional via governance). No
arm_results / no baseline mint: nothing is trained (the frozen agent is a deterministic
function of seed; there is no reusable OFF/baseline training cell).

GOV-REUSE-1: decisive readout = cross-cue-class AUC of ConditionedSafetyStore.predict.
Absent from every recorded manifest (603h was a behavioural survival contrast with no
predict() readout; pre-2026-07-12 manifests carry no substrate_hash) -> not recoverable -> run.
"""

import argparse
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._metrics import P0NotReady, check_degeneracy, p0_readiness_gate  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.safety import ConditionedSafetyStore  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_759_mech304_cue_specific_conditioned_safety_prediction"
QUEUE_ID = "V3-EXQ-759"
CLAIM_IDS = ["MECH-304"]
EXPERIMENT_PURPOSE = "evidence"

# --- Design constants (pre-registered) ---
SEEDS = [42, 7, 19, 3, 11, 23]
ROLLOUT_EPISODES = 6                 # world-state-sampling episodes per seed (the progress denom)
STEPS_PER_EP = 60
SELF_DIM = 32
WORLD_DIM = 32
ALPHA_WORLD = 0.9                    # SD-008: high z_world fidelity (default 0.3 under-differentiates)

REGION_DIV = 2                       # 2x2 = 4 coarse grid regions -> up to 4 cue classes
N_CUE_CLASSES = 3                    # 1 conditioned-safe + up to 2 neutral (uses fewer if fewer qualify)
MIN_REGION_INSTANCES = 12            # a region needs this many samples to be a usable cue class
POS_CONTROL_UPDATES = 30             # updates to drive the positive-control prototype
POS_CONTROL_PROBES = 40             # per-class probe instances for the positive-control AUC
POS_CONTROL_NOISE = 0.3            # within-class jitter for the positive control

# Pre-registered thresholds (NOT derived from the run's own statistics).
# The DV is scale-free: discrimination AUC of predict(safe) vs predict(neutral); 0.5 = chance.
AUC_CONFIRM = 0.65                  # load-bearing: mean AUC to confirm cue-specific discrimination
AUC_ANTI = 0.35                    # weakens band: mean AUC this low = reliable ANTI-discrimination
DIRECTIONAL_FRAC = 0.8             # fraction of valid seeds that must share the direction
POS_CONTROL_FLOOR_AUC = 0.90       # readiness: store AUC on known-discriminable inputs (same statistic)
PROTO_NORM_FLOOR = 0.10            # readiness: median post-pairing prototype L2 norm (store min_norm)
PREDICT_SPREAD_FLOOR = 1e-4        # readiness: pooled predict() std (constant predict => degenerate)

ENV_SEED_BASE = 80000
ACTION_SEED_BASE = 81000
POS_CONTROL_SEED = 82000

ENV_KWARGS: Dict[str, Any] = {
    "size": 12,
    "num_hazards": 5,      # richer layout -> distinct local content per region
    "num_resources": 5,
}


# ---------------------------------------------------------------------------- #
# Agent / encoder                                                              #
# ---------------------------------------------------------------------------- #
def _build_frozen_agent(seed: int) -> Tuple[REEAgent, CausalGridWorldV2]:
    """Frozen REEAgent (no grad, eval) with the conditioned safety store ARMED."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = CausalGridWorldV2(seed=ENV_SEED_BASE + seed, **ENV_KWARGS)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=ALPHA_WORLD,
    )
    # Explicitly arm the substrate under test (do NOT rely on default flag paths).
    cfg.use_conditioned_safety_store = True
    agent = REEAgent(cfg)
    if agent.conditioned_safety_store is None:
        raise RuntimeError("use_conditioned_safety_store armed but store is None -- wiring drift")
    for p in agent.parameters():
        p.requires_grad_(False)
    agent.eval()
    return agent, env


def _encode_zworld(agent: REEAgent, flat_obs: torch.Tensor) -> torch.Tensor:
    """Stateless z_world encode of one observation (the cue presentation). Frozen."""
    obs = flat_obs.reshape(1, -1).to(torch.float32)
    with torch.no_grad():
        latent = agent.latent_stack.encode(obs)
    return latent.z_world.detach().reshape(1, -1)


def _region_id(env: CausalGridWorldV2) -> int:
    """Coarse REGION_DIV x REGION_DIV grid-region id from the agent cell."""
    span = max(1, int(math.ceil(env.size / REGION_DIV)))
    rx = min(REGION_DIV - 1, int(env.agent_x) // span)
    ry = min(REGION_DIV - 1, int(env.agent_y) // span)
    return rx * REGION_DIV + ry


# ---------------------------------------------------------------------------- #
# Metric helpers                                                               #
# ---------------------------------------------------------------------------- #
def _auc(pos: List[float], neg: List[float]) -> float:
    """P(pos > neg), ties counted as 0.5. Returns 0.5 (chance) if either side is empty."""
    if not pos or not neg:
        return 0.5
    p = np.asarray(pos, dtype=np.float64)
    n = np.asarray(neg, dtype=np.float64)
    total = 0.0
    for v in p:
        total += float((v > n).sum()) + 0.5 * float((v == n).sum())
    return total / (len(p) * len(n))


def _centroid(vecs: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat(vecs, dim=0).mean(dim=0, keepdim=True)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    na = float(a.norm().item()) + 1e-8
    nb = float(b.norm().item()) + 1e-8
    return float(torch.dot(a, b).item()) / (na * nb)


# ---------------------------------------------------------------------------- #
# Positive control -- store CAN discriminate on genuinely-different inputs      #
# ---------------------------------------------------------------------------- #
def _positive_control_auc(ref_store: ConditionedSafetyStore) -> float:
    """Pair the store on class A (unit direction e0), then compute the discrimination AUC of
    predict(A-like) vs predict(B-like, orthogonal direction e1). SAME statistic (AUC) the
    load-bearing criterion routes on. A healthy store returns ~1.0; degenerate arithmetic ~0.5.
    """
    store = ConditionedSafetyStore(
        world_dim=ref_store.world_dim,
        ema_alpha=ref_store.ema_alpha,
        decay_rate=ref_store.decay_rate,
        min_norm=ref_store.min_norm,
        threshold=ref_store.threshold,
        gain=ref_store.gain,
    )
    rng = np.random.default_rng(POS_CONTROL_SEED)
    d = ref_store.world_dim
    a_dir = torch.zeros(1, d, dtype=torch.float32); a_dir[0, 0] = 1.0
    b_dir = torch.zeros(1, d, dtype=torch.float32); b_dir[0, 1] = 1.0

    def _jitter(base: torch.Tensor) -> torch.Tensor:
        noise = torch.tensor(rng.normal(0.0, POS_CONTROL_NOISE, size=(1, d)), dtype=torch.float32)
        v = base + noise
        return v / (v.norm() + 1e-8)

    for _ in range(POS_CONTROL_UPDATES):
        store.update(_jitter(a_dir), event_fired=True)
    a_pred = [float(store.predict(_jitter(a_dir))) for _ in range(POS_CONTROL_PROBES)]
    b_pred = [float(store.predict(_jitter(b_dir))) for _ in range(POS_CONTROL_PROBES)]
    return _auc(a_pred, b_pred)


# ---------------------------------------------------------------------------- #
# Per-seed probe                                                               #
# ---------------------------------------------------------------------------- #
def _run_seed(seed: int) -> Dict[str, Any]:
    """One seed: sample world states, partition into cue classes, pair the safe cue,
    probe safe vs pooled-neutral read-only. Returns per-seed metrics."""
    print(f"Seed {seed} Condition cue_probe", flush=True)
    agent, env = _build_frozen_agent(seed)
    act_rng = np.random.default_rng(ACTION_SEED_BASE + seed)
    action_dim = env.action_dim

    region_z: Dict[int, List[torch.Tensor]] = {}
    for ep in range(ROLLOUT_EPISODES):
        flat_obs, _obs_dict = env.reset()
        for _step in range(STEPS_PER_EP):
            z = _encode_zworld(agent, flat_obs)
            region_z.setdefault(_region_id(env), []).append(z)
            action = int(act_rng.integers(0, action_dim))
            flat_obs, _harm, done, _info, _obs2 = env.step(action)
            if done:
                break
        if (ep + 1) % 2 == 0 or ep + 1 == ROLLOUT_EPISODES:
            print(f"  [train] cue seed={seed} ep {ep + 1}/{ROLLOUT_EPISODES} "
                  f"regions={len(region_z)}", flush=True)

    qualifying = {r: zs for r, zs in region_z.items() if len(zs) >= MIN_REGION_INSTANCES}
    if len(qualifying) < 2:
        print(f"verdict: FAIL", flush=True)
        return {"seed": seed, "valid": False, "n_qualifying_regions": len(qualifying)}

    ranked = sorted(qualifying.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:N_CUE_CLASSES]
    safe_region, safe_zs = ranked[0]
    neutral = ranked[1:]

    def _parity_split(zs: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return ([z for i, z in enumerate(zs) if i % 2 == 0],
                [z for i, z in enumerate(zs) if i % 2 == 1])

    safe_pair, safe_probe = _parity_split(safe_zs)

    # Pavlovian pairing: reinforce ONLY the conditioned-safe cue via the real update path.
    store = agent.conditioned_safety_store
    store.reset()
    for z in safe_pair:
        store.update(z, event_fired=True)
    proto_norm = math.sqrt(sum(p * p for p in store._prototype))

    # Probe (READ-ONLY): conditioned-safe held-out vs pooled neutral held-out.
    safe_pred = [float(store.predict(z)) for z in safe_probe]
    neutral_pred: List[float] = []
    for _r, zs in neutral:
        _p, probe = _parity_split(zs)
        neutral_pred += [float(store.predict(z)) for z in probe]

    seed_auc = _auc(safe_pred, neutral_pred)
    safe_mean = float(np.mean(safe_pred)) if safe_pred else 0.0
    neutral_mean = float(np.mean(neutral_pred)) if neutral_pred else 0.0
    pooled = safe_pred + neutral_pred
    pooled_std = float(np.std(pooled)) if pooled else 0.0
    # Context only (NOT a gate): centroid-cosine separation under-reports the store's
    # discriminability -- kept for the record to show the store reads sub-centroid structure.
    cue_sep = 1.0 - max(_cosine(_centroid(safe_zs), _centroid(zs)) for _r, zs in neutral)

    print(f"verdict: {'PASS' if seed_auc > 0.5 else 'FAIL'}", flush=True)
    return {
        "seed": seed,
        "valid": True,
        "n_qualifying_regions": len(qualifying),
        "safe_region": safe_region,
        "neutral_regions": [r for r, _ in neutral],
        "auc": seed_auc,
        "safe_mean_predict": safe_mean,
        "neutral_mean_predict": neutral_mean,
        "mean_separation": safe_mean - neutral_mean,
        "pooled_predict_std": pooled_std,
        "proto_norm": proto_norm,
        "cue_zworld_separation": cue_sep,
        "n_safe_probe": len(safe_pred),
        "n_neutral_probe": len(neutral_pred),
    }


# ---------------------------------------------------------------------------- #
# Manifest                                                                     #
# ---------------------------------------------------------------------------- #
def _write_manifest(result: Dict[str, Any], started_at: float) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    full_config = {
        **ENV_KWARGS,
        "self_dim": SELF_DIM,
        "world_dim": WORLD_DIM,
        "alpha_world": ALPHA_WORLD,
        "rollout_episodes": ROLLOUT_EPISODES,
        "steps_per_ep": STEPS_PER_EP,
        "region_div": REGION_DIV,
        "n_cue_classes": N_CUE_CLASSES,
        "min_region_instances": MIN_REGION_INSTANCES,
        "auc_confirm": AUC_CONFIRM,
        "auc_anti": AUC_ANTI,
        "directional_frac": DIRECTIONAL_FRAC,
        "pos_control_floor_auc": POS_CONTROL_FLOOR_AUC,
        "proto_norm_floor": PROTO_NORM_FLOOR,
        "predict_spread_floor": PREDICT_SPREAD_FLOOR,
        "env_seed_base": ENV_SEED_BASE,
        "action_seed_base": ACTION_SEED_BASE,
        "use_conditioned_safety_store": True,
    }
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "timestamp_utc": timestamp,
        "interpretation": result["interpretation"],
        "non_degenerate": result.get("non_degenerate", True),
        "degeneracy_reason": result.get("degeneracy_reason", ""),
        "degenerate_metrics": result.get("degenerate_metrics", {}),
        "metrics": result.get("metrics", {}),
        "positive_control": result.get("positive_control", {}),
        "per_seed_results": result.get("per_seed_results", []),
        "wall_independence_note": (
            "DV is a read-only ConditionedSafetyStore.predict scalar on a frozen, "
            "action-free agent; random rollouts are only a world-state sampler. No "
            "committed action / goal / training -> independent of the competence wall "
            "(GAP-I). Precedent: functional-signature DVs 455/447/448."
        ),
        "gov_reuse_note": (
            "decisive readout = cross-cue-class predict() AUC; absent from all recorded "
            "manifests (603h behavioural survival contrast, no predict readout; "
            "pre-2026-07-12 manifests carry no substrate_hash) -> not recoverable -> run."
        ),
    }
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_path = write_flat_manifest(
        manifest, out_dir, config=full_config, seeds=SEEDS,
        script_path=Path(__file__), started_at=started_at,
    )
    return out_path


def _finish(result: Dict[str, Any], started_at: float, dry_run: bool,
            summary: str) -> Tuple[str, Any]:
    if dry_run:
        print(f"[dry-run] {summary}", flush=True)
        return result["outcome"], None
    out_path = _write_manifest(result, started_at)
    print(f"OUTCOME: {result['outcome']} ({result['evidence_direction']}) {summary}", flush=True)
    print(f"manifest: {out_path}", flush=True)
    return result["outcome"], out_path


def main(dry_run: bool = False) -> Tuple[str, Any]:
    started_at = time.perf_counter()
    seeds = SEEDS[:2] if dry_run else SEEDS

    per_seed = [_run_seed(s) for s in seeds]
    valid = [r for r in per_seed if r.get("valid")]

    # Positive control (store CAN discriminate on genuinely-different inputs).
    ref_agent, _ref_env = _build_frozen_agent(POS_CONTROL_SEED)
    pc_auc = _positive_control_auc(ref_agent.conditioned_safety_store)
    positive_control = {
        "discrimination_auc_on_orthogonal_classes": pc_auc,
        "floor": POS_CONTROL_FLOOR_AUC,
        "description": "pair on class A (e0); AUC of predict(A-like) vs predict(B-like, e1); same statistic as C1",
    }

    n_valid = len(valid)
    median_proto = statistics.median([r["proto_norm"] for r in valid]) if valid else 0.0
    median_spread = (
        statistics.median([r["pooled_predict_std"] for r in valid]) if valid else 0.0
    )
    min_valid_seeds = max(2, len(seeds) // 2)

    # ---- P0 readiness gate (self-route substrate_not_ready_requeue if unmet) ----
    try:
        preconditions = p0_readiness_gate([
            {"name": "positive_control_auc",
             "measured": pc_auc, "threshold": POS_CONTROL_FLOOR_AUC},
            {"name": "n_valid_seeds",
             "measured": float(n_valid), "threshold": float(min_valid_seeds)},
            {"name": "store_prototype_populated_median",
             "measured": median_proto, "threshold": PROTO_NORM_FLOOR},
            {"name": "predict_spread_nondegenerate",
             "measured": median_spread, "threshold": PREDICT_SPREAD_FLOOR},
        ])
    except P0NotReady as e:
        result = {
            "outcome": "FAIL",
            "evidence_direction": "non_contributory",
            "non_degenerate": False,
            "degeneracy_reason": e.reason,
            "degenerate_metrics": {},
            "interpretation": {
                "label": "substrate_not_ready_requeue",
                "preconditions": e.preconditions,
                "criteria": [{"name": "C1_cue_discrimination_auc",
                              "load_bearing": True, "passed": False}],
                "criteria_non_degenerate": {"C1": False},
            },
            "metrics": {"n_valid_seeds": n_valid, "median_proto_norm": median_proto,
                        "median_pooled_predict_std": median_spread, "positive_control_auc": pc_auc},
            "positive_control": positive_control,
            "per_seed_results": per_seed,
        }
        return _finish(result, started_at, dry_run,
                       "substrate_not_ready_requeue / non_contributory")

    # ---- Load-bearing criterion (preconditions met): scale-free AUC discrimination ----
    aucs = [r["auc"] for r in valid]
    mean_auc = float(np.mean(aucs))
    min_auc = float(np.min(aucs))
    max_auc = float(np.max(aucs))
    frac_above = sum(1 for a in aucs if a > 0.5) / len(aucs)
    frac_below = sum(1 for a in aucs if a < 0.5) / len(aucs)

    confirm = (mean_auc >= AUC_CONFIRM) and (frac_above >= DIRECTIONAL_FRAC)
    anti = (mean_auc <= AUC_ANTI) and (frac_below >= DIRECTIONAL_FRAC)

    degen = check_degeneracy({
        "cue_discrimination_auc": {"values": aucs},
        "pooled_predict_std": {"values": [r["pooled_predict_std"] for r in valid]},
    })

    if confirm:
        outcome, direction, label = "PASS", "supports", "cue_specific_prediction_confirmed"
        non_degen, degen_reason, degen_metrics = (
            degen["non_degenerate"], degen["degeneracy_reason"], degen["degenerate_metrics"])
    elif anti:
        outcome, direction, label = "FAIL", "weakens", "anti_cue_specific_prediction"
        non_degen, degen_reason, degen_metrics = (
            degen["non_degenerate"], degen["degeneracy_reason"], degen["degenerate_metrics"])
    else:
        # At chance / mixed: store finds no reliable cue structure -> non_contributory, requeue.
        outcome, direction, label = "FAIL", "non_contributory", "cue_discrimination_inconclusive_requeue"
        non_degen = False
        degen_reason = (f"AUC at chance / mixed (mean={mean_auc:.3f}, frac_above={frac_above:.2f}); "
                        "no reliable cue-specific discrimination on this z_world")
        degen_metrics = {"cue_discrimination_auc": degen_reason}

    result = {
        "outcome": outcome,
        "evidence_direction": direction,
        "non_degenerate": non_degen,
        "degeneracy_reason": degen_reason,
        "degenerate_metrics": degen_metrics,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria": [{"name": "C1_cue_discrimination_auc",
                          "load_bearing": True, "passed": bool(confirm)}],
            "criteria_non_degenerate": {"C1": bool(non_degen)},
        },
        "metrics": {
            "n_valid_seeds": n_valid,
            "mean_auc": mean_auc,
            "min_auc": min_auc,
            "max_auc": max_auc,
            "frac_seeds_above_chance": frac_above,
            "auc_confirm": AUC_CONFIRM,
            "directional_frac": DIRECTIONAL_FRAC,
            "median_proto_norm": median_proto,
            "median_pooled_predict_std": median_spread,
            "positive_control_auc": pc_auc,
        },
        "positive_control": positive_control,
        "per_seed_results": per_seed,
    }
    return _finish(result, started_at, dry_run,
                   f"mean_auc={mean_auc:.3f} min={min_auc:.3f} frac_above={frac_above:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    _outcome, _manifest_path = main(dry_run=args.dry_run)
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_manifest_path,
        dry_run=args.dry_run,
    )
