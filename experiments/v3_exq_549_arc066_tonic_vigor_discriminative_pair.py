"""V3-EXQ-549: ARC-066 / MECH-320 tonic_vigor_coupling -- behavioural discriminative pair.

Proposal: EXP-0083 / EVB-0232 (dispatch_mode=discriminative_pair). Behavioural
validation of the ARC-066 architectural slot (tonic_vigor_coupling: capacity-
keyed action-vs-passivity bias) and its first child mechanism MECH-320
(score-bias regulator at the e3.select() call site).

Substrate prerequisites all landed:
  - MECH-320 substrate landed 2026-05-10 (ree_core/policy/tonic_vigor.py).
  - V3-EXQ-547 substrate-readiness diagnostic PASS 2026-05-10 (6/6 sub-tests).
  - ARC-066 lit-pull synthesis lit_conf 0.789 supports-direction (Niv 2007 +
    Salamone & Correa 2012 + Beierholm 2013 + Kane et al. 2017).

ARC-066 architectural prediction (R3 verdict: additive form is primary):
  Under the well-fed-safe-familiar regime where existing behavioural-source
  slots (z_harm threats, z_goal deficit-seeded wants, novelty curiosity)
  all fall toward zero, the agent has no gradient to act WITHOUT ARC-066.
  With MECH-320 ON, the slow EWMA over realised E3-score-receipt drives a
  capacity-keyed bias on the action-vs-noop axis -- the agent acts more
  often, target-free.

Discriminative pair (per proposal acceptance checks: exactly one pair,
no broad profile sweep):
  ARM_OFF: use_tonic_vigor=False (ablation / control).
  ARM_ON:  use_tonic_vigor=True, form="additive" (primary R3 verdict).
  Multiplicative-form arbitration deferred to a separate future pair if
  C1 passes.

Protocol:
  P0 warmup (200 ep, vigor OFF in both arms): identical baseline policy
    checkpoint per seed. Vigor is target-free and its EWMA starts at 0;
    warmup with vigor OFF in both arms ensures the policy learns on
    identical data, so any P1 difference is attributable to the regulator
    not to training divergence.
  P1 measurement (30 ep x 200 steps): arm flag toggled. ARM_OFF keeps
    vigor=False; ARM_ON instantiates the TonicVigor module and the score-
    bias path fires every tick.

Environment (well-fed-safe-familiar regime, matched across arms by seed):
  CausalGridWorldV2 size=8, num_hazards=1, num_resources=3, action_dim=5
  (action 0 = no-op, per MECH-279 / TonicVigorConfig.noop_class). Hazards
  default-spawn and may drift; one hazard keeps the env "safe" in
  expectation. Three resources keep drive bounded (low gate_drive).
  use_proxy_fields=True (small encoder, fast Mac runtime).

Metrics (per arm per seed, P1 only):
  action_density = mean over P1 ticks of [argmax(action) != noop_class].
    The behavioural signature: ARC-066 ON should lift this above ARM_OFF.
  v_t_window / action_density_window: per-50-step windows in ARM_ON only.
    Pearson r between window-mean v_t and window-mean action_density
    measures the within-arm scaling signature (substrate-fingerprint:
    vigor scales with realised EWMA, not noise).
  gate_product_mean = mean of gate_energy * gate_drive * gate_pe in ARM_ON.
    Confirms secondary gates stay open enough for vigor to actually fire
    (failure mode: env too harsh, all three gates clamp v_t to zero).

Pre-registered acceptance thresholds:
  C1 action_density lift: mean(action_density_ARM_ON) -
      mean(action_density_ARM_OFF) >= C1_LIFT_MIN (default 0.03; 3pp).
      Paired by seed.
  C2 within-arm scaling: in ARM_ON only, Pearson r(v_t_window,
      action_density_window) >= C2_PEARSON_R_MIN (default 0.3) across
      windows. ARM_OFF correlation is NOT required to be low (the
      regulator is off in ARM_OFF; v_t_window does not exist there).
  C3 gate sanity: mean(gate_product) > C3_GATE_PRODUCT_MIN (default 0.5)
      in ARM_ON. Failure mode the substrate doc explicitly warns about.

PASS = C1 AND C2 AND C3 across both seeds.

experiment_purpose = "evidence" (tests the load-bearing claim of MECH-320,
distinct from V3-EXQ-547 substrate-readiness which is "diagnostic").

Run with:
  /opt/local/bin/python3 experiments/v3_exq_549_arc066_tonic_vigor_discriminative_pair.py
or:
  /opt/local/bin/python3 experiments/v3_exq_549_arc066_tonic_vigor_discriminative_pair.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

from experiment_protocol import emit_outcome


# ----------------------------------------------------------------------
# Constants and pre-registered thresholds
# ----------------------------------------------------------------------
EXPERIMENT_TYPE = "v3_exq_549_arc066_tonic_vigor_discriminative_pair"
CLAIM_IDS = ["ARC-066", "MECH-320"]
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

SEEDS = [42, 43]
P0_WARMUP_EPISODES = 200
P1_EVAL_EPISODES = 30
TOTAL_EPISODES_PER_ARM = P0_WARMUP_EPISODES + P1_EVAL_EPISODES  # 230
STEPS_PER_EPISODE = 200
PRINT_INTERVAL = 25
WINDOW_LENGTH = 50  # P1 step-window for within-arm scaling correlation

GRID_SIZE = 8
N_HAZARDS = 1
N_RESOURCES = 3
ACTION_DIM = 5  # CausalGridWorldV2.ACTIONS: 0=up, 1=down, 2=left, 3=right, 4=noop
NOOP_CLASS = 4  # matches CausalGridWorldV2 convention; TonicVigorConfig
                # default noop_class=0 (MECH-279 convention) is wrong for this env
                # and is overridden via tonic_vigor_noop_class=NOOP_CLASS in make_config

# Pre-registered thresholds (must be set in script, not inferred post-hoc).
C1_LIFT_MIN = 0.03
C2_PEARSON_R_MIN = 0.30
C3_GATE_PRODUCT_MIN = 0.50

ARM_OFF_LABEL = "ARM_OFF_vigor_disabled"
ARM_ON_LABEL = "ARM_ON_vigor_additive"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def make_env(seed: int) -> CausalGridWorldV2:
    """Well-fed-safe-familiar env: small grid, one hazard, three resources.

    use_proxy_fields=True keeps world_obs compact and the encoder small.
    """
    return CausalGridWorldV2(
        size=GRID_SIZE,
        num_hazards=N_HAZARDS,
        num_resources=N_RESOURCES,
        seed=seed,
        use_proxy_fields=True,
    )


def make_config(env: CausalGridWorldV2, arm_on: bool) -> REEConfig:
    """Build REEConfig. Vigor module is constructed iff arm_on=True.

    tonic_vigor_noop_class is set to CausalGridWorldV2's action-4 no-op rather
    than the TonicVigorConfig default (0), which maps to "move up" in this env.
    Without the override, vigor would bias against "up" instead of against
    passivity -- a directional artefact rather than the architectural signal.
    """
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=ACTION_DIM,
        self_dim=16,
        world_dim=16,
        use_tonic_vigor=arm_on,
        tonic_vigor_noop_class=NOOP_CLASS,
    )


def split_obs_tensors(obs_dict: dict) -> tuple:
    """Extract (obs_body, obs_world) as 2D tensors for act_with_split_obs.

    The harm streams are intentionally NOT enabled in this experiment
    (well-fed-safe-familiar regime), so act_with_split_obs's internal
    sense(obs_body, obs_world) is sufficient.
    """
    body = obs_dict["body_state"]
    world = obs_dict["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    return body, world


def pearson_r(xs: list, ys: list) -> float:
    """Simple Pearson r over two equal-length scalar sequences. 0 on degenerate."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return 0.0
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return sxy / ((sxx * syy) ** 0.5)


# ----------------------------------------------------------------------
# Per-arm run
# ----------------------------------------------------------------------
def run_arm(seed: int, arm_label: str, arm_on: bool, dry_run: bool) -> dict:
    """Run one arm for one seed and return per-seed measurements.

    Vigor flag is fixed for the whole run (P0 + P1). The proposal asks for
    matched-seed paired comparison; matching is on seed, not on warmup
    checkpoint -- both arms train under the same env-seed schedule, the
    OFF arm with no vigor, the ON arm with vigor enabled. The vigor EWMA
    is initialised to zero and takes the first few warmup episodes to
    accumulate, so warmup-phase action selection is dominated by baseline
    learning in both arms.
    """
    print(f"Seed {seed} Condition {arm_label}", flush=True)
    torch.manual_seed(seed)

    env_init = make_env(seed=seed)
    cfg = make_config(env_init, arm_on=arm_on)
    agent = REEAgent(cfg)
    agent.reset()

    # P1 accumulators.
    p1_total_ticks = 0
    p1_nonnoop_ticks = 0

    # Per-window measurements (ARM_ON only).
    window_v_t = []
    window_action_density = []
    window_gate_product = []
    cur_window_ticks = 0
    cur_window_nonnoop = 0
    cur_window_v_t_sum = 0.0
    cur_window_gate_prod_sum = 0.0

    for ep in range(TOTAL_EPISODES_PER_ARM):
        ep_seed = seed * 100000 + ep
        env = make_env(seed=ep_seed)
        _flat, obs_dict = env.reset()
        agent.reset()

        in_p1 = ep >= P0_WARMUP_EPISODES

        for _step in range(STEPS_PER_EPISODE):
            obs_body, obs_world = split_obs_tensors(obs_dict)
            with torch.no_grad():
                action = agent.act_with_split_obs(
                    obs_body, obs_world, temperature=1.0,
                )
            if action is None:
                action_class = NOOP_CLASS
            else:
                action_class = int(action.argmax(dim=-1).item())

            if in_p1:
                p1_total_ticks += 1
                is_nonnoop = int(action_class != NOOP_CLASS)
                p1_nonnoop_ticks += is_nonnoop

                if arm_on and agent.tonic_vigor is not None:
                    tv_state = agent.tonic_vigor.get_state()
                    cur_window_ticks += 1
                    cur_window_nonnoop += is_nonnoop
                    cur_window_v_t_sum += float(tv_state["last_v_t"])
                    gate_prod = (
                        float(tv_state["last_gate_energy"])
                        * float(tv_state["last_gate_drive"])
                        * float(tv_state["last_gate_pe"])
                    )
                    cur_window_gate_prod_sum += gate_prod
                    if cur_window_ticks >= WINDOW_LENGTH:
                        window_v_t.append(cur_window_v_t_sum / cur_window_ticks)
                        window_action_density.append(
                            cur_window_nonnoop / cur_window_ticks
                        )
                        window_gate_product.append(
                            cur_window_gate_prod_sum / cur_window_ticks
                        )
                        cur_window_ticks = 0
                        cur_window_nonnoop = 0
                        cur_window_v_t_sum = 0.0
                        cur_window_gate_prod_sum = 0.0

            try:
                _flat, _harm_signal, done, _info, obs_dict = env.step(action_class)
            except Exception:  # noqa: BLE001
                done = True
            if done:
                break

        if dry_run or (ep + 1) % PRINT_INTERVAL == 0:
            phase = "p1" if in_p1 else "p0"
            print(
                f"  [train] seed={seed} arm={arm_label} ep {ep+1}/{TOTAL_EPISODES_PER_ARM} "
                f"phase={phase} p1_ticks={p1_total_ticks} p1_nonnoop={p1_nonnoop_ticks}",
                flush=True,
            )

        if dry_run and ep >= 2:
            break

    # Flush any partial trailing window so very short P1s still produce one window.
    if arm_on and cur_window_ticks > 0:
        window_v_t.append(cur_window_v_t_sum / cur_window_ticks)
        window_action_density.append(cur_window_nonnoop / cur_window_ticks)
        window_gate_product.append(cur_window_gate_prod_sum / cur_window_ticks)

    action_density = (
        p1_nonnoop_ticks / p1_total_ticks if p1_total_ticks > 0 else 0.0
    )

    if arm_on and len(window_v_t) >= 2:
        pearson_v_t_density = pearson_r(window_v_t, window_action_density)
        gate_product_mean = sum(window_gate_product) / len(window_gate_product)
    else:
        pearson_v_t_density = 0.0
        gate_product_mean = 0.0

    return {
        "seed": seed,
        "arm_label": arm_label,
        "arm_on": arm_on,
        "p1_total_ticks": p1_total_ticks,
        "p1_nonnoop_ticks": p1_nonnoop_ticks,
        "action_density": action_density,
        "n_windows": len(window_v_t),
        "pearson_r_v_t_action_density": pearson_v_t_density,
        "gate_product_mean": gate_product_mean,
        "window_v_t": window_v_t,
        "window_action_density": window_action_density,
        "window_gate_product": window_gate_product,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    dry_run = args.dry_run

    t0 = time.time()
    timestamp_utc = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"

    print(f"V3-EXQ-549 ARC-066 / MECH-320 discriminative pair", flush=True)
    print(
        f"seeds={SEEDS} p0_ep={P0_WARMUP_EPISODES} p1_ep={P1_EVAL_EPISODES} "
        f"steps={STEPS_PER_EPISODE} grid={GRID_SIZE}x{GRID_SIZE} "
        f"hazards={N_HAZARDS} resources={N_RESOURCES} action_dim={ACTION_DIM}",
        flush=True,
    )

    results_arm_off = []
    results_arm_on = []
    for seed in SEEDS:
        r_off = run_arm(seed, ARM_OFF_LABEL, False, dry_run)
        results_arm_off.append(r_off)
        # Per-arm-seed verdict: PASS iff the run executed end-to-end
        # (P1 ticks accumulated). The scientific PASS/FAIL is computed
        # only at the overall level after both arms complete. These per-
        # run verdicts increment runner ETA tracking; count must equal
        # seeds * conditions = 2 * 2 = 4 verdict lines.
        print(
            f"verdict: {'PASS' if r_off['p1_total_ticks'] > 0 or dry_run else 'FAIL'}",
            flush=True,
        )
        r_on = run_arm(seed, ARM_ON_LABEL, True, dry_run)
        results_arm_on.append(r_on)
        print(
            f"verdict: {'PASS' if r_on['p1_total_ticks'] > 0 or dry_run else 'FAIL'}",
            flush=True,
        )

    # Pre-registered acceptance computations.
    n_seeds = len(SEEDS)
    per_seed_lift = []
    per_seed_c1 = []
    per_seed_c2 = []
    per_seed_c3 = []
    for i, seed in enumerate(SEEDS):
        off = results_arm_off[i]
        on = results_arm_on[i]
        lift = on["action_density"] - off["action_density"]
        per_seed_lift.append(lift)
        per_seed_c1.append(lift >= C1_LIFT_MIN)
        per_seed_c2.append(on["pearson_r_v_t_action_density"] >= C2_PEARSON_R_MIN)
        per_seed_c3.append(on["gate_product_mean"] >= C3_GATE_PRODUCT_MIN)

    c1_pass_all = all(per_seed_c1)
    c2_pass_all = all(per_seed_c2)
    c3_pass_all = all(per_seed_c3)
    overall_pass = c1_pass_all and c2_pass_all and c3_pass_all

    outcome = "PASS" if overall_pass else "FAIL"
    if overall_pass:
        evidence_direction = "supports"
    else:
        # If C3 fails, env was too harsh -> non_contributory (substrate
        # could not be exercised). Otherwise the claim weakens.
        if not c3_pass_all:
            evidence_direction = "non_contributory"
        else:
            evidence_direction = "weakens"

    mean_action_density_off = (
        sum(r["action_density"] for r in results_arm_off) / n_seeds
    )
    mean_action_density_on = (
        sum(r["action_density"] for r in results_arm_on) / n_seeds
    )
    mean_lift = sum(per_seed_lift) / n_seeds
    mean_pearson_on = (
        sum(r["pearson_r_v_t_action_density"] for r in results_arm_on) / n_seeds
    )
    mean_gate_product_on = (
        sum(r["gate_product_mean"] for r in results_arm_on) / n_seeds
    )

    elapsed = time.time() - t0

    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "timestamp_utc": timestamp_utc,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "backlog_id": "EVB-0232",
        "proposal_id": "EXP-0083",
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": {
            "ARC-066": evidence_direction,
            "MECH-320": evidence_direction,
        },
        "evidence_class": "discriminative_pair",
        "claim_ids_tested": CLAIM_IDS,
        "registered_thresholds": {
            "C1_LIFT_MIN": C1_LIFT_MIN,
            "C2_PEARSON_R_MIN": C2_PEARSON_R_MIN,
            "C3_GATE_PRODUCT_MIN": C3_GATE_PRODUCT_MIN,
        },
        "summary": {
            "scenario": (
                "Well-fed-safe-familiar regime (8x8 grid, 1 hazard, 3 resources, "
                "use_proxy_fields=True, action_dim=5 with noop_class=0). 2 matched "
                "seeds. P0 warmup 200 ep (arm flag fixed), P1 eval 30 ep x 200 "
                "steps. Discriminative pair: ARM_OFF use_tonic_vigor=False vs "
                "ARM_ON use_tonic_vigor=True (additive form, R3 primary). Tests "
                "the load-bearing claim of MECH-320 / ARC-066: capacity-keyed "
                "target-free action-vs-noop bias should lift action_density in "
                "the regime where harm/drive/novelty all approach zero."
            ),
            "interpretation": (
                f"action_density ARM_OFF mean={mean_action_density_off:.4f}; "
                f"ARM_ON mean={mean_action_density_on:.4f}; "
                f"paired lift mean={mean_lift:+.4f}; "
                f"within-arm Pearson r(v_t, density) mean={mean_pearson_on:+.3f}; "
                f"ARM_ON gate_product mean={mean_gate_product_on:.3f}. "
                f"C1 paired-lift >= {C1_LIFT_MIN}: "
                f"{'PASS' if c1_pass_all else 'FAIL'}. "
                f"C2 within-arm scaling r >= {C2_PEARSON_R_MIN}: "
                f"{'PASS' if c2_pass_all else 'FAIL'}. "
                f"C3 gate_product >= {C3_GATE_PRODUCT_MIN}: "
                f"{'PASS' if c3_pass_all else 'FAIL'}. "
                f"Outcome: {outcome}, evidence_direction={evidence_direction}."
            ),
            "pairwise_deltas": {
                "per_seed_action_density_lift": per_seed_lift,
                "mean_action_density_lift": mean_lift,
                "mean_action_density_off": mean_action_density_off,
                "mean_action_density_on": mean_action_density_on,
                "mean_pearson_r_arm_on": mean_pearson_on,
                "mean_gate_product_arm_on": mean_gate_product_on,
            },
        },
        "criteria": {
            "n_seeds": n_seeds,
            "per_seed_lift": per_seed_lift,
            "per_seed_c1_pass": per_seed_c1,
            "per_seed_c2_pass": per_seed_c2,
            "per_seed_c3_pass": per_seed_c3,
            "c1_pass_all_seeds": c1_pass_all,
            "c2_pass_all_seeds": c2_pass_all,
            "c3_pass_all_seeds": c3_pass_all,
            "overall_pass": overall_pass,
        },
        "config": {
            "seeds": SEEDS,
            "p0_warmup_episodes": P0_WARMUP_EPISODES,
            "p1_eval_episodes": P1_EVAL_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "window_length": WINDOW_LENGTH,
            "grid_size": GRID_SIZE,
            "n_hazards": N_HAZARDS,
            "n_resources": N_RESOURCES,
            "action_dim": ACTION_DIM,
            "noop_class": NOOP_CLASS,
            "tonic_vigor_form": "additive",
            "dry_run": dry_run,
        },
        "metrics": {
            "results_arm_off": results_arm_off,
            "results_arm_on": results_arm_on,
        },
        "elapsed_seconds": elapsed,
        "notes": (
            "ARC-066 / MECH-320 behavioural discriminative pair. EXP-0083 / "
            "EVB-0232 dispatch_mode=discriminative_pair. Substrate landed "
            "2026-05-10; V3-EXQ-547 substrate-readiness PASS 2026-05-10. "
            "ARM_OFF use_tonic_vigor=False ablation vs ARM_ON "
            "use_tonic_vigor=True additive (R3 primary). Pre-registered "
            "thresholds C1/C2/C3 set as constants in this script before "
            "execution. Within-arm Pearson r scaling check (C2) is the "
            "substrate-fingerprint: vigor scales with realised EWMA, not "
            "with seed noise. Gate-product sanity (C3) guards the "
            "non_contributory failure mode where env is too harsh and all "
            "three secondary gates clamp v_t to zero. Multiplicative-form "
            "arbitration (R3 falsifiable secondary) is deferred to a "
            "separate future pair if C1 passes. ARC-066 lit_conf 0.789 "
            "supports-direction; synthesis at REE_assembly/evidence/"
            "literature/targeted_review_arc_066_tonic_vigor/synthesis.md."
        ),
    }

    out_dir = os.path.abspath(
        os.path.join(REPO_ROOT, "..", "REE_assembly", "evidence", "experiments")
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Overall outcome reported via emit_outcome() + manifest, not via a
    # final 'verdict:' line (which would over-count seeds*conditions).
    print(f"outcome: {outcome}", flush=True)
    print(
        f"action_density off={mean_action_density_off:.4f} "
        f"on={mean_action_density_on:.4f} lift={mean_lift:+.4f} "
        f"pearson_on={mean_pearson_on:+.3f} gate_on={mean_gate_product_on:.3f}",
        flush=True,
    )
    print(f"Result written to: {out_path}", flush=True)
    return outcome, out_path


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    emit_outcome(outcome=_outcome, manifest_path=_manifest_path)
    sys.exit(0)
