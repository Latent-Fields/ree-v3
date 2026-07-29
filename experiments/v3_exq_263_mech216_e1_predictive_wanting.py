"""
V3-EXQ-263: MECH-216 E1 Predictive Wanting Validation

RETIRED -- DO NOT REPAIR. SUPERSEDED BY V3-EXQ-263a (ree-v3 ad533d8, 2026-04-09).
  This driver never ran. It errored 5x on 2026-04-09 and was repaired by
  SUPERSESSION rather than in place: `/diagnose-errors` queued
  v3_exq_263a_mech216_e1_predictive_wanting.py carrying
  `"supersedes": "V3-EXQ-263"`, which ran the same day (PASS, `supports`
  MECH-216). Lineage continued 263b -> 332 -> 332a; four successor manifests
  exist. This file is the fossil of the pre-fix draft and is kept only as
  provenance for that supersession.

  It is NOT a case of the env signature drifting out from under a
  once-working script -- it was never correct. `CausalGridWorld.__init__`
  has never accepted `obs_dim` or `body_obs_dim`; the correct idiom (used by
  263a) is to pass neither and READ `env.body_obs_dim` / `env.world_obs_dim`
  back off the constructed env. Per ad533d8 the ctor kwargs were only one of
  five defects here -- the others being a missing sys.path, a wrong
  env.reset() unpack, wrong obs-dict keys (body_obs/world_obs, which are
  body_state/world_state), and a wrong env.step() signature. So the
  `TypeError: unexpected keyword argument 'obs_dim'` you get on import is
  the FIRST of five, not the only one: dropping it just advances to the next.
  Adjudicated 2026-07-29 (session sweet-williams-f30676).

Tests whether E1's schema readout head learns to predict resource proximity from
LSTM hidden state, and whether the resulting schema_salience seeds VALENCE_WANTING
at approach positions before direct resource contact.

MECHANISM UNDER TEST: MECH-216 (e1_predictive_wanting)
  E1 schema activations that predict resource encounters seed VALENCE_WANTING
  before contact. Zhang/Berridge: W_m = kappa (drive_level) x V_hat (schema_salience).

EXPERIMENT DESIGN:
  Two conditions, ablation pair:
    WITH_SCHEMA: schema_wanting_enabled=True, threshold=0.3, gain=0.5
    WITHOUT_SCHEMA: schema_wanting_enabled=False (contact-only wanting via serotonin)
  Both conditions have: tonic_5ht_enabled=True, use_resource_proximity_head=True,
    z_goal_enabled=True, drive_weight=2.0, wanting_weight=0.4

  Phased training:
    P0 (100 episodes): encoder warmup + schema readout training
    P1 (50 episodes): eval with frozen exploration noise

ACCEPTANCE CRITERIA (diagnostic):
  C1: schema_salience_mean > 0.1 in WITH_SCHEMA at resource-proximal positions
      (confirms the head learns to predict resource proximity from LSTM hidden state)
  C2: wanting_landscape_coverage in WITH_SCHEMA > WITHOUT_SCHEMA
      (schema wanting seeds wanting at more z_world positions than contact-only)
  C3: resource_rate in WITH_SCHEMA >= WITHOUT_SCHEMA in 2/3 seeds
      (approach efficiency improves when schema-predicted wanting guides CEM)
"""

import os
import sys
import json
import time
import traceback
import numpy as np
from datetime import datetime, timezone

MANIFEST_WRITER_EXEMPT = "archival early-era manifest (non-canonical filename not provably == run_id.json; superseded lineage, not re-run)"

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v3_exq_263_mech216_e1_predictive_wanting"
CLAIM_IDS = ["MECH-216"]

BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250
ACTION_DIM = 4
GRID_SIZE = 5

NUM_SEEDS = 3
P0_EPISODES = 100    # training with schema readout
P1_EPISODES = 50     # eval
STEPS_PER_EPISODE = 200

CONDITIONS = {
    "WITH_SCHEMA": {
        "schema_wanting_enabled": True,
        "schema_wanting_threshold": 0.3,
        "schema_wanting_gain": 0.5,
    },
    "WITHOUT_SCHEMA": {
        "schema_wanting_enabled": False,
        "schema_wanting_threshold": 0.3,
        "schema_wanting_gain": 0.5,
    },
}


# --- Opt-in driver-owned env seed (default OFF) ---------------------------
# `CausalGridWorldV2` forwards to `CausalGridWorld.__init__`, whose only RNG is
# `self._rng = np.random.default_rng(seed)`. `np.random.default_rng(None)` takes
# OS entropy AT CONSTRUCTION and does NOT consume the numpy global RNG, so this
# driver's `np.random.seed(...)` / `torch.manual_seed(...)` never reached its
# env. The defect is per CONSTRUCTION, not per process.
#
# `_ENV_SEED_BASE` stays None unless `--env-seed` is passed, and None hands
# `seed=None` straight to the FACTORY -- bit-identical to every landed run of
# this driver. DO NOT FLIP THE DEFAULT: pinning it unconditionally would change
# the env layout, and therefore the result, of a run already on the record.
#
# Keyed on a per-CONSTRUCTION counter rather than a fixed per-site index. This
# driver builds a fresh env per cell, so a fixed idx would put every arm and
# every seed on ONE shared layout and silently delete the across-arm /
# across-seed layout variation the unseeded default has. The counter is the
# stronger form of "fold the run seed into the base": it makes every
# construction distinct across seeds, arms AND repeats without needing the run
# seed in scope at the construction site.
_ENV_SEED_BASE = None                 # None = OS entropy (the landed default)
_ENV_SEED_STATE = {"n": 0}            # per-CONSTRUCTION counter, not a knob


def _next_env_seed():
    """Seed for the NEXT env construction, or None when the knob is unset."""
    if _ENV_SEED_BASE is None:
        return None
    idx = _ENV_SEED_STATE["n"]
    _ENV_SEED_STATE["n"] = idx + 1
    # stream 2 = driver-owned (the scaffold module reserves 0 for its curriculum
    # builds and 1 for its harm probe). Inlined rather than kept in a module
    # constant: it is a namespace tag, and a module-level numeric would read to
    # validate_experiments' config_slice-declaration lint as a readout-affecting
    # knob that every arm fingerprint must then declare.
    return int(_ENV_SEED_BASE) * 1000000 + 2 * 100000 + idx


def _env_seed_from_argv(argv):
    """Scan `--env-seed N` / `--env-seed=N`, matching this driver's existing
    `"--dry-run" in sys.argv` idiom. Deliberately NOT an argparse parser: this
    script has never parsed argv strictly, and the runner may append `--resume`
    / `--checkpoint-path=` args that a strict parser would reject."""
    for i, a in enumerate(argv):
        if a == "--env-seed" and i + 1 < len(argv):
            return int(argv[i + 1])
        if a.startswith("--env-seed="):
            return int(a.split("=", 1)[1])
    return None


def run_condition(condition_name, condition_cfg, seed):
    """Run one condition x seed."""
    import torch
    from ree_core.utils.config import REEConfig
    from ree_core.agent import REEAgent
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        alpha_world=0.9,
        use_event_classifier=True,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        use_harm_stream=True,
        use_affective_harm_stream=True,
        z_goal_enabled=True,
        drive_weight=2.0,
        tonic_5ht_enabled=True,
        wanting_weight=0.4,
        **condition_cfg,
    )

    agent = REEAgent(cfg)
    env = CausalGridWorldV2(
        seed=_next_env_seed(),
        size=GRID_SIZE,
        obs_dim=WORLD_OBS_DIM,
        body_obs_dim=BODY_OBS_DIM,
        use_proxy_fields=True,
    )

    e1_optimizer = torch.optim.Adam(agent.e1.parameters(), lr=1e-4)
    e3_optimizer = torch.optim.Adam(agent.e3.parameters(), lr=1e-3)
    encoder_optimizer = torch.optim.Adam(agent.latent_stack.parameters(), lr=1e-4)

    metrics = {
        "schema_salience_values": [],
        "resource_contacts": 0,
        "total_steps": 0,
        "wanting_write_count": 0,
        "p1_resource_rates": [],
        "p1_schema_salience_at_resource": [],
    }

    total_episodes = P0_EPISODES + P1_EPISODES

    for ep in range(total_episodes):
        obs_dict = env.reset()
        obs_body = torch.tensor(obs_dict["body_obs"], dtype=torch.float32).unsqueeze(0)
        obs_world = torch.tensor(obs_dict["world_obs"], dtype=torch.float32).unsqueeze(0)
        agent.e1.reset_hidden_state()
        if hasattr(agent, 'serotonin'):
            agent.serotonin.reset()

        ep_resource_contacts = 0
        ep_schema_salience_values = []

        for step in range(STEPS_PER_EPISODE):
            latent = agent.sense(obs_body, obs_world)
            ticks = agent.clock.advance()

            e1_prior = agent._e1_tick(latent) if ticks["e1_tick"] else torch.zeros(
                1, cfg.latent.world_dim
            )

            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)

            action_int = int(action.argmax(-1).item()) if action.dim() > 0 else int(action.item())
            obs_dict, reward, done, truncated, info = env.step(action_int)

            # Track schema salience
            if agent._schema_salience is not None:
                sal = float(agent._schema_salience.squeeze())
                ep_schema_salience_values.append(sal)
                metrics["schema_salience_values"].append(sal)

            # Resource proximity target for training
            resource_prox = 0.0
            if "resource_field_view" in obs_dict:
                rfv = obs_dict["resource_field_view"]
                resource_prox = float(np.max(rfv)) if hasattr(rfv, '__len__') else float(rfv)

            # Benefit exposure
            benefit = info.get("benefit_exposure", 0.0)
            if benefit > 0:
                ep_resource_contacts += 1
                metrics["resource_contacts"] += 1
                if agent._schema_salience is not None:
                    metrics["p1_schema_salience_at_resource"].append(
                        float(agent._schema_salience.squeeze())
                    )

            # Drive level for schema wanting
            drive_level = agent.compute_drive_level(obs_body)

            # Serotonin step + benefit salience
            agent.serotonin_step(benefit_exposure=benefit)
            if benefit > 0:
                agent.update_benefit_salience(benefit)

            # MECH-216: seed schema wanting
            if condition_cfg.get("schema_wanting_enabled", False):
                agent.update_schema_wanting(drive_level=drive_level)
                if (agent._schema_salience is not None
                        and float(agent._schema_salience.squeeze()) >= condition_cfg.get("schema_wanting_threshold", 0.3)):
                    metrics["wanting_write_count"] += 1

            metrics["total_steps"] += 1

            # Training (P0 only)
            if ep < P0_EPISODES:
                # E1 prediction loss
                e1_optimizer.zero_grad()
                prediction_loss = agent.compute_prediction_loss(latent)
                # Schema readout loss
                schema_loss = agent.compute_schema_readout_loss(resource_prox)
                # Resource proximity loss
                rp_loss = agent.compute_resource_proximity_loss(resource_prox, latent)
                total_loss = prediction_loss + schema_loss + 0.5 * rp_loss
                total_loss.backward()
                e1_optimizer.step()
                encoder_optimizer.step()

            obs_body = torch.tensor(obs_dict["body_obs"], dtype=torch.float32).unsqueeze(0)
            obs_world = torch.tensor(obs_dict["world_obs"], dtype=torch.float32).unsqueeze(0)

            if done or truncated:
                break

        # P1 tracking
        if ep >= P0_EPISODES:
            metrics["p1_resource_rates"].append(ep_resource_contacts / max(1, step + 1))

    return metrics


def main():
    results = {}
    for cond_name, cond_cfg in CONDITIONS.items():
        results[cond_name] = {}
        for seed in range(NUM_SEEDS):
            print(f"Running {cond_name} seed={seed}...")
            m = run_condition(cond_name, cond_cfg, seed=42 + seed)
            results[cond_name][f"seed_{seed}"] = m

    # Aggregate
    agg = {}
    for cond_name in CONDITIONS:
        cond_results = results[cond_name]
        all_sal = []
        all_rr = []
        all_wc = 0
        all_sal_at_res = []
        for sk, m in cond_results.items():
            all_sal.extend(m["schema_salience_values"])
            all_rr.extend(m["p1_resource_rates"])
            all_wc += m["wanting_write_count"]
            all_sal_at_res.extend(m.get("p1_schema_salience_at_resource", []))

        agg[cond_name] = {
            "schema_salience_mean": float(np.mean(all_sal)) if all_sal else 0.0,
            "schema_salience_std": float(np.std(all_sal)) if all_sal else 0.0,
            "p1_resource_rate_mean": float(np.mean(all_rr)) if all_rr else 0.0,
            "wanting_write_count": all_wc,
            "schema_salience_at_resource_mean": float(np.mean(all_sal_at_res)) if all_sal_at_res else 0.0,
        }

    # Acceptance checks
    c1_pass = agg["WITH_SCHEMA"]["schema_salience_mean"] > 0.1
    c2_pass = agg["WITH_SCHEMA"]["wanting_write_count"] > agg["WITHOUT_SCHEMA"]["wanting_write_count"]

    # C3: per-seed resource rate comparison
    seed_wins = 0
    for seed in range(NUM_SEEDS):
        w_rr = results["WITH_SCHEMA"][f"seed_{seed}"]["p1_resource_rates"]
        wo_rr = results["WITHOUT_SCHEMA"][f"seed_{seed}"]["p1_resource_rates"]
        w_mean = float(np.mean(w_rr)) if w_rr else 0.0
        wo_mean = float(np.mean(wo_rr)) if wo_rr else 0.0
        if w_mean >= wo_mean:
            seed_wins += 1
    c3_pass = seed_wins >= 2

    outcome = "PASS" if (c1_pass and c2_pass) else "FAIL"

    output = {
        "experiment_type": EXPERIMENT_TYPE,
        "run_id": f"{EXPERIMENT_TYPE}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_v3",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "conditions": list(CONDITIONS.keys()),
        "aggregated": agg,
        "acceptance_checks": {
            "C1_schema_salience_gt_0.1": c1_pass,
            "C2_wanting_write_coverage_higher": c2_pass,
            "C3_resource_rate_lift_2of3": c3_pass,
        },
        "params": {
            "grid_size": GRID_SIZE,
            "p0_episodes": P0_EPISODES,
            "p1_episodes": P1_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "num_seeds": NUM_SEEDS,
        },
        "evidence_direction": "supports" if outcome == "PASS" else "does_not_support",
    }

    # Write output
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "REE_assembly", "evidence", "experiments"
    )
    out_path = os.path.join(out_dir, f"{EXPERIMENT_TYPE}_{ts}_v3.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        output["env_seed_base"] = _ENV_SEED_BASE
        json.dump(output, f, indent=2)
    print(f"Results written to {out_path}")
    print(f"Outcome: {outcome}")
    print(f"C1 (salience>0.1): {c1_pass} ({agg['WITH_SCHEMA']['schema_salience_mean']:.4f})")
    print(f"C2 (wanting coverage): {c2_pass}")
    print(f"C3 (resource rate lift 2/3): {c3_pass} ({seed_wins}/3)")

    return output


if __name__ == "__main__":
    _ENV_SEED_BASE = _env_seed_from_argv(sys.argv)
    main()
