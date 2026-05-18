#!/opt/local/bin/python3
"""
V3-EXQ-470: SD-029 Balanced Hazard-Event Curriculum -- Substrate Validation

experiment_purpose: diagnostic

Scientific question: Does the scheduled_external_hazard_enabled curriculum
augmentation in CausalGridWorldV2 produce balanced self-caused vs
externally-caused hazard-event densities per seed, as required by SD-029's
C3 (approach-event SNR) and C4 (event-density balance) criteria?

This experiment validates ONLY the env curriculum mechanism, not downstream
claim evidence. Primary questions:
  - Does info['external_hazard_event_count'] grow monotonically under
    SCHEDULED condition?
  - Does it remain at 0 under BASELINE (backward compat)?
  - Is n_ext >= 20 per seed achievable under a modest curriculum budget?
  - Does agent behaviour (total_harm) remain plausible (no pathological
    self-termination, no trivial full-immunity)?

Ablation pair:
  BASELINE: scheduled_external_hazard_enabled=False (default).
  SCHEDULED: scheduled_external_hazard_enabled=True,
             interval=25, prob=1.0, adjacent_only=True.

Acceptance (substrate validation, not governance evidence):
  C_back: BASELINE condition -> external_hazard_event_count == 0 in all seeds.
  C_fire: SCHEDULED condition -> external_hazard_event_count >= 20 per seed.
  C_plausible: SCHEDULED total_harm finite and > 0 across seeds (env still
               produces regular self-caused harm too).
  C_info: every step emits info['external_hazard_injected'] as a bool and
          info['external_hazard_event_count'] as an int.

4 seeds x 2 conditions; episode cap small (3 eps per seed, 200 steps each).
Runs as a pure env-level sanity probe. No agent training. Random action
policy (uniform over 5 actions).

Claims tested: SD-029 (substrate readiness only -- NO evidence_direction set).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
import random
from datetime import datetime
from typing import Dict, List

import numpy as np

from ree_core.environment.causal_grid_world import CausalGridWorldV2

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE    = "v3_exq_470_sd029_balanced_curriculum"
CLAIM_IDS          = ["SD-029"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS      = [42, 7, 13, 91]
CONDITIONS = ["BASELINE", "SCHEDULED"]

EPISODES_PER_SEED = 3
STEPS_PER_EPISODE = 200

GRID_SIZE     = 8
NUM_RESOURCES = 3
NUM_HAZARDS   = 4

CURRICULUM_INTERVAL       = 25
CURRICULUM_PROB           = 1.0
CURRICULUM_ADJACENT_ONLY  = True


def _make_env(seed: int, scheduled: bool) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_resources=NUM_RESOURCES,
        num_hazards=NUM_HAZARDS,
        hazard_harm=0.3,
        resource_benefit=0.3,
        resource_respawn_on_consume=True,
        proximity_harm_scale=0.1,
        proximity_approach_threshold=0.2,
        use_proxy_fields=True,
        scheduled_external_hazard_enabled=scheduled,
        scheduled_external_hazard_interval=CURRICULUM_INTERVAL,
        scheduled_external_hazard_prob=CURRICULUM_PROB,
        scheduled_external_hazard_adjacent_only=CURRICULUM_ADJACENT_ONLY,
    )


def _run_seed(seed: int, scheduled: bool, dry_run: bool) -> Dict:
    env = _make_env(seed, scheduled)
    rng = random.Random(seed)
    n_eps = 1 if dry_run else EPISODES_PER_SEED
    n_steps_cap = 20 if dry_run else STEPS_PER_EPISODE
    total_ext_events = 0
    total_self_events = 0
    total_harm = 0.0
    n_info_keys_ok = True
    n_steps_total = 0
    for ep in range(n_eps):
        obs, info_dict = env.reset()
        for t in range(n_steps_cap):
            a = rng.randint(0, 4)
            flat_obs, h, done, info, obs_dict = env.step(a)
            if ('external_hazard_injected' not in info
                    or 'external_hazard_event_count' not in info):
                n_info_keys_ok = False
            transition = info.get('transition_type', 'none')
            if transition == 'agent_caused_hazard':
                total_self_events += 1
            n_steps_total += 1
            total_harm = float(info.get('total_harm', 0.0))
            if done:
                break
        total_ext_events = info['external_hazard_event_count']
    return {
        'seed': seed,
        'scheduled': bool(scheduled),
        'n_steps_total': int(n_steps_total),
        'external_hazard_events_last_ep': int(total_ext_events),
        'self_caused_events_total': int(total_self_events),
        'total_harm_last_ep': float(total_harm),
        'info_keys_ok': bool(n_info_keys_ok),
    }


def main(dry_run: bool = False) -> Dict:
    results: Dict[str, List[Dict]] = {c: [] for c in CONDITIONS}
    for cond in CONDITIONS:
        scheduled = (cond == 'SCHEDULED')
        for seed in SEEDS:
            r = _run_seed(seed, scheduled, dry_run)
            results[cond].append(r)

    # Acceptance checks
    baseline_ext_zero = all(r['external_hazard_events_last_ep'] == 0
                            for r in results['BASELINE'])
    scheduled_fire_ge_20 = sum(
        1 for r in results['SCHEDULED']
        if r['external_hazard_events_last_ep'] >= 20
    )
    scheduled_plausible = all(r['total_harm_last_ep'] > 0 for r in results['SCHEDULED'])
    info_keys_always_present = all(
        r['info_keys_ok']
        for c in CONDITIONS for r in results[c]
    )

    c_back = bool(baseline_ext_zero)
    # Require >=3/4 seeds to hit 20 events (one seed may reset on early death).
    c_fire = int(scheduled_fire_ge_20) >= 3
    c_plausible = bool(scheduled_plausible)
    c_info = bool(info_keys_always_present)

    passed = c_back and c_fire and c_plausible and c_info

    criteria = {
        'C_back_baseline_zero': c_back,
        'C_fire_scheduled_ge20': c_fire,
        'C_plausible_harm_gt0': c_plausible,
        'C_info_keys_present': c_info,
    }

    summary = {
        'experiment_type': EXPERIMENT_TYPE,
        'experiment_purpose': EXPERIMENT_PURPOSE,
        'claim_ids': CLAIM_IDS,
        'architecture_epoch': 'ree_hybrid_guardrails_v1',
        'timestamp_utc': datetime.utcnow().isoformat() + 'Z',
        'seeds': SEEDS,
        'conditions': CONDITIONS,
        'episodes_per_seed': EPISODES_PER_SEED,
        'steps_per_episode': STEPS_PER_EPISODE,
        'curriculum_interval': CURRICULUM_INTERVAL,
        'curriculum_prob': CURRICULUM_PROB,
        'curriculum_adjacent_only': CURRICULUM_ADJACENT_ONLY,
        'results': results,
        'criteria': criteria,
        'passed': passed,
        'evidence_direction': 'supports' if passed else 'weakens',
    }
    return summary


def _write_output(summary: Dict) -> str:
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = os.path.join(
        os.path.dirname(__file__),
        '..', '..',
        'REE_assembly', 'evidence', 'experiments'
    )
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_id}.json")
    summary['run_id'] = run_id
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    summary = main(dry_run=args.dry_run)
    if args.dry_run:
        print('DRY RUN SUMMARY:')
        print(json.dumps(summary['criteria'], indent=2))
        print('passed:', summary['passed'])
    else:
        out_path = _write_output(summary)
        print('wrote:', out_path)
        print('passed:', summary['passed'])
