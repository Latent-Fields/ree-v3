#!/opt/local/bin/python3
"""
V3-EXQ-047m -- MECH-095 TPJ Agency-Routing CORRECTED Retest on the SD-047 Env

Claims: MECH-095 (tpj.agency_detection_comparator)
Experiment purpose: evidence
Supersedes: V3-EXQ-047l

Why this experiment exists
--------------------------
V3-EXQ-047l (the first SD-047 retest of MECH-095) FAILed 3/5 with
non_degenerate:false, but the /failure-autopsy
(REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-047l_2026-07-11.md)
adjudicated it NON_CONTRIBUTORY -- a measurement/test-design degeneracy, NOT a
substrate_ceiling hit. Dispositive root cause: the probe's contact-vs-no-contact
partition SATURATED. 047l's probe used
    is_contact = (ttype in CONTACT_SET) or (env_events > 0)
and on SD-047's dense per-step dynamics `multi_source_n_env_events` is a per-step
counter (causal_grid_world.py:2292 resets+increments it EVERY step from drift /
transient / weather sources), so `env_events > 0` was true on essentially every
probe step -> the no-contact (negative) class was EMPTY in all 8 cells
(n_contact == n_probe: 126/119/113/105) -> the balanced contact probe never ran
-> contact_recall_world was structurally pinned at 0.0 in BOTH ROUTED and
BASELINE. The MECH-095 comparator was never exercised. Proof by contrast: the
EXQ-047k PASS (same size-12 / 4-hazard grid, is_contact WITHOUT the env_events
fold) had n_contact_min=164 and recall_routed=0.796.

This is the corrected same-question retest (alphabetic suffix, implementation
fix). The scientific question, operationalisation, arms, seeds, and pre-registered
acceptance criteria are UNCHANGED from 047l/047k.

The three autopsy-mandated fixes (047l -> 047m)
-----------------------------------------------
1. KEEP the env-events fold in the TRAINING routing label (`is_world`): it is
   NECESSARY for ROUTED != BASELINE on SD-047 (the transient/drift/weather events
   increment multi_source_n_env_events but do NOT set transition_type, so without
   the fold the routing head would be blind to the SD-047 enrichment and the test
   would be vacuous). UNCHANGED from 047l.
2. REVERT the PROBE partition `is_contact` to 047k's non-folded form
       is_contact = ttype in CONTACT_SET
   (drop the `or env_events > 0`). This restores the partition that gave a
   populated negative class (n_contact_min=164) + recall 0.796 on the same grid.
   The additive fold that is correct in a *training* signal must NOT be mirrored
   into the *evaluation* partition -- on a dense substrate it saturates the eval
   class and makes the probe vacuous.
3. ADD a probe-partition non-degeneracy guard with two thresholds:
   - BLOCKING floor PROBE_NEG_FLOOR=5: if the no-contact (negative) class is below
     this in ANY cell the balanced probe cannot run (the exact 047l failure ->
     spurious 0.0), so self-route non_contributory (test-bed-not-ready) via
     non_degenerate:false rather than score a `weakens`. Floor set to 5 (not the
     autopsy's estimated ~20) because on this 12-grid `hazard_approach` (proximity)
     saturates ~91% of steps on BOTH the thin and SD-047 envs, so 047k's own
     PASSing negative class was only ~6-28 samples per seed -- a floor above that
     would veto the very regime this retest reproduces.
   - ADVISORY balanced target PROBE_NEG_BALANCED_TARGET=20: between the floor and
     this the probe ran but is imbalanced/underpowered; surfaced as a non-blocking
     WARN so the recall is read with caution, but it does NOT set non_degenerate or
     non_contributory (047k PASSed in exactly this regime).

Secondary (claim-orthogonal) signal carried forward from the 047l autopsy
-------------------------------------------------------------------------
047l also showed a genuinely non-degenerate side effect: routing collapsed the
self-vs-world action dissociation (BASELINE +0.10..+0.23 -> ROUTED ~0), i.e.
z_world became about as action-predictive as z_self under routing. Real but
claim-orthogonal (MECH-095 predicts world-vs-self CAUSATION encoding, not action
predictability). 047m reports action_dissoc for BOTH arms so the collapse is
visible alongside the recall result. Do not over-read it.

Operationalisation (reused verbatim from EXQ-047k / 047l)
---------------------------------------------------------
Two arms, ROUTED (ON) vs BASELINE (ABLATED), 4 seeds (42, 7, 123, 99):
  ROUTED   -- split latent + a CE routing head trained (BCE) to predict, from
              z_world, whether the previous transition was world-caused. This is
              the agency comparator: it pushes z_world to encode world- vs
              self-causation. The training label folds env_events>0 additively.
  BASELINE -- split latent, no routing head (ablated comparator).
Probes (frozen agent, uniform-random policy):
  contact_recall_world  -- balanced linear probe on z_world: contact vs no-contact.
                           Partition uses 047k's non-folded CONTACT_SET.
  action prediction     -- linear probe accuracy from z_self and from z_world.

Pre-registered acceptance criteria (identical to EXQ-047k / 047l)
-----------------------------------------------------------------
  C1: contact_recall_world_routed        > 0.55
  C2: recall improvement (routed-baseline) > 0.04
  C3: action_dissoc_mean (self - world)  > -0.05  (routing must not hurt action pred)
  C4: n_contact_probe (min over arms)    >= 20
  C5: no fatal errors
PASS = C1..C5 all met -> evidence_direction supports (re-gates MECH-095).
FAIL (any) with a well-posed probe -> weakens/mixed, route /failure-autopsy.
Probe-partition degenerate (n_no_contact < 5 in any cell) -> non_contributory
(test-bed-not-ready), non_degenerate:false, route /failure-autopsy. n_no_contact
in [5, 20) -> probe runs with a non-blocking imbalance WARN.

Re-derive brake note
--------------------
The 047l autopsy registered 1 non_contributory reading for MECH-095 (brake
threshold 2). If 047m ALSO returns non_contributory, that implicates the
operationalisation itself (not one probe bug) -- governance should then route to
substrate/test-bed work, NOT a third 047 letter.

SD-047 is enabled at multi_source_intensity_scale=1.0 (ARM_2 default -- the
calibration point V3-EXQ-509 validated, ratio 2.03 in target band); the two arms
are the comparator ON/ABLATED, not an intensity sweep.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_047m_mech095_agency_routing_sd047.py --dry-run
  /opt/local/bin/python3 experiments/v3_exq_047m_mech095_agency_routing_sd047.py
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from _metrics import check_degeneracy


EXPERIMENT_TYPE = "v3_exq_047m_mech095_agency_routing_sd047"
CLAIM_IDS = ["MECH-095"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES = "V3-EXQ-047l"

# 047k transition-type label sets (unchanged).
WORLD_CAUSED = frozenset({"env_caused_hazard", "hazard_approach"})
CONTACT_SET = frozenset({"hazard_approach", "agent_caused_hazard", "env_caused_hazard"})

# Probe-partition non-degeneracy guard (047m fix for the 047l empty-negative-class
# failure). BLOCKING floor: every cell must have at least this many no-contact
# (negative) probe samples, else the balanced contact probe cannot run at all (it
# returns a spurious 0.0 -- the exact 047l failure mode) and the run self-routes
# non_contributory. Set to 5 to match the regime the EXQ-047k PASS actually ran in:
# on this 12-grid `hazard_approach` (proximity) saturates ~91% of steps on BOTH the
# thin and SD-047 envs, so 047k's own PASSing negative class was only ~6-28 samples
# per seed (min ~6). A floor above that would veto the very regime we reproduce.
PROBE_NEG_FLOOR = 5

# ADVISORY (non-blocking) balanced-probe target. Between PROBE_NEG_FLOOR and this,
# the balanced probe runs but is imbalanced / underpowered (047k ran here). We
# surface it as a WARN in the manifest so the result is read with caution, but it
# does NOT set non_degenerate or non_contributory -- the probe genuinely ran.
PROBE_NEG_BALANCED_TARGET = 20

# SD-047 source defaults at intensity_scale=1.0 (matches V3-EXQ-509 / 510 / 047l).
WEATHER_SUPER_CELLS = 4
WEATHER_ALPHA_AR1 = 0.95
WEATHER_SIGMA = 0.10
TRANSIENT_P_APPEAR = 5e-3
TRANSIENT_P_DISAPPEAR = 0.10
N_DRIFT_SOURCES = 2
DRIFT_POLICY = "random_walk"


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _fit_balanced_probe(
    X_pos: torch.Tensor,
    X_neg: torch.Tensor,
    n_steps: int = 400,
    lr: float = 1e-2,
) -> float:
    n = min(len(X_pos), len(X_neg))
    if n < 5:
        return 0.0
    idx_pos = torch.randperm(len(X_pos))[:n]
    idx_neg = torch.randperm(len(X_neg))[:n]
    X = torch.cat([X_pos[idx_pos], X_neg[idx_neg]], dim=0).float()
    y = torch.cat([
        torch.ones(n, dtype=torch.long),
        torch.zeros(n, dtype=torch.long),
    ])
    probe = nn.Linear(X.shape[1], 2)
    opt = optim.Adam(probe.parameters(), lr=lr)
    for _ in range(n_steps):
        logits = probe(X)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = probe(X_pos[idx_pos]).argmax(dim=1)
        recall = float((preds == 1).float().mean().item())
    return recall


def _fit_action_probe(
    X: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
    n_steps: int = 400,
    lr: float = 1e-2,
) -> float:
    X = X.detach().float()
    y = y.detach().long()
    probe = nn.Linear(X.shape[1], n_classes)
    opt = optim.Adam(probe.parameters(), lr=lr)
    for _ in range(n_steps):
        logits = probe(X)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = probe(X).argmax(dim=1)
        acc = float((preds == y).float().mean().item())
    return acc


def _run_single(
    seed: int,
    use_routing: bool,
    warmup_episodes: int,
    probe_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    alpha_world: float,
    alpha_self: float,
    harm_scale: float,
    proximity_harm_scale: float,
    lambda_route: float,
    intensity_scale: float,
) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    cond_label = "ROUTED" if use_routing else "BASELINE"

    # SD-047 enriched CausalGridWorldV2: 047k's agent-side grid + all three
    # multi-source dynamics sources enabled at intensity_scale (ARM_2 default).
    env = CausalGridWorldV2(
        seed=seed,
        size=12,
        num_hazards=4,
        num_resources=5,
        hazard_harm=harm_scale,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=proximity_harm_scale,
        proximity_benefit_scale=proximity_harm_scale * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        multi_source_dynamics_enabled=True,
        multi_source_intensity_scale=intensity_scale,
        weather_field_enabled=True,
        weather_super_cells=WEATHER_SUPER_CELLS,
        weather_alpha_ar1=WEATHER_ALPHA_AR1,
        weather_sigma=WEATHER_SIGMA,
        transient_events_enabled=True,
        transient_p_appear=TRANSIENT_P_APPEAR,
        transient_p_disappear=TRANSIENT_P_DISAPPEAR,
        background_drift_enabled=True,
        n_drift_sources=N_DRIFT_SOURCES,
        drift_policy=DRIFT_POLICY,
    )

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=0,
    )
    config.latent.unified_latent_mode = False

    agent = REEAgent(config)

    routing_head: nn.Module = nn.Sequential(
        nn.Linear(world_dim, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF = 2000

    standard_params = (
        list(agent.parameters()) + list(routing_head.parameters())
        if use_routing else list(agent.parameters())
    )
    standard_params = [
        p for p in standard_params
        if not any(
            p is ph
            for ph in agent.e3.harm_eval_head.parameters()
        )
    ]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    optimizer = optim.Adam(standard_params, lr=lr)
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)

    counts: Dict[str, int] = {
        "hazard_approach": 0, "env_caused_hazard": 0,
        "agent_caused_hazard": 0, "none": 0,
    }
    env_event_ticks = 0
    total_routing_loss = 0.0
    routing_steps = 0

    agent.train()
    routing_head.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        prev_ttype: str = None
        prev_env_events: int = 0

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world

            routing_loss = torch.zeros(1)
            if use_routing and prev_ttype is not None:
                # TRAINING routing label folds SD-047 env events additively (strict
                # superset of 047k's transition_type-only label). KEEP unchanged --
                # this is what makes ROUTED differ from BASELINE on SD-047. The fold
                # belongs HERE (training signal) and NOT in the eval probe partition.
                is_world = (
                    1.0 if (prev_ttype in WORLD_CAUSED or prev_env_events > 0)
                    else 0.0
                )
                label = torch.tensor([[is_world]])
                routing_loss = lambda_route * F.binary_cross_entropy_with_logits(
                    routing_head(z_world_curr),
                    label,
                )
                total_routing_loss += routing_loss.item()
                routing_steps += 1

            z_world_track = z_world_curr.detach()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)
            prev_ttype = info.get("transition_type", "none")
            prev_env_events = int(info.get("multi_source_n_env_events", 0))
            ttype = prev_ttype
            if ttype in counts:
                counts[ttype] += 1
            if prev_env_events > 0:
                env_event_ticks += 1

            is_harm = float(harm_signal) < 0
            if is_harm:
                harm_buf_pos.append(z_world_track)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_world_track)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss + routing_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
                zw_pos = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zw_neg = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)
                zw_b = torch.cat([zw_pos, zw_neg], dim=0)
                target = torch.cat([
                    torch.ones(k_pos, 1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)
                pred_harm = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred_harm, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5,
                    )
                    harm_eval_optimizer.step()

            if done:
                break

        if (ep + 1) % 50 == 0 or ep == warmup_episodes - 1:
            mean_route = (
                total_routing_loss / max(1, routing_steps) if use_routing else 0.0
            )
            print(
                f"  [train] seed={seed} cond={cond_label}"
                f" ep {ep+1}/{warmup_episodes}"
                f" approach={counts['hazard_approach']}"
                f" contact={counts['env_caused_hazard']+counts['agent_caused_hazard']}"
                f" env_ev_ticks={env_event_ticks}"
                f" route_loss={mean_route:.4f}",
                flush=True,
            )

    agent.eval()
    routing_head.eval()

    probe_self_contact: List[torch.Tensor] = []
    probe_self_no_contact: List[torch.Tensor] = []
    probe_world_contact: List[torch.Tensor] = []
    probe_world_no_contact: List[torch.Tensor] = []
    probe_self_all: List[torch.Tensor] = []
    probe_world_all: List[torch.Tensor] = []
    probe_actions: List[int] = []
    n_fatal = 0

    for _ in range(probe_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, _, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            try:
                with torch.no_grad():
                    obs_body_next  = obs_dict["body_state"]
                    obs_world_next = obs_dict["world_state"]
                    latent_next = agent.sense(obs_body_next, obs_world_next)
                    agent.clock.advance()

                    zs = latent_next.z_self.detach()
                    zw = latent_next.z_world.detach()
                    # 047m FIX: probe partition REVERTS to 047k's non-folded
                    # CONTACT_SET (drop the `or env_events>0` fold that 047l over-
                    # applied here). On SD-047's dense per-step dynamics the fold
                    # was true on every step -> empty no-contact class -> vacuous
                    # probe. The env-events fold stays in the TRAINING label only.
                    is_contact = ttype in CONTACT_SET

                    probe_self_all.append(zs)
                    probe_world_all.append(zw)
                    probe_actions.append(action_idx)

                    if is_contact:
                        probe_self_contact.append(zs)
                        probe_world_contact.append(zw)
                    else:
                        probe_self_no_contact.append(zs)
                        probe_world_no_contact.append(zw)

                    _, _, done2, _, obs_dict = env.step(
                        _action_to_onehot(
                            random.randint(0, env.action_dim - 1),
                            env.action_dim, agent.device,
                        )
                    )
                    if done2:
                        break
            except Exception:
                n_fatal += 1

            if done:
                break

    action_acc_self  = 0.0
    action_acc_world = 0.0
    contact_recall_self  = 0.0
    contact_recall_world = 0.0
    n_contact_probe = len(probe_self_contact)
    n_no_contact_probe = len(probe_self_no_contact)

    if len(probe_self_all) >= 20:
        X_self_all  = torch.cat(probe_self_all,  dim=0).float()
        X_world_all = torch.cat(probe_world_all, dim=0).float()
        y_action    = torch.tensor(probe_actions, dtype=torch.long)

        action_acc_self  = _fit_action_probe(X_self_all,  y_action, env.action_dim)
        action_acc_world = _fit_action_probe(X_world_all, y_action, env.action_dim)

    if n_contact_probe >= 5 and n_no_contact_probe >= 5:
        X_self_pos  = torch.cat(probe_self_contact,     dim=0).float()
        X_self_neg  = torch.cat(probe_self_no_contact,  dim=0).float()
        X_world_pos = torch.cat(probe_world_contact,    dim=0).float()
        X_world_neg = torch.cat(probe_world_no_contact, dim=0).float()

        contact_recall_self  = _fit_balanced_probe(X_self_pos,  X_self_neg)
        contact_recall_world = _fit_balanced_probe(X_world_pos, X_world_neg)

    action_dissociation = action_acc_self - action_acc_world
    mean_route_loss = total_routing_loss / max(1, routing_steps) if use_routing else 0.0

    print(
        f"  [probe] seed={seed} cond={cond_label}"
        f" action: self={action_acc_self:.3f} world={action_acc_world:.3f}"
        f" dissoc={action_dissociation:+.3f}"
        f"  contact_recall: world={contact_recall_world:.3f}"
        f" self={contact_recall_self:.3f}"
        f" n_contact={n_contact_probe} n_no_contact={n_no_contact_probe}"
        f" route_loss={mean_route_loss:.4f}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "use_routing": use_routing,
        "action_acc_self":        float(action_acc_self),
        "action_acc_world":       float(action_acc_world),
        "action_dissociation":    float(action_dissociation),
        "contact_recall_self":    float(contact_recall_self),
        "contact_recall_world":   float(contact_recall_world),
        "n_contact_probe":        int(n_contact_probe),
        "n_no_contact_probe":     int(n_no_contact_probe),
        "n_probe_steps":          int(len(probe_self_all)),
        "mean_routing_loss":      float(mean_route_loss),
        "train_approach_events":  int(counts["hazard_approach"]),
        "train_contact_events":   int(
            counts["env_caused_hazard"] + counts["agent_caused_hazard"]
        ),
        "train_env_event_ticks":  int(env_event_ticks),
        "n_fatal": int(n_fatal),
    }


def run(
    seeds: Tuple = (42, 7, 123, 99),
    warmup_episodes: int = 400,
    probe_episodes: int = 20,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_harm_scale: float = 0.05,
    lambda_route: float = 0.1,
    intensity_scale: float = 1.0,
    **kwargs,
) -> dict:
    """ROUTED (ON) vs BASELINE (ABLATED), 4 seeds, on the SD-047 enriched env."""
    results_routed:   List[Dict] = []
    results_baseline: List[Dict] = []

    for seed in seeds:
        for use_routing in [True, False]:
            label = "ROUTED" if use_routing else "BASELINE"
            # Runner progress boundary: resets episodes_in_run for this run.
            print(f"Seed {seed} Condition {label}", flush=True)
            print(
                f"\n[V3-EXQ-047m] {label} seed={seed}"
                f" warmup={warmup_episodes} lambda_route={lambda_route}"
                f" alpha_world={alpha_world} intensity_scale={intensity_scale}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                use_routing=use_routing,
                warmup_episodes=warmup_episodes,
                probe_episodes=probe_episodes,
                steps_per_episode=steps_per_episode,
                self_dim=self_dim,
                world_dim=world_dim,
                lr=lr,
                alpha_world=alpha_world,
                alpha_self=alpha_self,
                harm_scale=harm_scale,
                proximity_harm_scale=proximity_harm_scale,
                lambda_route=lambda_route,
                intensity_scale=intensity_scale,
            )
            if use_routing:
                results_routed.append(r)
            else:
                results_baseline.append(r)
            # Runner progress verdict: per-run completion marker (adequacy of this
            # seed x condition run), increments runs_done. NOT the scientific
            # verdict -- that is the aggregate PASS/FAIL below.
            run_ok = (r["n_fatal"] == 0 and r["n_contact_probe"] >= 20)
            print(f"verdict: {'PASS' if run_ok else 'FAIL'}", flush=True)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    def _std(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        mean = sum(vals) / max(1, len(vals))
        variance = sum((v - mean) ** 2 for v in vals) / max(1, len(vals))
        return float(variance ** 0.5)

    contact_recall_routed   = _avg(results_routed,   "contact_recall_world")
    contact_recall_baseline = _avg(results_baseline, "contact_recall_world")
    action_dissoc_routed    = _avg(results_routed,   "action_dissociation")
    action_dissoc_std       = _std(results_routed,   "action_dissociation")
    # Secondary (claim-orthogonal) signal: BASELINE action dissociation, so the
    # routing-induced collapse (BASELINE +ve -> ROUTED ~0) is visible in the manifest.
    action_dissoc_baseline     = _avg(results_baseline, "action_dissociation")
    action_dissoc_baseline_std = _std(results_baseline, "action_dissociation")
    mean_route_loss         = _avg(results_routed,   "mean_routing_loss")
    n_contact_min = min(
        r["n_contact_probe"] for r in results_routed + results_baseline
    )
    n_no_contact_min = min(
        r["n_no_contact_probe"] for r in results_routed + results_baseline
    )

    c1_pass = contact_recall_routed > 0.55
    c2_pass = (contact_recall_routed - contact_recall_baseline) > 0.04
    # Relaxed from >0 to >-0.05 (047k): routing must not significantly hurt action pred.
    c3_pass = action_dissoc_routed > -0.05
    c4_pass = n_contact_min >= 20
    c5_pass = all(r["n_fatal"] == 0 for r in results_routed + results_baseline)

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])
    status = "PASS" if all_pass else "FAIL"

    routing_active = mean_route_loss > 0.001

    # 047m probe-partition non-degeneracy guard: the balanced contact probe is only
    # valid if BOTH classes are populated in every cell. If the no-contact negative
    # class is below PROBE_NEG_FLOOR anywhere, the balanced probe cannot run and
    # contact_recall_world is a spurious 0.0 (the exact 047l failure) -> the honest
    # reading is non_contributory (test-bed-not-ready), NOT a weakens. Above the
    # floor but below PROBE_NEG_BALANCED_TARGET, the probe ran but is imbalanced --
    # a non-blocking WARN only (047k PASSed in exactly this regime).
    probe_partition_ok = n_no_contact_min >= PROBE_NEG_FLOOR
    probe_partition_balanced = n_no_contact_min >= PROBE_NEG_BALANCED_TARGET

    if not probe_partition_ok:
        decision = "probe_partition_degenerate_requeue"
    elif all_pass and routing_active:
        decision = "ceiling_lifted_regate_mech095"
    elif c1_pass and c2_pass:
        decision = "partial_discrimination"
    else:
        decision = "ceiling_confirmed_route_autopsy"

    print(f"\n[V3-EXQ-047m] Final results ({len(seeds)} seeds, SD-047 enriched):", flush=True)
    print(
        f"  contact_recall: routed={contact_recall_routed:.3f}"
        f"  baseline={contact_recall_baseline:.3f}"
        f"  improvement={contact_recall_routed - contact_recall_baseline:+.3f}",
        flush=True,
    )
    print(
        f"  action_dissoc: routed={action_dissoc_routed:+.3f} (std={action_dissoc_std:.3f})"
        f"  baseline={action_dissoc_baseline:+.3f} (std={action_dissoc_baseline_std:.3f})"
        f"  mean_route_loss={mean_route_loss:.4f}",
        flush=True,
    )
    print(
        f"  probe partition: n_contact_min={n_contact_min}"
        f" n_no_contact_min={n_no_contact_min}"
        f" (floor={PROBE_NEG_FLOOR} {'OK' if probe_partition_ok else 'DEGENERATE'};"
        f" balanced_target={PROBE_NEG_BALANCED_TARGET}"
        f" {'MET' if probe_partition_balanced else 'IMBALANCED-warn'})",
        flush=True,
    )
    print(
        f"  decision={decision}  status={status} ({criteria_met}/5)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not probe_partition_ok:
        failure_notes.append(
            f"PROBE PARTITION DEGENERATE: n_no_contact_min={n_no_contact_min}"
            f" < floor {PROBE_NEG_FLOOR} -- no-contact negative class too small for the"
            f" balanced contact probe to run (spurious recall). non_contributory."
        )
    elif not probe_partition_balanced:
        failure_notes.append(
            f"WARN (non-blocking): n_no_contact_min={n_no_contact_min} < balanced-probe"
            f" target {PROBE_NEG_BALANCED_TARGET} -- probe ran but is imbalanced /"
            f" underpowered on this grid (hazard_approach saturates ~91% of steps;"
            f" 047k PASSed at ~6). Result stands; read the recall with caution."
        )
    if not routing_active:
        failure_notes.append(
            f"WARN: mean_route_loss={mean_route_loss:.4f} <= 0.001"
            " (CE routing head not active)"
        )
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: contact_recall_world_routed={contact_recall_routed:.3f} <= 0.55"
        )
    if not c2_pass:
        gap = contact_recall_routed - contact_recall_baseline
        failure_notes.append(
            f"C2 FAIL: recall improvement={gap:+.3f} <= 0.04"
            f" (routed={contact_recall_routed:.3f}"
            f" vs baseline={contact_recall_baseline:.3f})"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: action_dissoc_mean={action_dissoc_routed:+.3f} <= -0.05"
            f" (std={action_dissoc_std:.3f}) -- routing hurts action prediction"
        )
    if not c4_pass:
        failure_notes.append(f"C4 FAIL: n_contact_min={n_contact_min} < 20")
    if not c5_pass:
        failure_notes.append("C5 FAIL: fatal errors occurred")

    for note in failure_notes:
        print(f"  {note}", flush=True)

    per_routed_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" contact_recall_world={r['contact_recall_world']:.3f}"
        f" contact_recall_self={r['contact_recall_self']:.3f}"
        f" action_dissoc={r['action_dissociation']:+.3f}"
        f" route_loss={r['mean_routing_loss']:.4f}"
        f" n_contact={r['n_contact_probe']}"
        f" n_no_contact={r['n_no_contact_probe']}"
        f" env_ev_ticks={r['train_env_event_ticks']}"
        for r in results_routed
    )
    per_baseline_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" contact_recall_world={r['contact_recall_world']:.3f}"
        f" contact_recall_self={r['contact_recall_self']:.3f}"
        f" action_dissoc={r['action_dissociation']:+.3f}"
        f" n_contact={r['n_contact_probe']}"
        f" n_no_contact={r['n_no_contact_probe']}"
        f" env_ev_ticks={r['train_env_event_ticks']}"
        for r in results_baseline
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-047m -- MECH-095 TPJ Agency-Routing CORRECTED Retest on SD-047\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-095\n"
        f"**Supersedes:** {SUPERSEDES}\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Substrate:** SD-047 multi-source dynamics (intensity_scale={intensity_scale})\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**lambda_route:** {lambda_route}\n"
        f"**Warmup:** {warmup_episodes} eps  **Probe:** {probe_episodes} eps\n"
        f"**mean_routing_loss (ROUTED):** {mean_route_loss:.4f}"
        f" ({'ACTIVE' if routing_active else 'INACTIVE'})\n"
        f"**Probe partition:** n_contact_min={n_contact_min}"
        f" n_no_contact_min={n_no_contact_min} (floor={PROBE_NEG_FLOOR}"
        f" {'OK' if probe_partition_ok else 'DEGENERATE'};"
        f" balanced_target={PROBE_NEG_BALANCED_TARGET}"
        f" {'MET' if probe_partition_balanced else 'IMBALANCED-warn'})\n"
        f"**047m fix vs 047l:** training is_world label KEEPS the env_events fold;\n"
        f"probe is_contact REVERTS to 047k's non-folded CONTACT_SET; added a probe-\n"
        f"partition non-degeneracy guard (blocking floor n_no_contact >= {PROBE_NEG_FLOOR}/arm\n"
        f"= non_contributory; advisory balanced target {PROBE_NEG_BALANCED_TARGET} = WARN only).\n\n"
        f"## Pre-Registered Thresholds (identical to EXQ-047k / 047l)\n\n"
        f"C1: contact_recall_world_routed > 0.55\n"
        f"C2: recall improvement (routed - baseline) > 0.04\n"
        f"C3: action_dissoc_mean > -0.05\n"
        f"C4: n_contact_probe >= 20\n"
        f"C5: no fatal errors\n\n"
        f"## Results\n\n"
        f"| Condition | contact_recall | action_dissoc (mean +/- std) | route_loss |\n"
        f"|-----------|----------------|------------------------------|------------|\n"
        f"| ROUTED    | {contact_recall_routed:.3f}          |"
        f" {action_dissoc_routed:+.3f} +/- {action_dissoc_std:.3f}   | {mean_route_loss:.4f}     |\n"
        f"| BASELINE  | {contact_recall_baseline:.3f}          |"
        f" {action_dissoc_baseline:+.3f} +/- {action_dissoc_baseline_std:.3f}   | --         |\n\n"
        f"**Recall improvement: {contact_recall_routed - contact_recall_baseline:+.3f}**\n\n"
        f"**Secondary (claim-orthogonal) action-dissociation collapse:"
        f" BASELINE {action_dissoc_baseline:+.3f} -> ROUTED {action_dissoc_routed:+.3f}**\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: contact_recall_world > 0.55 | {'PASS' if c1_pass else 'FAIL'}"
        f" | {contact_recall_routed:.3f} |\n"
        f"| C2: improvement > 0.04 | {'PASS' if c2_pass else 'FAIL'}"
        f" | {contact_recall_routed - contact_recall_baseline:+.3f} |\n"
        f"| C3: action_dissoc > -0.05 | {'PASS' if c3_pass else 'FAIL'}"
        f" | {action_dissoc_routed:+.3f} (std={action_dissoc_std:.3f}) |\n"
        f"| C4: n_contact >= 20 | {'PASS' if c4_pass else 'FAIL'}"
        f" | {n_contact_min} |\n"
        f"| C5: no fatal errors | {'PASS' if c5_pass else 'FAIL'} | -- |\n\n"
        f"Criteria met: {criteria_met}/5 -> **{status}**\n\n"
        f"## Per-Seed\n\n"
        f"ROUTED:\n{per_routed_rows}\n\n"
        f"BASELINE:\n{per_baseline_rows}\n"
        f"{failure_section}\n"
    )

    # Non-degeneracy self-report (whole-run). Two independent degeneracy signatures:
    #   (a) probe-partition degenerate: the no-contact negative class is undersized
    #       in some cell (the 047l failure) -> contact_recall_world is a spurious
    #       0.0, the comparator was never exercised. non_contributory (test-bed-not-
    #       ready). This is the dominant, 047m-specific guard.
    #   (b) all-cells-at-chance (WOO_SPELKE analog): even with populated partitions,
    #       every cell (ROUTED and BASELINE) at the balanced-probe chance level means
    #       z_world encodes no contact anywhere -> ceiling read, non_contributory.
    # Either sets non_degenerate=false (indexer scoring-excluded="degenerate"),
    # routes /failure-autopsy, and does NOT ding MECH-095 confidence as a `weakens`.
    all_recall_cells = [
        r["contact_recall_world"] for r in results_routed + results_baseline
    ]
    degeneracy = check_degeneracy({
        "contact_recall_world": {"values": all_recall_cells, "floor": 0.55},
    })
    if not probe_partition_ok:
        partition_reason = (
            f"probe_negative_class undersized: n_no_contact_min={n_no_contact_min}"
            f" < {PROBE_NEG_FLOOR} (empty/near-empty no-contact partition -> balanced"
            f" contact probe not validly exercised; contact_recall_world spurious)"
        )
        degeneracy["non_degenerate"] = False
        degeneracy["degeneracy_reason"] = (
            partition_reason + " | " + degeneracy["degeneracy_reason"]
            if degeneracy.get("degeneracy_reason") else partition_reason
        )
        dm = dict(degeneracy.get("degenerate_metrics") or {})
        dm["probe_negative_class"] = partition_reason
        degeneracy["degenerate_metrics"] = dm

    metrics = {
        "contact_recall_routed":   float(contact_recall_routed),
        "contact_recall_baseline": float(contact_recall_baseline),
        "recall_improvement":      float(contact_recall_routed - contact_recall_baseline),
        "action_dissoc_routed":    float(action_dissoc_routed),
        "action_dissoc_std":       float(action_dissoc_std),
        "action_dissoc_baseline":  float(action_dissoc_baseline),
        "action_dissoc_baseline_std": float(action_dissoc_baseline_std),
        "mean_routing_loss":       float(mean_route_loss),
        "n_contact_min":           float(n_contact_min),
        "n_no_contact_min":        float(n_no_contact_min),
        "probe_neg_floor":         float(PROBE_NEG_FLOOR),
        "probe_neg_balanced_target": float(PROBE_NEG_BALANCED_TARGET),
        "probe_partition_balanced": 1.0 if probe_partition_balanced else 0.0,
        "lambda_route":            float(lambda_route),
        "intensity_scale":         float(intensity_scale),
        "n_seeds":                 float(len(seeds)),
        "alpha_world":             float(alpha_world),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "crit5_pass": 1.0 if c5_pass else 0.0,
        "criteria_met": float(criteria_met),
        "probe_partition_ok": 1.0 if probe_partition_ok else 0.0,
    }

    # evidence_direction: honest self-route. A degenerate probe partition is
    # non_contributory (test-bed-not-ready), NOT a weakens. Non-degenerate outcomes
    # follow the 047k/047l pass/criteria logic.
    if not probe_partition_ok:
        evidence_direction = "non_contributory"
    elif all_pass:
        evidence_direction = "supports"
    elif criteria_met >= 3:
        evidence_direction = "mixed"
    else:
        evidence_direction = "weakens"

    result_dict = {
        "status": status,
        "outcome": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": evidence_direction,
        "experiment_type": EXPERIMENT_TYPE,
        "supersedes": SUPERSEDES,
        "decision": decision,
        "fatal_error_count": sum(
            r["n_fatal"] for r in results_routed + results_baseline
        ),
        "per_seed_routed": results_routed,
        "per_seed_baseline": results_baseline,
    }
    # Merge non_degenerate / degeneracy_reason / degenerate_metrics at manifest root.
    result_dict.update(degeneracy)
    if not degeneracy["non_degenerate"]:
        print(
            f"  DEGENERATE: {degeneracy['degeneracy_reason']}"
            " -- run is scoring-excluded (non_contributory); route /failure-autopsy",
            flush=True,
        )
    return result_dict


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",           type=int,   nargs="+", default=[42, 7, 123, 99])
    parser.add_argument("--warmup",          type=int,   default=400)
    parser.add_argument("--probe-eps",       type=int,   default=20)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    parser.add_argument("--lambda-route",    type=float, default=0.1)
    parser.add_argument("--intensity-scale", type=float, default=1.0)
    parser.add_argument("--dry-run",         action="store_true",
                        help="Short smoke run (2 seeds, tiny episodes); relocates manifest.")
    args = parser.parse_args()

    if args.dry_run:
        seeds = tuple(args.seeds[:2]) if len(args.seeds) >= 2 else tuple(args.seeds)
        warmup = 3
        probe_eps = 3
        steps = 12
    else:
        seeds = tuple(args.seeds)
        warmup = args.warmup
        probe_eps = args.probe_eps
        steps = args.steps

    result = run(
        seeds=seeds,
        warmup_episodes=warmup,
        probe_episodes=probe_eps,
        steps_per_episode=steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        harm_scale=args.harm_scale,
        proximity_harm_scale=args.proximity_scale,
        lambda_route=args.lambda_route,
        intensity_scale=args.intensity_scale,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)

    # --- runner-conformance sentinel ---
    _outcome_raw = str(result.get("status", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
