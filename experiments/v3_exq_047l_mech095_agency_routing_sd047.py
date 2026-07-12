#!/opt/local/bin/python3
"""
V3-EXQ-047l -- MECH-095 TPJ Agency-Routing Retest on the SD-047 Multi-Source Env

Claims: MECH-095 (tpj.agency_detection_comparator)
Experiment purpose: evidence

Why this experiment exists
--------------------------
MECH-095 was diagnosed substrate_ceiling on 2026-05-02: CausalGridWorldV2 is
"a thin world", and 7 successor attempts (EXQ-089, 047i, 047j, 098b x2, 121 x2)
all came back weakens/mixed after the lone EXQ-047k PASS. SD-047 (multi-source
environmental dynamics: AR(1) weather field + Poisson transient events +
background drift -- the agent-independent causal background dense enough for
honest substrate-level testing) landed 2026-05-03 and is exactly resolution
path (a) named in the MECH-095 evidence_quality_note. The substrate-ceiling
audit now classifies MECH-095 ceiling_may_have_lifted (pending_retest_after_
substrate); after the 2026-07-11 GOV-CEIL-1 flooring MECH-095 sits
candidate/substrate_ceiling. This is the owed positive-discrimination retest on
the enriched substrate.

Relationship to prior SD-047 x MECH-095 work
--------------------------------------------
V3-EXQ-510 already ran a MECH-095 comparator on the SD-047 live env, but with a
DIFFERENT operationalisation -- the EXQ-506 E2_harm_s counterfactual-gap
comparator (agent_caused vs env_caused counterfactual-forward gap ratios). It
flat-failed WOO_SPELKE (C1-C3 FAIL across all four intensity arms, 2026-05-04).
This experiment instead reuses the EXQ-047k TPJ-routing operationalisation -- the
one operationalisation that ever produced a MECH-095 PASS (047k, on the thin
env) -- and asks whether it discriminates on the enriched substrate. A PASS here
is the positive discrimination on a richer substrate that would re-gate MECH-095
promotion out of candidate; a FAIL routes to /failure-autopsy.

Operationalisation (reused verbatim from EXQ-047k)
-------------------------------------------------
Two arms, ROUTED (ON) vs BASELINE (ABLATED), 4 seeds (42, 7, 123, 99):
  ROUTED   -- split latent + a CE routing head trained (BCE) to predict, from
              z_world, whether the previous transition was world-caused. This is
              the agency comparator: it pushes z_world to encode world- vs
              self-causation.
  BASELINE -- split latent, no routing head (ablated comparator).
Probes (frozen agent, uniform-random policy):
  contact_recall_world  -- balanced linear probe on z_world: contact vs no-contact.
  action prediction     -- linear probe accuracy from z_self and from z_world.

Pre-registered acceptance criteria (identical to EXQ-047k)
----------------------------------------------------------
  C1: contact_recall_world_routed        > 0.55
  C2: recall improvement (routed-baseline) > 0.04
  C3: action_dissoc_mean (self - world)  > -0.05  (routing must not hurt action pred)
  C4: n_contact_probe (min over arms)    >= 20
  C5: no fatal errors
PASS = C1..C5 all met -> evidence_direction supports (re-gates MECH-095).
FAIL (any) -> route /failure-autopsy.

The ONE principled adaptation for the enriched substrate (additive, removes
nothing from the 047k labels)
--------------------------------------------------------------------------
On SD-047 the transient/drift/weather events increment info["multi_source_n_env_
events"] but do NOT set transition_type (only direct agent-hazard contact sets
transition_type -- verified in causal_grid_world.py step()). So the 047k
transition_type-only labels would be BLIND to the SD-047 enrichment and ON would
be indistinguishable from ABLATED (vacuous test). The world-caused routing label
and the contact definition therefore additively fold multi_source_n_env_events>0:
  world-caused tick  := prev_ttype in WORLD_CAUSED  OR  prev_env_events > 0
  contact tick       := ttype in CONTACT_SET        OR  env_events > 0
This is a STRICT SUPERSET of the 047k labels (env_caused_hazard already implies
env_events>0; hazard_approach stays world; agent_caused_hazard stays contact) --
it only lets the operationalisation SEE the new agent-independent env source.
Same pattern V3-EXQ-510 used to fold multi_source_n_env_events into its taxonomy.

SD-047 is enabled at multi_source_intensity_scale=1.0 (ARM_2 default -- the
calibration point V3-EXQ-509 validated, ratio 2.03 in target band); the two arms
are the comparator ON/ABLATED, not an intensity sweep.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_047l_mech095_agency_routing_sd047.py --dry-run
  /opt/local/bin/python3 experiments/v3_exq_047l_mech095_agency_routing_sd047.py
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
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from _metrics import check_degeneracy


EXPERIMENT_TYPE = "v3_exq_047l_mech095_agency_routing_sd047"
CLAIM_IDS = ["MECH-095"]
EXPERIMENT_PURPOSE = "evidence"

# 047k transition-type label sets (unchanged). Enrichment folded additively.
WORLD_CAUSED = frozenset({"env_caused_hazard", "hazard_approach"})
CONTACT_SET = frozenset({"hazard_approach", "agent_caused_hazard", "env_caused_hazard"})

# SD-047 source defaults at intensity_scale=1.0 (matches V3-EXQ-509 / 510).
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
                # World-caused label folds SD-047 env events additively (strict
                # superset of 047k's transition_type-only label).
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
            env_events = int(info.get("multi_source_n_env_events", 0))

            try:
                with torch.no_grad():
                    obs_body_next  = obs_dict["body_state"]
                    obs_world_next = obs_dict["world_state"]
                    latent_next = agent.sense(obs_body_next, obs_world_next)
                    agent.clock.advance()

                    zs = latent_next.z_self.detach()
                    zw = latent_next.z_world.detach()
                    # Contact folds SD-047 env events additively (strict superset
                    # of 047k's transition_type-only contact set).
                    is_contact = (ttype in CONTACT_SET) or (env_events > 0)

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

    if len(probe_self_all) >= 20:
        X_self_all  = torch.cat(probe_self_all,  dim=0).float()
        X_world_all = torch.cat(probe_world_all, dim=0).float()
        y_action    = torch.tensor(probe_actions, dtype=torch.long)

        action_acc_self  = _fit_action_probe(X_self_all,  y_action, env.action_dim)
        action_acc_world = _fit_action_probe(X_world_all, y_action, env.action_dim)

    if n_contact_probe >= 5 and len(probe_self_no_contact) >= 5:
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
        f" n_contact={n_contact_probe}"
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
                f"\n[V3-EXQ-047l] {label} seed={seed}"
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
    mean_route_loss         = _avg(results_routed,   "mean_routing_loss")
    n_contact_min = min(
        r["n_contact_probe"] for r in results_routed + results_baseline
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

    if all_pass and routing_active:
        decision = "ceiling_lifted_regate_mech095"
    elif c1_pass and c2_pass:
        decision = "partial_discrimination"
    else:
        decision = "ceiling_confirmed_route_autopsy"

    print(f"\n[V3-EXQ-047l] Final results ({len(seeds)} seeds, SD-047 enriched):", flush=True)
    print(
        f"  contact_recall: routed={contact_recall_routed:.3f}"
        f"  baseline={contact_recall_baseline:.3f}"
        f"  improvement={contact_recall_routed - contact_recall_baseline:+.3f}",
        flush=True,
    )
    print(
        f"  action_dissoc: mean={action_dissoc_routed:+.3f}"
        f"  std={action_dissoc_std:.3f}"
        f"  mean_route_loss={mean_route_loss:.4f}",
        flush=True,
    )
    print(
        f"  decision={decision}  status={status} ({criteria_met}/5)",
        flush=True,
    )

    failure_notes: List[str] = []
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
        f" env_ev_ticks={r['train_env_event_ticks']}"
        for r in results_routed
    )
    per_baseline_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" contact_recall_world={r['contact_recall_world']:.3f}"
        f" contact_recall_self={r['contact_recall_self']:.3f}"
        f" action_dissoc={r['action_dissociation']:+.3f}"
        f" n_contact={r['n_contact_probe']}"
        f" env_ev_ticks={r['train_env_event_ticks']}"
        for r in results_baseline
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-047l -- MECH-095 TPJ Agency-Routing Retest on SD-047 Env\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-095\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Substrate:** SD-047 multi-source dynamics (intensity_scale={intensity_scale})\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**lambda_route:** {lambda_route}\n"
        f"**Warmup:** {warmup_episodes} eps  **Probe:** {probe_episodes} eps\n"
        f"**mean_routing_loss (ROUTED):** {mean_route_loss:.4f}"
        f" ({'ACTIVE' if routing_active else 'INACTIVE'})\n"
        f"**Operationalisation:** EXQ-047k routing comparator, world/contact labels\n"
        f"fold multi_source_n_env_events>0 additively. See also V3-EXQ-510\n"
        f"(counterfactual-gap comparator on SD-047 -> WOO_SPELKE flat-FAIL).\n\n"
        f"## Pre-Registered Thresholds (identical to EXQ-047k)\n\n"
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
        f"| BASELINE  | {contact_recall_baseline:.3f}          | --                           | --         |\n\n"
        f"**Recall improvement: {contact_recall_routed - contact_recall_baseline:+.3f}**\n\n"
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

    # Non-degeneracy self-report (whole-run). Load-bearing discriminative metric
    # is contact_recall_world (drives C1 + the C2 routed-vs-baseline contrast).
    # The vacuity signature is EVERY cell (ROUTED and BASELINE) pinned at the
    # balanced-probe chance level: z_world then encodes no contact anywhere, so
    # the MECH-095 mechanism was never actually exercised on this substrate --
    # the honest reading is non_contributory (ceiling still holds, route to
    # /failure-autopsy), NOT a `weakens` that would ding MECH-095 confidence.
    # floor=0.55 (just above the 0.5 chance level) => degenerate iff all 8 cells
    # <= 0.55. A genuine-but-weak effect (any cell clears 0.55) is a real result,
    # not degenerate. This is the WOO_SPELKE-analog guard (cf. V3-EXQ-510).
    all_recall_cells = [
        r["contact_recall_world"] for r in results_routed + results_baseline
    ]
    degeneracy = check_degeneracy({
        "contact_recall_world": {"values": all_recall_cells, "floor": 0.55},
    })

    metrics = {
        "contact_recall_routed":   float(contact_recall_routed),
        "contact_recall_baseline": float(contact_recall_baseline),
        "recall_improvement":      float(contact_recall_routed - contact_recall_baseline),
        "action_dissoc_routed":    float(action_dissoc_routed),
        "action_dissoc_std":       float(action_dissoc_std),
        "mean_routing_loss":       float(mean_route_loss),
        "n_contact_min":           float(n_contact_min),
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
    }

    result_dict = {
        "status": status,
        "outcome": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 3 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
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
        / "REE_assembly" / "evidence" / "experiments"
    )
    out_path = write_flat_manifest(
        result,
        out_dir,
        dry_run=args.dry_run,
        config=result.get("config"),
        seeds=None,
        script_path=Path(__file__),
    )

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
