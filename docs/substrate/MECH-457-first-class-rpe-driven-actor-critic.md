## MECH-457: first-class RPE-driven actor-critic action-learning substrate (2026-07-12)
- MECH-457: action_learning.first_class_actor_critic_substrate -- IMPLEMENTED 2026-07-12.
  Module: ree_core/action_learning/actor_critic.py (ActorCriticPolicy + ActorCriticStep).
  A dorsal-striatal-analog ACTOR (own Categorical over config.e2.action_dim) + value
  CRITIC, architecturally distinct from the lateral_pfc / ofc bias_head REINFORCE
  readout (which only nudges E3's cost argmin with a clamped +/-bias_scale scalar). Two
  critic forms via config: plain value head V(z_world) (cand-A) and successor-feature
  critic V_SF = psi(z).w (cand-B, w grounded in MECH-229 VALENCE_WANTING; biological
  anchor = hippocampal predictive map, Stachenfeld 2017).
  Config (REEConfig, all no-op default; set to enable): use_actor_critic (master),
  actor_critic_cotrain_encoder (THE co-shaping ablation lever -- frozen reads
  z_world.detach() = V3-EXQ-737's refuted arm, cotrain reads live z_world so the
  action-learning gradient co-shapes the encoder), actor_critic_use_sf_critic
  (plain vs SF critic), actor_critic_hidden (128), actor_critic_sf_feature_dim (32).
  The 4 validation arms A0..A3 = cotrain_encoder{F,T} x use_sf_critic{F,T}.
  Data flow: agent.sense() -> LatentState.z_world -> (cotrain ? z_world : z_world.detach())
  -> ActorCriticPolicy -> Categorical(logits).sample/argmax. Agent hooks:
  actor_critic_step(latent, deterministic) (applies the co-shape/detach lever, returns
  the graph-connected step), actor_critic_reward(latent) (substrate RPE teacher
  R_t = detach(e3.benefit_eval - e3.harm_eval), ARC-108), actor_critic_parameters(),
  actor_critic_encoder_parameters() (= latent_stack.parameters(), the co-shape seam
  the cotrain arm adds to the optimizer). Training UPDATE (PPO/GAE + SF-TD losses)
  lives in the experiment, reusing x734._ppo_update/_compute_gae (same convention as the
  bias_head REINFORCE update).
  Backward compatible: use_actor_critic=False -> agent.action_critic is None;
  select_action byte-identical; existing experiments unaffected (verified).
  MECH-094: not applicable (no memory writes on simulated/non-waking ticks).
  Phased training required: yes (P0 world-model warmup -> P1 actor-critic PPO;
  frozen arm holds the encoder = Stooke-2021 decoupling, cotrain arm co-adapts it).
  Routed from failure_autopsy_734-737-conversion-ceiling-competence_2026-07-11 (PRIMARY).
  Validation experiment: V3-EXQ-<TBD> queued (frozen-vs-cotrain x plain-vs-SF, 4 arms;
  denominator = V3-EXQ-738 local-view ceiling 48.05 @D3, NOT the global oracle).
  Design doc: REE_assembly/docs/architecture/sd_actor_critic_action_learning.md.
  See MECH-457 (candidate/v3_pending -- PROMOTES NOTHING until the ON/OFF validation
  runs + is reviewed), SD-056, MECH-229, ARC-108, f_dominance_conversion_ceiling.
