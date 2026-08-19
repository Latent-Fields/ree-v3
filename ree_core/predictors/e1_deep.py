"""
E1 Deep Predictor — V3 Implementation

Unchanged in function from V2: slow world model, LSTM-based, trains on
sensory prediction error.

V3 change (SD-005): E1 produces predictions over BOTH z_self and z_world
channels (associative prior). In V2, E1 predicted over unified z_gamma.
In V3, E1 reads the concatenated [z_self, z_world] vector and predicts
the next [z_self, z_world] jointly.

Module contract:
  - Input:  z_self + z_world (concatenated [batch, self_dim + world_dim])
  - Output: predicted (z_self_next, z_world_next) + prior for HippocampalModule
  - E1 is READ-ONLY with respect to z_self and z_world:
    it contributes predictions (associative prior) but does not drive
    the primary write path for either channel.

Episode-boundary semantics (preserved from V2):
E1 maintains self._hidden_state across episode steps. Reset at episode
start via reset_hidden_state(). The prediction-loss computation in REEAgent
uses a save/restore pattern so training replays do not disturb inference.
"""

import math
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ree_core.utils.config import E1Config
from ree_core.latent.stack import LatentState


class ContextMemory(nn.Module):
    """Context memory for E1's long-horizon predictions (unchanged from V2)."""

    def __init__(self, latent_dim: int, memory_dim: int = 128, num_slots: int = 16,
                 gated_content_write: bool = False,
                 write_selection: str = "argmin",
                 write_refractory_k: int = 2,
                 write_usage_weight: float = 1.0,
                 write_usage_decay: float = 0.99,
                 write_gumbel_tau_init: float = 1.0,
                 write_gumbel_tau_min: float = 0.1,
                 write_gumbel_anneal_steps: int = 2000):
        super().__init__()
        self.latent_dim = latent_dim
        self.memory_dim = memory_dim
        self.num_slots = num_slots
        self.gated_content_write = bool(gated_content_write)

        # V3-EXQ-436f follow-up (2026-08-18): write-address selection mode.
        # See write() for the defect, the mechanism, and the measurement table.
        self.write_selection = str(write_selection)
        if self.write_selection not in ("argmin", "refractory", "usage", "gumbel"):
            raise ValueError(
                "contextmemory_write_selection must be 'argmin', 'refractory', "
                f"'usage' or 'gumbel', got {self.write_selection!r}"
            )
        self.write_refractory_k = int(write_refractory_k)
        self.write_usage_weight = float(write_usage_weight)
        self.write_usage_decay = float(write_usage_decay)
        self.write_gumbel_tau_init = float(write_gumbel_tau_init)
        self.write_gumbel_tau_min = float(write_gumbel_tau_min)
        self.write_gumbel_anneal_steps = int(write_gumbel_anneal_steps)
        self._write_step = 0

        # Non-persistent so the state_dict is unchanged and existing checkpoints
        # load untouched; these are bookkeeping, not learned parameters.
        self.register_buffer(
            "slot_usage", torch.zeros(num_slots), persistent=False
        )
        # Cumulative per-slot write count + last written index. ALWAYS maintained,
        # in every mode including the legacy default, because instrumentation must
        # never have to RE-DERIVE the selection: V3-EXQ-436f's occupancy tracker
        # duplicated write()'s own `scores.mean(0).argmin()` expression to learn
        # which slot was written, which silently reports the WRONG slot the moment
        # the selection rule changes. Read these instead.
        self.register_buffer(
            "slot_write_counts", torch.zeros(num_slots), persistent=False
        )
        self._recent_write_idx: List[int] = []
        self._last_write_idx: Optional[int] = None

        self.memory = nn.Parameter(torch.randn(num_slots, memory_dim) * 0.01)
        self.query_proj = nn.Linear(latent_dim, memory_dim)
        # SD-016 Part A (2026-04-25): key_proj bias removed.
        # With memory init scale 0.01 and a default-init bias on Linear, the bias
        # term dominates over W_K @ memory_s for every slot, collapsing all keys
        # to ~b_K and softmax to uniform attention (entropy = ln(num_slots)).
        # EXQ-477 EXP-0155 measured this directly:
        #   bias_over_content_ratio = 9.88 (pre-train), 3.41 (post-P0)
        #   attn_entropy_mean = 2.773 (uniform reference 2.7726)
        #   cue_context_per_channel_std = 2.86e-08 (cue_context constant across batch)
        # bias=False is structural -- there is no bias parameter for the optimiser
        # to drift back, and key projection is a pure linear map of slot content.
        # See REE_assembly/docs/architecture/context_memory_writepath_fix.md.
        self.key_proj = nn.Linear(memory_dim, memory_dim, bias=False)
        self.value_proj = nn.Linear(memory_dim, memory_dim)
        self.output_proj = nn.Linear(memory_dim, latent_dim)
        self.write_gate = nn.Sequential(
            nn.Linear(latent_dim, memory_dim),
            nn.Sigmoid()
        )
        # V3-EXQ-436c / V3-EXQ-861a follow-up (2026-08-03): gated content write.
        #
        # The legacy write path uses write_gate's POST-SIGMOID output as the
        # write PAYLOAD (`self.memory[i] = 0.9*self.memory[i] + 0.1*write_signal`).
        # A sigmoid output is confined to (0,1) and, at this module's operating
        # point, sits within +-0.02 of 0.5 on every channel -- so every write
        # blends the slot toward the SAME vector 0.5*ones(memory_dim). Measured
        # live on a 436c-configured agent (state rms 0.078, ||x|| 0.63):
        #   ||mean write_signal||        = 5.6604  (= ||0.5*ones(128)|| = 5.6569)
        #   mean ||write_signal - mean|| = 0.0293
        #   const_over_varying_ratio     = 193.2
        #   pairwise cos sim of the vectors actually written = 0.999961
        # Replaying 436c's 800-write SWS load drives whole-bank slot cosine
        # similarity 0.007 -> 0.9999, reproducing 436c's reported homogenization
        # (~0.9999-1.0 in 4/5 seeds) exactly.
        #
        # NOTE FOR ANYONE RE-DERIVING THIS: the failure_autopsy hypothesis was
        # that write_gate carries key_proj's bias-over-content collapse
        # (SD-016 Part A / EXQ-477). The bias ratio IS elevated
        # (bias_over_content_ratio = 1.60 at the operating point, i.e. ||b||
        # exceeds ||W x||), but removing the bias DOES NOT FIX THIS and must not
        # be mistaken for a fix: measured A/B, `sigmoid(Wx+b)` -> `sigmoid(Wx)`
        # moves const_over_varying 44.87 -> 44.78 and post-load slot cosine
        # similarity 0.999910 -> 0.999914. The constant is sigmoid's own midpoint
        # (||0.5*ones(128)|| = 5.6569, matching the measured ||const|| to 4 s.f.),
        # which no change to the bias can move. key_proj's fix does not transfer.
        #
        # The actual defect is semantic: a *gate* should MODULATE content, not BE
        # content. When gated_content_write=True, write_gate is restored to its
        # namesake role and a separate linear projection carries the payload:
        #     write_signal = sigmoid(write_gate_pre(x)) * write_content(x)
        # which is zero-centred and content-bearing. Measured at the same
        # operating point: const_over_varying 44.9 -> 0.07, written-vector
        # pairwise cos sim 0.99950 -> 0.00023, and post-800-write slot cosine
        # similarity {0.9999, 0.9999, 1.0000, -0.0120, 0.2776} (seeds 0-4;
        # 3/5 fully collapsed, matching 436c's reported 4/5) ->
        # {0.0062, 0.0491, 0.0434, 0.0288, 0.0319} -- differentiation, the
        # direction SD-017/ARC-045/MECH-166 predict.
        #
        # Default False: bit-identical to the legacy path (the parameter is not
        # even constructed), so no in-flight experiment changes semantics until
        # it opts in via REEConfig.from_dims(contextmemory_gated_content_write=True).
        # Validation: re-run the 436-family under a new letter (e.g. V3-EXQ-436d).
        # bias=False is load-bearing, and is where SD-016 Part A's insight DOES
        # apply. With a default-init bias on write_content, ||b_c|| = 0.803 vs
        # mean ||W_c x|| = 0.513 at the operating point (bias_over_content_ratio
        # 1.56) -- so the payload collapses back onto a constant (sigmoid(.)*b_c)
        # and the repair only half-works: written-vector pairwise cos sim 0.707
        # and post-800-write slot cosine similarity still 0.968. With bias=False
        # the payload is a pure linear map of slot content: pairwise cos sim
        # 0.0002 and slot cosine similarity 0.16. Same reasoning, and same
        # structural (not zero-init) remedy, as key_proj above.
        if self.gated_content_write:
            self.write_content = nn.Linear(latent_dim, memory_dim, bias=False)
        else:
            self.write_content = None

    def read(self, query: torch.Tensor) -> torch.Tensor:
        batch_size = query.shape[0]
        memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        q = self.query_proj(query).unsqueeze(1)
        k = self.key_proj(memory)
        v = self.value_proj(memory)
        scores = torch.bmm(q, k.transpose(1, 2)) / (self.memory_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, v).squeeze(1)
        return self.output_proj(context)

    def write(self, state: torch.Tensor) -> None:
        """Blend `state`'s write payload into one slot, chosen by write_selection.

        THE DEFECT THIS ADDRESSES (substrate_queue
        `contextmemory-write-path-addressing-degeneracy`, severity `corrupting`):
        the legacy address rule is a hard `scores.mean(0).argmin()`. The write
        PAYLOAD (`write_gate(x) * write_content(x)`) is an unaligned projection of
        the state relative to the ADDRESS space (`query_proj(x)`), so a write to
        the argmin slot can push that slot FURTHER from the query -- which
        re-selects it on the next write, forever. V3-EXQ-436e established the
        closed-form sign discriminator `q . (write_signal - memory[argmin])`:
        negative predicts LOCK, positive predicts ROTATE, 5/5. Under a
        near-constant query stream the lock is the common case, and it is
        CORRUPTING rather than merely wrong because write() returns normally,
        thousands of calls are logged, and the resulting 1-slot bank yields a
        well-formed null that reads as a genuine "sleep has no effect" finding.
        Confirmed twice: V3-EXQ-436e and V3-EXQ-436f (n_occupied_slots = 1 of 16
        in BOTH arms on 3/5 seeds despite 2,837-4,903 write() calls per arm).

        MODES, and the measurement that should govern the choice between them.
        Probe: 5 seeds x 3000 writes, memory_dim 128, num_slots 16, at the
        measured operating point (state rms 0.078). Two streams -- "degenerate"
        is the near-constant stream that triggers the lock; "2-context" is a
        varied stream on which the differentiation DV is meaningful.

          mode                 degenerate: seeds   2-context: occupied-slot
                               with >= 2 slots     cosine sim / cluster Jaccard
          ------------------   -----------------   ----------------------------
          argmin (legacy)      3/5   (LOCKS)       +0.6060  /  0.329
          refractory k=2       5/5                 +0.5919  /  0.364
          usage                5/5                 (content-blind)
          gumbel + usage       5/5                 +0.7525  /  1.000

        Read the second column before trusting the first. `usage` and `gumbel`
        reach full occupancy by making selection CONTENT-BLIND: at Jaccard 1.000
        both contexts write to the SAME slot set, i.e. the bank is filled by
        noise rather than by context, and the occupied-slot cosine similarity
        (the DV that SD-017/ARC-045/MECH-166 actually turn on, lower = more
        differentiated) gets WORSE than the legacy path it replaces. They are
        implemented because the substrate_queue entry names both, and because
        they are the correct comparison arms for the validation experiment --
        NOT because occupancy alone recommends them. Selecting one of them on
        the strength of the n_occupied floor would swap this corrupting defect
        for a quieter one.

        `refractory` is the recommended mode: the k most-recently-written slots
        are ineligible, so the single-slot fixed point CANNOT form (occupancy is
        structurally >= k+1), while selection among the remaining slots is the
        unmodified content argmin -- which is why it is the only mode that holds
        context-conditioning at the legacy level.
        """
        write_signal = self.write_gate(state)
        if self.gated_content_write:
            # Gate modulates content rather than serving as the content itself.
            # See the __init__ note for the measured pathology this repairs.
            write_signal = write_signal * self.write_content(state)
        with torch.no_grad():
            query = self.query_proj(state)
            scores = torch.mm(query, self.memory.t())
            sim = scores.mean(0)
            min_idx = self._select_write_slot(sim)
            self.memory.data[min_idx] = (
                0.9 * self.memory.data[min_idx] + 0.1 * write_signal.mean(0)
            )
            self._record_write(int(min_idx))

    def _select_write_slot(self, sim: torch.Tensor):
        """Resolve the write address from per-slot similarity `sim` [num_slots].

        Runs inside write()'s torch.no_grad() block. Deliberately has NO
        straight-through estimator on the gumbel branch, unlike the read path's
        _sd016_gumbel_select: there is no gradient here to pass through, so the
        ST algebra would be dead code that only implied a learning signal that
        does not exist.
        """
        mode = self.write_selection
        if mode == "argmin":
            return sim.argmin()  # legacy: bit-identical, tensor index preserved

        if mode == "refractory":
            k = max(self.write_refractory_k, 0)
            if k <= 0 or not self._recent_write_idx:
                return sim.argmin()
            eligible = sim.clone()
            # Never mask every slot: with k >= num_slots there would be nothing
            # left to write to. Cap the horizon at num_slots - 1.
            k = min(k, self.num_slots - 1)
            for prev in self._recent_write_idx[-k:]:
                eligible[prev] = float("inf")
            return eligible.argmin()

        # "usage" and "gumbel" share an availability score. sim is standardised
        # so the usage penalty has a comparable scale regardless of the (tiny,
        # measured ~0.013) natural spread of sim across slots.
        avail = -self._zscore(sim)
        if self.write_usage_weight > 0.0:
            avail = avail - self.write_usage_weight * self._zscore(self.slot_usage)
        if mode == "usage":
            return avail.argmax()

        # "gumbel": anneal tau linearly from init to min over anneal_steps writes.
        anneal_steps = max(self.write_gumbel_anneal_steps, 1)
        frac = min(self._write_step / anneal_steps, 1.0)
        tau = max(
            self.write_gumbel_tau_init
            + frac * (self.write_gumbel_tau_min - self.write_gumbel_tau_init),
            1e-3,
        )
        u = torch.rand_like(avail).clamp(min=1e-10, max=1.0 - 1e-10)
        gumbel_noise = -torch.log(-torch.log(u))
        return (avail / tau + gumbel_noise).argmax()

    @staticmethod
    def _zscore(v: torch.Tensor) -> torch.Tensor:
        """Standardise across slots. clamp_min on the std keeps an all-equal
        vector (e.g. slot_usage before the first write) at exactly zero rather
        than amplifying float noise into a spurious penalty."""
        return (v - v.mean()) / v.std().clamp_min(1e-6)

    def _record_write(self, idx: int) -> None:
        """Bookkeeping after a write. Runs in EVERY mode, including the legacy
        default, so instrumentation can read the slot that was actually written
        instead of re-deriving it (see __init__)."""
        self._last_write_idx = idx
        self._write_step += 1
        self.slot_write_counts[idx] += 1.0
        self.slot_usage.mul_(self.write_usage_decay)
        self.slot_usage[idx] += 1.0 - self.write_usage_decay
        self._recent_write_idx.append(idx)
        # Bounded: only the last (num_slots - 1) entries can ever be consulted.
        if len(self._recent_write_idx) > self.num_slots:
            del self._recent_write_idx[:-self.num_slots]

    @property
    def last_write_index(self) -> Optional[int]:
        """Slot index mutated by the most recent write(), or None before any."""
        return self._last_write_idx

    def occupied_slots(self) -> List[int]:
        """Slot indices written to at least once since construction."""
        return [i for i in range(self.num_slots)
                if float(self.slot_write_counts[i]) > 0.0]

    def compute_diversification_loss(self) -> torch.Tensor:
        # SD-016 Path 1 (2026-04-25): mean squared off-diagonal cosine similarity
        # of slot vectors. Pushes slots toward mutual orthogonality on the unit
        # sphere. Returns a scalar tensor; gradient flows into self.memory.
        # See REE_assembly/docs/architecture/sd_016_writepath_v3_diversification_loss.md.
        slot_norm = F.normalize(self.memory, dim=-1)
        sim = slot_norm @ slot_norm.T
        n = self.num_slots
        mask = 1.0 - torch.eye(n, device=sim.device, dtype=sim.dtype)
        return (sim * mask).pow(2).sum() / (n * (n - 1))


class E1DeepPredictor(nn.Module):
    """
    E1 Deep Predictor — V3.

    Long-horizon world model, LSTM-based. Trains on sensory prediction error
    over [z_self, z_world] concatenated latent.

    Provides associative prior into HippocampalModule (SD-002, preserved from V2).
    Prior is now over world_dim (for HippocampalModule terrain conditioning).
    """

    def __init__(self, config: Optional[E1Config] = None):
        super().__init__()
        self.config = config or E1Config()

        # V3: total latent dim = self_dim + world_dim
        total_dim = self.config.self_dim + self.config.world_dim

        self.context_memory = ContextMemory(
            latent_dim=total_dim,
            memory_dim=self.config.hidden_dim,
            num_slots=16,
            gated_content_write=getattr(
                self.config, "contextmemory_gated_content_write", False
            ),
            write_selection=getattr(
                self.config, "contextmemory_write_selection", "argmin"
            ),
            write_refractory_k=getattr(
                self.config, "contextmemory_write_refractory_k", 2
            ),
            write_usage_weight=getattr(
                self.config, "contextmemory_write_usage_weight", 1.0
            ),
            write_usage_decay=getattr(
                self.config, "contextmemory_write_usage_decay", 0.99
            ),
            write_gumbel_tau_init=getattr(
                self.config, "contextmemory_write_gumbel_tau_init", 1.0
            ),
            write_gumbel_tau_min=getattr(
                self.config, "contextmemory_write_gumbel_tau_min", 0.1
            ),
            write_gumbel_anneal_steps=getattr(
                self.config, "contextmemory_write_gumbel_anneal_steps", 2000
            ),
        )

        self.transition_rnn = nn.LSTM(
            input_size=total_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=0.1 if self.config.num_layers > 1 else 0,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, total_dim),
        )

        # Prior generator: for HippocampalModule (SD-002).
        # In V3 the prior is over world_dim (terrain conditioning).
        self.prior_generator = nn.Sequential(
            nn.Linear(total_dim * 2, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.world_dim),
        )

        self._hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # MECH-120: SHY-analog offline mode flag.
        # When True, context_memory.write() calls are suppressed during waking steps.
        # Set via REEAgent.enter_offline_mode() / exit_offline_mode().
        # This prevents new waking observations from overwriting fresh schema slots
        # installed during the SWS-analog pass.
        self._offline_mode: bool = False

        # MECH-116: optional goal conditioning projection
        # Projects [z_self, z_world, z_goal] -> latent_dim before LSTM
        goal_dim = getattr(config, 'goal_dim', 0)
        self._goal_dim = goal_dim
        if goal_dim > 0:
            self.goal_input_proj = nn.Linear(
                config.latent_dim + goal_dim, config.latent_dim
            )
        else:
            self.goal_input_proj = None

        # MECH-216: E1 predictive wanting (schema readout head).
        # Linear(hidden_dim, 1) + Sigmoid on LSTM top-layer hidden state -> schema_salience [0,1].
        # High salience at resource-proximal positions seeds VALENCE_WANTING before contact.
        self._schema_wanting_enabled = getattr(config, 'schema_wanting_enabled', False)
        if self._schema_wanting_enabled:
            self.schema_readout_head = nn.Sequential(
                nn.Linear(self.config.hidden_dim, 1),
                nn.Sigmoid(),
            )
        self._last_schema_salience: Optional[torch.Tensor] = None

        # SD-016: frontal cue-indexed integration (MECH-150/151/152, ARC-041).
        # Three new projections gated by sd016_enabled for backward compatibility.
        # world_query_proj: projects z_world -> memory_dim for z_world-only query
        #   (bypasses ContextMemory.query_proj which expects full latent_dim input).
        # cue_action_proj:  cue_context [latent_dim] -> action_bias [action_object_dim]
        # cue_terrain_proj: cue_context [latent_dim] -> 2 (w_harm, w_goal logits)
        if getattr(config, 'sd016_enabled', False):
            cue_context_dim = self.config.self_dim + self.config.world_dim  # = latent_dim = 64
            action_object_dim = getattr(config, 'action_object_dim', 16)
            self.world_query_proj = nn.Linear(self.config.world_dim, self.config.hidden_dim)
            # EXQ-449a fix: cue_action_proj now consumes [cue_context, z_world]. The
            # ContextMemory attention path collapses to uniform softmax (entropy = ln(num_slots))
            # because key_proj's bias dominates over the small-init memory slots, so cue_context
            # is constant across batch and cue_action_proj output had per-channel std ~2.7e-8.
            # Concatenating z_world guarantees per-input variation in action_bias.
            self.cue_action_proj  = nn.Linear(cue_context_dim + self.config.world_dim, action_object_dim)
            self.cue_terrain_proj = nn.Linear(cue_context_dim, 2)

            # SD-016 Path 4 (V3-EXQ-418g): learnable attention temperature on the
            # z_world-only ContextMemory query. EXQ-418e diagnosed slot-side
            # diversification as insufficient: even with perfectly orthogonal slots
            # (A2_div_only mean cosine ~1e-3), attention stayed pinned at the
            # uniform rail (attn_entropy_mean ~ ln(16) = 2.7726) and cue_context
            # made zero behavioural contribution. Replacing the fixed
            # sqrt(memory_dim) divisor with exp(log_tau) gives the optimiser a
            # per-pass selectivity knob; pair with compute_attention_entropy_loss
            # to apply gradient pressure toward peaky attention.
            # Initialised at log(sqrt(memory_dim)) so step-0 behaviour is bit-
            # identical to the fixed-temperature baseline -- the optimiser can
            # then drive log_tau down for sharper attention if it pays off.
            self._sd016_temperature_learnable = bool(
                getattr(config, 'sd016_temperature_learnable', False)
            )
            if self._sd016_temperature_learnable:
                init_log_tau = math.log(self.config.hidden_dim ** 0.5)
                self.sd016_log_temperature = nn.Parameter(
                    torch.tensor(init_log_tau, dtype=torch.float32)
                )
            else:
                self.sd016_log_temperature = None

            # SD-016 Path 3 (V3-EXQ-418i follow-up): feedforward cue->slot tagger.
            # V3-EXQ-418i established the q.k attention slot-selection is pinned
            # at the uniform ln(num_slots) saddle ("the attention bottleneck is
            # categorically in query selectivity"); Path 1 div-loss was exhausted
            # across weights 1.0/2.0/5.0. The tagger replaces ONLY the slot-
            # selection scores in extract_cue_context: z_world -> slot logits via
            # a fresh MLP that sits off the uniform saddle, so the existing
            # cue_terrain_proj terrain_loss gradient flows back into it and shapes
            # contextual selectivity. The slot-content path (value_proj/output_proj)
            # is untouched. Bit-identical when sd016_cue_slot_tagger is False.
            self._sd016_cue_slot_tagger = bool(
                getattr(config, 'sd016_cue_slot_tagger', False)
            )
            self._sd016_cue_slot_tagger_temperature = float(
                getattr(config, 'sd016_cue_slot_tagger_temperature', 1.0)
            )
            # SD-016 H3 (V3-EXQ-898 autopsy follow-up): hard/competitive
            # selection operator swap on the tagger's slot-selection scores.
            # "soft" default is bit-identical to the softmax path above.
            self._sd016_cue_slot_tagger_selection = str(
                getattr(config, 'sd016_cue_slot_tagger_selection', 'soft')
            )
            if self._sd016_cue_slot_tagger_selection not in ('soft', 'gumbel', 'topk'):
                raise ValueError(
                    "sd016_cue_slot_tagger_selection must be 'soft', 'gumbel' or "
                    f"'topk', got {self._sd016_cue_slot_tagger_selection!r}"
                )
            self._sd016_cue_slot_tagger_topk_k = int(
                getattr(config, 'sd016_cue_slot_tagger_topk_k', 1)
            )
            self._sd016_cue_slot_tagger_gumbel_tau_init = float(
                getattr(config, 'sd016_cue_slot_tagger_gumbel_tau_init', 1.0)
            )
            self._sd016_cue_slot_tagger_gumbel_tau_min = float(
                getattr(config, 'sd016_cue_slot_tagger_gumbel_tau_min', 0.1)
            )
            self._sd016_cue_slot_tagger_gumbel_anneal_steps = int(
                getattr(config, 'sd016_cue_slot_tagger_gumbel_anneal_steps', 2000)
            )
            self._sd016_gumbel_step = 0  # forward-call counter for the annealed schedule
            if self._sd016_cue_slot_tagger:
                tagger_hidden = int(getattr(config, 'sd016_cue_slot_tagger_hidden', 32))
                self.cue_slot_tagger = nn.Sequential(
                    nn.Linear(self.config.world_dim, tagger_hidden),
                    nn.ReLU(),
                    nn.Linear(tagger_hidden, self.context_memory.num_slots),
                )
            else:
                self.cue_slot_tagger = None
        else:
            self._sd016_temperature_learnable = False
            self.sd016_log_temperature = None
            self._sd016_cue_slot_tagger = False
            self._sd016_cue_slot_tagger_temperature = 1.0
            self._sd016_cue_slot_tagger_selection = 'soft'
            self._sd016_cue_slot_tagger_topk_k = 1
            self._sd016_cue_slot_tagger_gumbel_tau_init = 1.0
            self._sd016_cue_slot_tagger_gumbel_tau_min = 0.1
            self._sd016_cue_slot_tagger_gumbel_anneal_steps = 2000
            self._sd016_gumbel_step = 0
            self.cue_slot_tagger = None

    def reset_hidden_state(self) -> None:
        """Reset hidden state for a new episode."""
        self._hidden_state = None

    @property
    def total_latent_dim(self) -> int:
        return self.config.self_dim + self.config.world_dim

    def predict_long_horizon(
        self,
        current_state: torch.Tensor,
        horizon: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Predict latent states over a long horizon.

        Args:
            current_state: Concatenated [z_self, z_world] [batch, self_dim+world_dim]
            horizon: Number of steps to predict

        Returns:
            Predicted states [batch, horizon, self_dim+world_dim]
        """
        horizon = horizon or self.config.prediction_horizon
        batch_size = current_state.shape[0]
        device = current_state.device

        context = self.context_memory.read(current_state)
        combined = torch.cat([current_state, context], dim=-1)
        prior = self.prior_generator(combined)

        # prior is world_dim; expand back to total_dim for LSTM input
        # by concatenating with zeros for z_self part
        prior_self = torch.zeros(batch_size, self.config.self_dim, device=device)
        prior_full = torch.cat([prior_self, prior], dim=-1)  # [batch, total_dim]

        if self._hidden_state is None or self._hidden_state[0].shape[1] != batch_size:
            h0 = torch.zeros(
                self.config.num_layers, batch_size, self.config.hidden_dim, device=device
            )
            c0 = torch.zeros(
                self.config.num_layers, batch_size, self.config.hidden_dim, device=device
            )
            self._hidden_state = (h0, c0)

        predictions = []
        hidden = self._hidden_state
        input_state = prior_full.unsqueeze(1)  # [batch, 1, total_dim]

        for _ in range(horizon):
            output, hidden = self.transition_rnn(input_state, hidden)
            predicted = self.output_proj(output.squeeze(1))
            predictions.append(predicted)
            input_state = predicted.unsqueeze(1)

        self._hidden_state = (hidden[0].detach(), hidden[1].detach())

        # MECH-216: extract schema salience from LSTM top-layer hidden state.
        if self._schema_wanting_enabled:
            h_top = hidden[0][-1]  # [batch, hidden_dim] — top layer
            self._last_schema_salience = self.schema_readout_head(h_top)  # [batch, 1]
        else:
            self._last_schema_salience = None

        return torch.stack(predictions, dim=1)  # [batch, horizon, total_dim]

    def generate_prior(self, current_state: torch.Tensor) -> torch.Tensor:
        """
        Generate world-domain prior for HippocampalModule conditioning (SD-002).

        Args:
            current_state: Concatenated [z_self, z_world] [batch, total_dim]

        Returns:
            Prior over z_world [batch, world_dim]
        """
        context = self.context_memory.read(current_state)
        combined = torch.cat([current_state, context], dim=-1)
        return self.prior_generator(combined)

    def extract_cue_context(
        self,
        z_world: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        E1 frontal cue-indexed association retrieval (SD-016, MECH-150).

        Queries ContextMemory with z_world ONLY (exteroceptive cues only),
        bypassing ContextMemory.query_proj (which expects full latent_dim input)
        via a dedicated world_query_proj (world_dim -> memory_dim).

        Unlike generate_prior() -- which queries with [z_self, z_world] and
        produces a bulk terrain prior for HippocampalModule -- this method
        queries with z_world alone and projects to two downstream signals:

            action_bias:    E1 -> E2 affordance modulation (MECH-151)
            terrain_weight: E1 -> E3 precision modulation  (MECH-152)

        Must only be called when sd016_enabled=True (attribute check: hasattr(e1, 'world_query_proj')).

        Args:
            z_world: [batch, world_dim] -- exteroceptive latent

        Returns:
            action_bias:    [batch, action_object_dim] -- additive to E2.action_object o_t
            terrain_weight: [batch, 2] in (0, 1) -- [w_harm, w_goal] for E3 scoring
        """
        batch_size = z_world.shape[0]
        memory_dim = self.context_memory.memory_dim

        # z_world-only retrieval over ContextMemory slots. The slot-CONTENT path
        # (value_proj of the slots) is shared by both selection branches; only the
        # slot-SELECTION weights differ.
        memory = self.context_memory.memory  # [num_slots, memory_dim]
        v = self.context_memory.value_proj(memory).unsqueeze(0).expand(batch_size, -1, -1)

        if getattr(self, '_sd016_cue_slot_tagger', False) and self.cue_slot_tagger is not None:
            # SD-016 Path 3 (V3-EXQ-418i follow-up): feedforward tagger replaces the
            # saddle-stuck q.k attention. z_world -> slot logits via a fresh MLP that
            # sits off the uniform softmax saddle, so terrain_loss gradient (which
            # trains cue_terrain_proj) flows back into it and shapes selectivity.
            slot_logits = self.cue_slot_tagger(z_world)                    # [batch, num_slots]
            temp = max(self._sd016_cue_slot_tagger_temperature, 1e-3)
            selection_mode = getattr(self, '_sd016_cue_slot_tagger_selection', 'soft')
            if selection_mode == 'gumbel':
                weights = self._sd016_gumbel_select(slot_logits, temp).unsqueeze(1)
            elif selection_mode == 'topk':
                weights = self._sd016_topk_select(slot_logits, temp).unsqueeze(1)
            else:
                weights = F.softmax(slot_logits / temp, dim=-1).unsqueeze(1)   # [batch, 1, num_slots]
        else:
            # Legacy path: world_query_proj maps z_world -> memory_dim (bypasses
            # query_proj), then q.k attention over the slot keys.
            q = self.world_query_proj(z_world).unsqueeze(1)               # [batch, 1, memory_dim]
            k = self.context_memory.key_proj(memory).unsqueeze(0).expand(batch_size, -1, -1)
            scale = self._sd016_attention_scale(memory_dim)
            scores = torch.bmm(q, k.transpose(1, 2)) / scale              # [batch, 1, num_slots]
            weights = F.softmax(scores, dim=-1)                           # [batch, 1, num_slots]

        # Read-only diagnostic: cache the last slot-selection distribution so a
        # validation experiment can measure selection entropy (the V3-EXQ-418i
        # bottleneck metric: uniform == ln(num_slots) ~ 2.773 for 16 slots).
        self._last_cue_slot_weights = weights.detach().squeeze(1)         # [batch, num_slots]

        context = torch.bmm(weights, v).squeeze(1)                         # [batch, memory_dim]

        cue_context = self.context_memory.output_proj(context)  # [batch, latent_dim=64]

        # EXQ-449a fix: feed z_world directly alongside cue_context so the action_bias
        # has a non-collapsed input path (cue_context is constant under uniform attention).
        action_bias    = self.cue_action_proj(torch.cat([cue_context, z_world], dim=-1))
        terrain_weight = torch.sigmoid(self.cue_terrain_proj(cue_context))    # [batch, 2] in (0,1)
        return action_bias, terrain_weight

    def _sd016_gumbel_select(self, slot_logits: torch.Tensor, temp: float) -> torch.Tensor:
        """SD-016 H3: annealed straight-through Gumbel-softmax selection.

        Forward pass is a one-hot argmax over Gumbel-perturbed logits --
        structurally sparse regardless of what terrain_loss demands.
        Backward pass flows through the soft relaxation at the current
        annealed temperature (straight-through estimator). Temperature
        anneals linearly from sd016_cue_slot_tagger_gumbel_tau_init down to
        sd016_cue_slot_tagger_gumbel_tau_min over
        sd016_cue_slot_tagger_gumbel_anneal_steps TRAINING-mode forward
        calls; eval-mode calls neither advance the schedule nor sample
        Gumbel noise, so inference is deterministic and reproducible.
        """
        anneal_steps = max(self._sd016_cue_slot_tagger_gumbel_anneal_steps, 1)
        frac = min(self._sd016_gumbel_step / anneal_steps, 1.0)
        tau_init = self._sd016_cue_slot_tagger_gumbel_tau_init
        tau_min = self._sd016_cue_slot_tagger_gumbel_tau_min
        tau = max(tau_init + frac * (tau_min - tau_init), 1e-3)

        logits = slot_logits / max(temp, 1e-3)
        if self.training:
            eps = 1e-10
            u = torch.rand_like(logits).clamp(min=eps, max=1.0 - eps)
            gumbel_noise = -torch.log(-torch.log(u))
            y = logits + gumbel_noise
            self._sd016_gumbel_step += 1
        else:
            y = logits  # deterministic at eval: no sampling noise

        soft = F.softmax(y / tau, dim=-1)
        index = soft.argmax(dim=-1, keepdim=True)
        hard = torch.zeros_like(soft).scatter_(-1, index, 1.0)
        return hard + soft - soft.detach()  # straight-through estimator

    def _sd016_topk_select(self, slot_logits: torch.Tensor, temp: float) -> torch.Tensor:
        """SD-016 H3: straight-through top-k selection.

        Forward pass keeps only the top-k softmax-weighted slots
        (renormalised to sum to 1) -- structurally sparse regardless of
        what terrain_loss demands. Backward pass flows through the full
        softmax gradient, mirroring the gumbel path's ST mechanics.
        k = sd016_cue_slot_tagger_topk_k, clamped to [1, num_slots].
        """
        soft = F.softmax(slot_logits / max(temp, 1e-3), dim=-1)
        num_slots = soft.shape[-1]
        k = max(1, min(self._sd016_cue_slot_tagger_topk_k, num_slots))
        topk_vals, topk_idx = soft.topk(k, dim=-1)
        hard = torch.zeros_like(soft).scatter_(-1, topk_idx, topk_vals)
        hard = hard / hard.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return hard + soft - soft.detach()  # straight-through estimator

    def _sd016_attention_scale(self, memory_dim: int):
        """Resolve the divisor used inside the SD-016 z_world-only attention.

        Returns the learnable exp(log_tau) when sd016_temperature_learnable
        is set, otherwise the fixed sqrt(memory_dim) scale used pre-EXQ-418g.
        """
        if self._sd016_temperature_learnable and self.sd016_log_temperature is not None:
            return torch.exp(self.sd016_log_temperature) + 1e-3
        return memory_dim ** 0.5

    def compute_attention_entropy_loss(self, z_world: torch.Tensor) -> torch.Tensor:
        """SD-016 Path 4: attention-entropy minimisation on the z_world-only query.

        Recomputes the SD-016 attention pattern (matching extract_cue_context's
        scale resolution, including the optional learnable temperature) and
        returns the mean Shannon entropy across the batch. Minimising this
        loss pushes world_query_proj + key_proj (and, when enabled,
        sd016_log_temperature) toward queries that select specific slots
        rather than producing uniform softmax. Targets the EXQ-418e diagnostic
        finding that diversified slots alone do not break the uniform-rail
        attention regime.
        """
        if not getattr(self.config, 'sd016_enabled', False):
            return torch.zeros((), device=z_world.device, dtype=z_world.dtype)
        batch_size = z_world.shape[0]
        memory_dim = self.context_memory.memory_dim
        q = self.world_query_proj(z_world).unsqueeze(1)
        memory = self.context_memory.memory
        k = self.context_memory.key_proj(memory).unsqueeze(0).expand(batch_size, -1, -1)
        scale = self._sd016_attention_scale(memory_dim)
        scores = torch.bmm(q, k.transpose(1, 2)) / scale
        weights = F.softmax(scores, dim=-1).squeeze(1)
        probs = weights.clamp(min=1e-12)
        return -(probs * probs.log()).sum(dim=-1).mean()

    def compute_context_divergence_loss(
        self,
        z_world_safe: torch.Tensor,
        z_world_dangerous: torch.Tensor,
    ) -> torch.Tensor:
        """SD-016 H1 (V3-EXQ-907 CONFIRMED 2026-08-10): context-divergence
        auxiliary loss, promoted verbatim from the V3-EXQ-907 driver's
        _context_divergence probe into substrate (per that script's own
        "SUBSTRATE REQUIREMENT" note: "If H1 confirms the mechanism helps,
        THEN promote it to a real substrate knob").

        Rewards cue_slot_tagger for producing DIFFERENT mean slot-selection
        distributions on a safe-context batch vs a dangerous-context batch --
        the one signal no prior SD-016 selection mechanism was ever given,
        and the drive V3-EXQ-907 found breaks the uniform-softmax saddle
        (confirmed at every lambda in {0.1, 0.5, 2.0} tested). Calls
        cue_slot_tagger DIRECTLY (the _last_cue_slot_weights diagnostic
        cache is .detach()ed and carries no gradient), exactly as the
        driver did.

        Already weighted by sd016_context_divergence_weight and already
        negated -- this term REWARDS divergence, so it is ready to ADD
        directly to a total loss (mirrors LAMBDA_TERRAIN * terrain_loss's
        additive convention; the sign flip needed to turn "maximise
        divergence" into "minimise this loss" is applied here, not by
        the caller). Returns a zero tensor with no graph -- no-op,
        bit-identical to every other SD-016 loss term's off-state -- when
        the weight is 0.0 (default) or the tagger is disabled.

        Args:
            z_world_safe:      [batch, world_dim] safe-context z_world batch
                                (typically detached and fixed for the P1
                                training window, matching V3-EXQ-907).
            z_world_dangerous: [batch, world_dim] dangerous-context z_world
                                batch, same convention.

        Returns:
            Scalar loss contribution, already weighted -- ADD to total_loss.
        """
        weight = float(getattr(self.config, 'sd016_context_divergence_weight', 0.0))
        if weight <= 0.0 or self.cue_slot_tagger is None:
            return torch.zeros((), device=z_world_safe.device, dtype=z_world_safe.dtype)

        temp = max(self._sd016_cue_slot_tagger_temperature, 1e-3)
        logits_safe = self.cue_slot_tagger(z_world_safe)   # [batch, num_slots]
        logits_dang = self.cue_slot_tagger(z_world_dangerous)
        w_safe = F.softmax(logits_safe / temp, dim=-1).mean(dim=0)   # [num_slots]
        w_dang = F.softmax(logits_dang / temp, dim=-1).mean(dim=0)
        divergence = (w_safe - w_dang).abs().sum()
        return -weight * divergence

    def get_schema_salience(self) -> Optional[torch.Tensor]:
        """MECH-216: return last schema salience [batch, 1] or None."""
        return self._last_schema_salience

    def compute_schema_readout_loss(
        self, resource_proximity_target: torch.Tensor
    ) -> torch.Tensor:
        """MECH-216: MSE loss between schema_salience and resource proximity target.

        Args:
            resource_proximity_target: [batch, 1] float in [0, 1].
        Returns:
            Scalar MSE loss (0.0 if schema wanting disabled or no salience).
        """
        if not self._schema_wanting_enabled or self._last_schema_salience is None:
            return torch.tensor(0.0, device=resource_proximity_target.device)
        return F.mse_loss(self._last_schema_salience, resource_proximity_target)

    def split_prediction(
        self, prediction: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split a prediction tensor into (z_self_pred, z_world_pred)."""
        z_self_pred  = prediction[:, :self.config.self_dim]
        z_world_pred = prediction[:, self.config.self_dim:]
        return z_self_pred, z_world_pred

    def shy_normalise(self, decay: float = 0.85) -> None:
        """
        MECH-120: SHY-analog synaptic homeostasis normalisation.

        Decays ContextMemory slot weights toward the slot-mean, flattening
        dominant attractors before replay repopulates them (Tononi SHY hypothesis).
        This must run BEFORE replay (Phase 1 before Phase 2 in offline_phases.md):
        replaying into a landscape dominated by a recent high-salience experience
        reinforces the dominant trace rather than consolidating diverse content.

        decay=0.85 preserves 85% of each slot's deviation from the mean.
        Lower decay = more aggressive normalisation.
        No gradient: direct .data write.

        Args:
            decay: EMA weight toward mean. 0.85 = mild homeostasis.
        """
        with torch.no_grad():
            mem = self.context_memory.memory.data   # [num_slots, memory_dim]
            slot_mean = mem.mean(dim=0, keepdim=True)  # [1, memory_dim]
            # Decay deviation from mean: new = mean + (old - mean) * decay
            self.context_memory.memory.data = slot_mean + (mem - slot_mean) * decay

    def update_from_observation(
        self,
        observation_state: torch.Tensor,
        prediction_error: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if not self._offline_mode:
            self.context_memory.write(observation_state)
        return {
            "e1_error_magnitude": prediction_error.pow(2).mean(),
            "context_updated": torch.tensor(float(not self._offline_mode)),
        }

    def integrate_experience(
        self,
        experience_buffer: List[torch.Tensor],
        num_iterations: int = 10,
    ) -> Dict[str, float]:
        if len(experience_buffer) < 2:
            return {"integration_loss": 0.0}

        total_loss = 0.0
        for _ in range(num_iterations):
            self.reset_hidden_state()
            start_idx = int(torch.randint(0, len(experience_buffer) - 1, (1,)).item())
            end_idx = min(start_idx + self.config.prediction_horizon, len(experience_buffer))
            sequence = torch.stack(experience_buffer[start_idx:end_idx])
            if sequence.dim() == 2:
                sequence = sequence.unsqueeze(0)
            initial = sequence[:, 0, :]
            predictions = self.predict_long_horizon(initial, horizon=sequence.shape[1] - 1)
            targets = sequence[:, 1:, :]
            loss = F.mse_loss(predictions[:, :targets.shape[1], :], targets)
            total_loss += loss.item()

        return {"integration_loss": total_loss / num_iterations}

    def forward(
        self,
        current_state: torch.Tensor,
        horizon: Optional[int] = None,
        z_goal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            current_state: [z_self, z_world] concatenated [batch, total_dim]
            z_goal:        optional goal latent [1, goal_dim] for MECH-116 conditioning

        Returns:
            predictions: [batch, horizon, total_dim]
            prior:       world-domain prior [batch, world_dim] for HippocampalModule
        """
        # MECH-116: apply goal conditioning if provided
        total_state = current_state
        if self.goal_input_proj is not None and z_goal is not None:
            goal_exp = z_goal.expand(total_state.shape[0], -1)
            total_state = self.goal_input_proj(
                torch.cat([total_state, goal_exp], dim=-1)
            )
        predictions = self.predict_long_horizon(total_state, horizon)
        prior = self.generate_prior(current_state)
        return predictions, prior
