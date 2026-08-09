"""BLAAttributionHead -- SD-035 / MECH-074d trainable predictor-attribution head.

Why this module exists
----------------------
MECH-074d's registration text (2026-04-21) already said what this file is:

    "For the initial non-trainable pass this is approximated by selecting
     codes whose contribution to the harm-PE exceeds a threshold; a
     learnable attribution head is deferred."

That deferred second pass is what this module builds. Two independent
experiments closed off the alternative reading first:

  V3-EXQ-894  (FAIL, weakens) -- attribution selectivity C1/C2 do not reach
      2/3 seeds; flagged an over-firing / dilution confound (fire-fraction
      inversely correlated with the attribution signal across seeds).
  V3-EXQ-894a (FAIL, weakens) -- swept `bla_remap_pe_sigma_threshold` over
      [1.0, 1.5, 2.0, 2.5] to test exactly that confound, and REFUTED it:
      mean attribution-mass-excess falls monotonically with sigma
      (Spearman -1.0), tracking the fall in fire-fraction rather than
      diverging as dilution would predict. The per-seed outcome (one
      selective, one context-blind-deterministic, one null) is stable at
      every threshold.

The diagnosis (failure_autopsy_V3-EXQ-894a_2026-08-08) is
`competence_implementation_gap`: the gate mechanics work, but the
attribution computation is a fixed, non-trainable rule -- the slot-attention
softmax over ContextMemory in `REEAgent._get_context_memory_code_contributions`
-- which has no learned component that could differentiate genuinely
context-predictive codes from incidentally-high-contribution ones, and no
mechanism by which it could become more selective with experience.

Biological grounding
--------------------
Moita, Rosis, Zhou, LeDoux & Blair 2004 (J Neurosci) -- amygdala-driven
place-cell remapping is selective for CONTEXT-predictive representations
(contextual Z = -1.36 vs auditory Z = -0.34, p = 0.02), not a broadcast
update. Crucially for this module, that dissociation DEVELOPS OVER
TRAINING: real BLA attribution is a learned cue-outcome association, which
is the property the fixed threshold rule structurally cannot have.

Nader, Schafe & LeDoux 2000 (Nature) -- reconsolidation necessity; supports
the PE-triggered half of the gate (unchanged by this module).

What the head computes
----------------------
A learned, context-conditioned attribution over ContextMemory slots,
supervised by whether attending to a slot helps PREDICT the affective harm
stream. A slot's attribution weight is high exactly when reading that slot
lets the head predict z_harm_a -- i.e. when the code is harm-predictive in
the current context, which is the operational content of "predictor
attribution".

    state   = cat(z_self, z_world)            [detached]
    memory  = ContextMemory.memory            [detached, n_slots x slot_dim]

    q       = W_q state                       [key_dim]
    k_i     = W_k memory_i                    [key_dim]
    a       = softmax( q . k_i / (sqrt(key_dim) * tau) )      <- ATTRIBUTION
    read    = sum_i a_i memory_i
    z_hat   = W_p read
    loss    = MSE(z_hat, z_harm_a) / Var_ema(z_harm_a)
                                 + entropy_weight * H_norm(a)

The division by the running target variance is load-bearing: z_harm_a's
magnitude is regime-dependent, and a raw MSE against a small-magnitude
target is numerically dominated by the entropy term (measured on this
substrate: pred_loss 1.9e-4 vs entropy_weight 0.02 -- a ~100x imbalance
that left attribution exactly uniform). See `target_var_ema_alpha`.

`a` is the attribution vector handed to BLAAnalog.tick() as
`candidate_code_contributions`; the PE gate and the Moita-2004
remap_code_fraction top-k selection downstream of it are UNCHANGED. This
module replaces only the thing that was never learned.

Four design decisions that are load-bearing (do not "simplify" these)
---------------------------------------------------------------------
1. **The predictor sees ONLY the attended read, never the raw state.**
   Concatenating `state` into the predictor input is the standard
   attention-shortcut failure: the head would predict harm directly from
   the state, the attention would carry no gradient signal, and `a` would
   drift back to uninformative -- reproducing the exact defect this module
   exists to fix, but with a trainable veneer on it.

2. **`W_q`, `W_k` and `W_p` all have `bias=False`.** This is SD-016 Part A,
   measured in this codebase on this very memory bank: with slot init scale
   0.01, a default-init Linear bias DOMINATES `W_k @ memory_i` for every
   slot (bias_over_content_ratio 9.88 pre-train, 3.41 post-P0), collapsing
   all keys onto ~b_k and the softmax to uniform attention. Uniform
   attention is precisely the C1-null the head must escape. See
   `ree_core/predictors/e1_deep.py` ContextMemory.key_proj and
   REE_assembly/docs/architecture/context_memory_writepath_fix.md. Same
   reasoning applies to the predictor: with a bias it can emit the mean
   harm vector without consulting the read at all.

2c. **Scoring is COSINE (`q` and `k` L2-normalised before the dot product),
   not a raw scaled dot product.** Same pathology as (2), reached by a
   different route: ContextMemory initialises its slots at scale 0.01
   (`torch.randn(num_slots, memory_dim) * 0.01`), so `W_k m_i` lands around
   1e-3, the logits land around 1e-4, and softmax returns EXACTLY uniform
   attention -- measured on this substrate 2026-08-09 with a scaled-dot-
   product head: normalised entropy 1.000, max weight 0.0625 = 1/16 to four
   decimals, mass excess 0.0000, unmoved after 259 training steps. Note
   this is not slow learning that more steps would fix: the gradient into
   `W_q`/`W_k` is itself scaled by those tiny activations. Cosine scoring
   makes the logit range depend on the ANGLE between query and slot
   content and not on slot magnitude, which is also the semantically right
   choice -- a slot should not out-compete its neighbours for attribution
   merely by having a larger norm. `sqrt(key_dim)` scaling is dropped as
   redundant once the vectors are unit-norm; the learned temperature is
   what sets the sharpness.

2b. **The predictor `W_p` is LINEAR, and that is what makes `a` an
   attribution rather than a generic gate.** Because `W_p` is linear it
   commutes with the weighted sum:

       z_hat = W_p (sum_i a_i m_i) = sum_i a_i (W_p m_i)

   so the head is exactly an attribution-weighted MIXTURE OF PER-SLOT HARM
   PREDICTIONS: `W_p m_i` is slot i's own prediction of the affective
   stream, and `a_i` is how much that slot's prediction is trusted in this
   context. A slot scores high precisely when its stored content maps onto
   the observed harm. Replacing `W_p` with an MLP would destroy this
   reading -- the weights would become an opaque gate over an entangled
   read, and `a_i` would no longer be interpretable as slot i's
   contribution. Do not "upgrade" the predictor.

   Honest limit, stated so nobody over-reads the metric: this is a LEARNED
   ASSOCIATIVE attribution, not a causal-contribution measure. `a_i` is
   identified only up to the equivalence class of slot subsets that support
   the same prediction -- when two slots are equally decodable into the same
   target, which one is selected is arbitrary. That is a weaker claim than
   "code i causally drives the harm PE", and MECH-074d's C1/C2 instrument
   measures the weaker property (mass concentration + context
   differentiation), which is what Moita 2004's dissociation actually
   operationalises. A causal variant (per-slot occlusion: re-predict with
   slot i's contribution removed and score by the loss increase) is the
   natural next pass if C1/C2 pass here but the selectivity turns out not
   to transfer.

3. **The MSE and entropy terms are in DELIBERATE tension, and that tension
   is what earns C2.** The entropy penalty pushes `a` toward peaked
   (C1: mass excess over chance). Peakedness alone is satisfiable by a
   context-blind constant top-k -- which is exactly V3-EXQ-894a's seed 43
   signature (jaccard_gap exactly 0.0 at every threshold: the same ~5 slots
   regardless of context). What rules that out is the MSE term: a fixed
   slot set can only predict a constant, so it pays full variance cost
   whenever the harm target differs by context. Keep `entropy_weight`
   small; a large value buys C1 at C2's expense and reproduces seed 43.

4. **Every input is detached and the head owns its own optimiser.** No
   gradient reaches E1, ContextMemory, or the latent stack, so enabling the
   head cannot perturb representation learning (the joint-training head-
   collapse failure mode behind the standing P0/P1/P2 phased-training rule,
   EXQ-166b/c/d). The head is a read-only consumer of the substrate that
   happens to learn.

Backward compatibility
----------------------
Inert unless `REEConfig.bla_attribution_head == "trainable"` (default
`"contribution_threshold"`, the legacy fixed rule). With the default, this
module is never instantiated and no parameter, optimiser or tensor is
created.

MECH-094: the head never trains or attributes on simulation/replay ticks --
`BLAAnalog.tick(simulation_mode=True)` already returns a zeroed output and
the agent skips the head on those ticks. Learned weights are waking-stream
only.

Falsification signature (inherited from MECH-074d, unchanged): if remap
still perturbs untagged codes uniformly under a trained head -- i.e. C1
attribution_mass_excess stays at chance -- the attribution mechanism is
wrong at a level deeper than trainability.

See CLAUDE.md: SD-035. Spec:
REE_assembly/docs/architecture/sd_035_amygdala_analog.md
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math

import torch
import torch.nn as nn


@dataclass
class BLAAttributionHeadConfig:
    """Configuration for the MECH-074d trainable attribution head.

    Defaults are chosen for the REE scale (16-32 ContextMemory slots,
    memory_dim 128, a few thousand training ticks per experiment phase),
    not for ImageNet-scale attention.
    """

    # Query/key projection width. Deliberately small -- this is a scoring
    # function over at most a few dozen slots, not a transformer.
    key_dim: int = 16

    # Adam learning rate for the head's own optimiser.
    learning_rate: float = 1e-3

    # Weight decay (AdamW-style when > 0; 0.0 uses plain Adam).
    weight_decay: float = 0.0

    # Softmax temperature. < 1.0 sharpens. Initialised here and, when
    # learn_temperature is True, adapted by the same optimiser (stored as
    # log-temperature so it stays positive).
    temperature_init: float = 0.5
    learn_temperature: bool = True

    # Lower clamp on the temperature, applied every forward pass. Without
    # it the entropy term can drive tau -> 0 and produce a hard argmax
    # whose gradient vanishes (a saturated-softmax dead head).
    temperature_min: float = 0.05

    # Weight on the normalised-entropy penalty. Small on purpose: see
    # design decision (3) in the module docstring. 0.0 disables the
    # sparsity pressure entirely (pure prediction-driven attribution).
    entropy_weight: float = 0.02

    # EMA rate for the running target mean/variance used to NORMALISE the
    # prediction loss. Load-bearing, not cosmetic: z_harm_a's magnitude is
    # regime-dependent and can be tiny, and a raw MSE against a tiny target
    # is numerically dwarfed by the entropy term -- measured on this
    # substrate 2026-08-09, pred_loss 1.9e-4 against entropy_weight 0.02,
    # i.e. the sparsity penalty outweighed the actual objective ~100x and
    # attribution stayed exactly uniform (normalised entropy 1.0). Dividing
    # by the running target variance makes the prediction term
    # scale-invariant (~1.0 when uninformative, -> 0 when perfect), so
    # entropy_weight means the same thing across harm regimes.
    target_var_ema_alpha: float = 0.02

    # Floor on the running target variance, so a near-constant target
    # cannot blow the normalised loss up to a huge value.
    target_var_floor: float = 1e-4

    # Gradient-norm clip on the head's parameters. The read-then-predict
    # path multiplies attention by raw slot content, so an outlier slot
    # magnitude can produce a large step; clipping is cheap insurance.
    grad_clip: float = 1.0

    # Number of training steps before `attribute()` returns anything. Below
    # this the head is randomly initialised and its output would be noise
    # driving real ContextMemory writes; the caller falls back to the
    # legacy contribution proxy until the head has seen some experience.
    # This is also what makes the head's advantage OBSERVABLE: pre-warmup
    # behaviour is the legacy rule by construction.
    warmup_steps: int = 200

    # Master training switch. Set False for a frozen-head evaluation phase
    # (P2) after training on the same weights.
    train: bool = True

    # Candidate floor. Attribution weights at or below this are dropped
    # from the candidate dict entirely, so the downstream Moita-2004 top-k
    # is taken over genuinely-scored codes rather than the whole bank.
    # 0.0 keeps every slot as a candidate (the legacy proxy's behaviour).
    min_candidate_weight: float = 0.0


class _AttributionScorer(nn.Module):
    """Query/key attribution scorer plus the harm predictor it is trained by.

    Separated from the stateful wrapper so the wrapper can construct it
    lazily, once the ContextMemory and latent dimensions are known from the
    first real tick rather than guessed from config.
    """

    def __init__(
        self,
        state_dim: int,
        slot_dim: int,
        target_dim: int,
        key_dim: int,
        temperature_init: float,
        learn_temperature: bool,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.slot_dim = int(slot_dim)
        self.target_dim = int(target_dim)
        self.key_dim = int(key_dim)

        # bias=False on all three -- SD-016 Part A, see module docstring
        # design decision (2).
        self.query_proj = nn.Linear(self.state_dim, self.key_dim, bias=False)
        self.key_proj = nn.Linear(self.slot_dim, self.key_dim, bias=False)
        self.predictor = nn.Linear(self.slot_dim, self.target_dim, bias=False)

        log_t = math.log(max(1e-3, float(temperature_init)))
        self.log_temperature = nn.Parameter(
            torch.tensor(float(log_t), dtype=torch.float32),
            requires_grad=bool(learn_temperature),
        )

    def forward(
        self,
        state: torch.Tensor,
        memory: torch.Tensor,
        temperature_min: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (attribution weights [B, n_slots], harm prediction [B, target_dim])."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        q = self.query_proj(state)                       # [B, key_dim]
        k = self.key_proj(memory)                        # [n_slots, key_dim]
        # COSINE scoring, not a raw dot product -- see design decision (2c).
        q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        k = k / k.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        tau = torch.exp(self.log_temperature).clamp(min=float(temperature_min))
        scores = torch.matmul(q, k.t()) / tau            # [B, n_slots], in [-1/tau, 1/tau]
        attn = torch.softmax(scores, dim=-1)
        read = torch.matmul(attn, memory)                # [B, slot_dim]
        pred = self.predictor(read)                      # [B, target_dim]
        return attn, pred


class BLAAttributionHead:
    """Stateful wrapper: lazy construction, own optimiser, diagnostics.

    Mirrors the TrainableEscapeAffordanceLearner pattern already used in
    ree_core/pfc -- a plain class holding an nn.Module, so the head is not
    swept into the agent's parameter list and cannot be co-trained by an
    outer optimiser by accident.

    Learned weights persist across `reset()` (which is per-episode); only
    the diagnostic latches are cleared.
    """

    def __init__(self, config: Optional[BLAAttributionHeadConfig] = None) -> None:
        self.config = config or BLAAttributionHeadConfig()

        self._model: Optional[_AttributionScorer] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None

        self._n_updates: int = 0
        self._n_attribute_calls: int = 0
        self._n_warmup_fallbacks: int = 0
        # Running target statistics for the scale-invariant prediction loss.
        # Persist across reset() alongside the weights -- they describe the
        # harm regime, not one episode.
        self._target_mean: Optional[torch.Tensor] = None
        self._target_var: float = float(self.config.target_var_floor)
        self._last_loss: float = 0.0
        self._last_pred_loss: float = 0.0
        self._last_entropy: float = 0.0
        self._last_max_weight: float = 0.0

    # -- State management --

    def reset(self) -> None:
        """Per-episode reset. Learned weights are deliberately PRESERVED.

        The whole point of the head is that attribution improves with
        experience (Moita 2004's dissociation develops over training), so
        wiping weights each episode would reproduce the non-trainable
        rule's defect with extra steps.
        """
        self._last_loss = 0.0
        self._last_pred_loss = 0.0
        self._last_entropy = 0.0
        self._last_max_weight = 0.0

    @property
    def is_warm(self) -> bool:
        """True once the head has trained past `warmup_steps`."""
        return self._n_updates >= int(self.config.warmup_steps)

    # -- Lazy construction --

    def _ensure_model(
        self,
        state_dim: int,
        slot_dim: int,
        target_dim: int,
    ) -> _AttributionScorer:
        if self._model is not None:
            return self._model
        self._model = _AttributionScorer(
            state_dim=state_dim,
            slot_dim=slot_dim,
            target_dim=target_dim,
            key_dim=int(self.config.key_dim),
            temperature_init=float(self.config.temperature_init),
            learn_temperature=bool(self.config.learn_temperature),
        )
        params = [p for p in self._model.parameters() if p.requires_grad]
        wd = float(self.config.weight_decay)
        if wd > 0.0:
            self._optimizer = torch.optim.AdamW(
                params, lr=float(self.config.learning_rate), weight_decay=wd
            )
        else:
            self._optimizer = torch.optim.Adam(
                params, lr=float(self.config.learning_rate)
            )
        return self._model

    # -- Inference: the attribution the BLA gate consumes --

    def attribute(
        self,
        state: torch.Tensor,
        memory: torch.Tensor,
    ) -> Optional[Dict[int, float]]:
        """Return {slot_index: attribution_weight}, or None while warming up.

        None is a deliberate signal to the caller to fall back to the
        legacy contribution proxy -- a randomly-initialised head would
        otherwise drive real ContextMemory remap writes with noise.

        Never trains, never mutates memory, no gradient.
        """
        self._n_attribute_calls += 1
        if not self.is_warm:
            self._n_warmup_fallbacks += 1
            return None
        if self._model is None:
            self._n_warmup_fallbacks += 1
            return None

        with torch.no_grad():
            st = state.detach()
            if st.dim() == 1:
                st = st.unsqueeze(0)
            attn, _ = self._model(
                st, memory.detach(), float(self.config.temperature_min)
            )
            weights = attn.mean(dim=0)

        floor = float(self.config.min_candidate_weight)
        out: Dict[int, float] = {}
        max_w = 0.0
        for idx in range(weights.numel()):
            w = float(weights[idx].item())
            if w > max_w:
                max_w = w
            if w > floor:
                out[int(idx)] = w
        self._last_max_weight = max_w
        if not out:
            # Every weight at or below the floor -- degenerate; hand back
            # None so the gate stays shut rather than firing on an empty
            # candidate set.
            return None
        return out

    # -- Training --

    def train_step(
        self,
        state: torch.Tensor,
        memory: torch.Tensor,
        harm_target: torch.Tensor,
        simulation_mode: bool = False,
    ) -> Optional[float]:
        """One supervised step: predict `harm_target` from the attended read.

        Args:
            state: cat(z_self, z_world) for this tick. Detached internally.
            memory: ContextMemory slot bank [n_slots, slot_dim]. Detached
                internally -- no gradient reaches E1.
            harm_target: observed z_harm_a for this tick [z_harm_a_dim] or
                [1, z_harm_a_dim]. Detached internally.
            simulation_mode: MECH-094 gate. True -> no-op, returns None.

        Returns the scalar loss, or None if the step was skipped.
        """
        if simulation_mode or not bool(self.config.train):
            return None

        st = state.detach()
        if st.dim() == 1:
            st = st.unsqueeze(0)
        mem = memory.detach()
        tgt = harm_target.detach()
        if tgt.dim() == 1:
            tgt = tgt.unsqueeze(0)
        if st.shape[0] != tgt.shape[0]:
            # Broadcast a single state/target pair; anything else is a
            # caller error and is skipped rather than silently averaged.
            if st.shape[0] == 1:
                st = st.expand(tgt.shape[0], -1)
            elif tgt.shape[0] == 1:
                tgt = tgt.expand(st.shape[0], -1)
            else:
                return None

        model = self._ensure_model(
            state_dim=int(st.shape[-1]),
            slot_dim=int(mem.shape[-1]),
            target_dim=int(tgt.shape[-1]),
        )
        if self._optimizer is None:
            return None

        # Running target mean/variance -> scale-invariant prediction loss.
        # Updated BEFORE the loss so a first-ever step still has a sane
        # divisor; statistics are pure bookkeeping and carry no gradient.
        with torch.no_grad():
            tgt_mean_now = tgt.mean(dim=0)
            a_t = float(self.config.target_var_ema_alpha)
            if self._target_mean is None:
                self._target_mean = tgt_mean_now.clone()
            else:
                self._target_mean = (
                    (1.0 - a_t) * self._target_mean + a_t * tgt_mean_now
                )
            dev = float(((tgt - self._target_mean) ** 2).mean().item())
            self._target_var = (1.0 - a_t) * self._target_var + a_t * dev
        var_div = max(float(self.config.target_var_floor), float(self._target_var))

        attn, pred = model(st, mem, float(self.config.temperature_min))
        pred_loss = torch.nn.functional.mse_loss(pred, tgt) / var_div

        n_slots = int(attn.shape[-1])
        if n_slots > 1:
            log_n = math.log(float(n_slots))
            p = attn.clamp(min=1e-12)
            norm_entropy = (-(p * torch.log(p)).sum(dim=-1) / log_n).mean()
        else:
            norm_entropy = torch.zeros((), dtype=attn.dtype)

        ew = float(self.config.entropy_weight)
        loss = pred_loss + ew * norm_entropy

        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip = float(self.config.grad_clip)
        if clip > 0.0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], clip
            )
        self._optimizer.step()

        self._n_updates += 1
        self._last_loss = float(loss.detach().item())
        self._last_pred_loss = float(pred_loss.detach().item())
        self._last_entropy = float(norm_entropy.detach().item())
        return self._last_loss

    # -- Read-only accessors --

    @property
    def temperature(self) -> float:
        if self._model is None:
            return float(self.config.temperature_init)
        return float(
            torch.exp(self._model.log_temperature.detach())
            .clamp(min=float(self.config.temperature_min))
            .item()
        )

    @property
    def diagnostics(self) -> Dict[str, float]:
        return {
            "n_updates": int(self._n_updates),
            "n_attribute_calls": int(self._n_attribute_calls),
            "n_warmup_fallbacks": int(self._n_warmup_fallbacks),
            "is_warm": bool(self.is_warm),
            "last_loss": float(self._last_loss),
            "last_pred_loss": float(self._last_pred_loss),
            "last_norm_entropy": float(self._last_entropy),
            "target_var_ema": float(self._target_var),
            "last_max_weight": float(self._last_max_weight),
            "temperature": float(self.temperature),
        }
