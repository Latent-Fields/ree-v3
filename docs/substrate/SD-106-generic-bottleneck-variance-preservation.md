## SD-106 Generic Bottleneck Variance Preservation (2026-09-11)

- SD-106: `encoder.generic_bottleneck_variance_preservation` -- IMPLEMENTED 2026-09-11.
  `ree_core/latent/zworld_p0.py` (objective) and `ree_core/latent/stack.py::SplitEncoder`
  (encoder bypass).
  Config: `ZWorldP0Config.preservation_weight` (default `0.0`; set `200.0` to enable) and
  `LatentStackConfig.use_world_encoder_skip` (default `False`; set `True` to enable).
  Data flow: `world_obs` -> `world_encoder` (+ zero-init linear bypass) -> `x sigmoid(world_precision_logit)`
  -> `z_world` -> [P0 training only] `_preserve_head` -> normalised preservation loss
  `MSE(recon, world_obs) / mean_var(world_obs)`.
  Backward compatible: disabled by default; existing experiments unaffected. Verified
  bit-identical OFF -- `state_dict` keys unchanged (the bypass module is not constructed, so
  existing checkpoints still load), forward pass `torch.equal`, and the P0 `final_loss`
  identical to full precision under RNG-controlled repetition.
  Phased training required: yes -- this is P0 only, unchanged from SD-070. P1 must train on
  stop-gradient `z_world` with the encoder optimiser NOT stepped (EXQ-166b/c/d).
  MECH-094: not applicable -- trains an encoder on live observations and writes nothing to
  memory during any non-waking state.
  Validation experiment: V3-EXQ-1015 queued (re-runs
  `experiments/v3_exq_1010_zworld_overcapacity_decoder_sweep.py` UNCHANGED against an
  SD-106-ON warmup).
  See SD-018 (the superseded single-feature shape), SD-070 (the P0 recipe this extends),
  SD-015, ARC-030, MECH-117, MECH-457, ARC-065.

### Why this shape, and not SD-018's

SD-018 supervises ONE NAMED FEATURE (resource proximity); V3-EXQ-978 returned it NULL. This
entry is the successor SHAPE minted by `/governance` gov-20260911 on user decision, from the
CONFIRMED autopsy `failure_autopsy_V3-EXQ-1010_2026-09-11.json` (REE_assembly `652ababa92`).
What the evidence names is GENERIC variance/reconstruction preservation at the
observation->z_world bottleneck, not another named feature.

### The measurement that located the defect

V3-EXQ-1010 swept decoder capacity over ~77,000x in parameters against the frozen latent
SD-070's P0 produces: 0/3 seeds clear the 0.80 held-out oracle-action agreement bar (best
0.6839 / 0.6846 / 0.6656) while the SAME ladder memorises the training split at 0.9996-0.9998,
a PCA-32 of the encoder's own 250-dim input at the encoder's own 32-dim width clears it 3/3
(0.8836 / 0.8729 / 0.8763), and the SAME architecture at RANDOM INITIALISATION scores at or
ABOVE the trained latent on 3/3. Not width, not consumer capacity, not consumer learning
(H-B eliminated at V3-EXQ-1002).

`ZWorldP0Config()` ALREADY carried `reconstruction_weight=10.0`, so "add reconstruction" was
not the missing piece. Measured on the x724 rung after a full run of the shipped recipe, the
trained-state loss decomposes as:

| term | weight | raw | contribution | share |
|---|---|---|---|---|
| variance hinge | 25 | 0.712 | 17.79 | 80.3% |
| covariance | 50 | 0.041 | 2.07 | 9.3% |
| 4 grounding CE heads | 1 each | -- | 2.11 | 9.5% |
| reconstruction | 10 | 0.0177 | **0.177** | **0.80%** |

VICReg anti-collapse is 90% of the objective; reconstruction is 0.8% of it. The leg is not
under-weighted -- it is on the WRONG SCALE. Raw MSE against a sparse one-hot-dominated
observation is ~0.018, where the hinge and CE terms it competes with are O(1) by construction.
The consequence, measured as linear-decodable `world_obs` content:

| latent | recon R^2 | input variance discarded |
|---|---|---|
| untrained encoder | 0.9029 | 9.7% |
| trained (shipped recipe) | 0.9652 | 3.5% |
| PCA-32 (the acceptance anchor) | 0.9984 | 0.16% |

The trained latent discards ~22x more input variance than PCA-32. PCA-32 explains 99.84% of
`world_obs`, so a 32-dim code CAN be near-lossless here; the objective simply never asks for it.

### The two additions

1. **Scale-normalised preservation term** (`preservation_weight`). Divides the reconstruction
   MSE by the TRAIN SPLIT's own mean per-element variance, making the term fraction-of-variance-
   unexplained (`1 - R^2`) -- O(1) and dataset-scale-free, so a weight means what it appears to
   mean. Added ALONGSIDE `reconstruction_weight` with its own linear decoder rather than by
   re-weighting the existing leg: a separate default-0.0 term cannot silently change any
   existing P0 run, and a separate module keeps the OFF path provably untouched. The decoder is
   LINEAR on purpose -- with a linear decoder the MSE optimum IS the principal subspace
   (Baldi & Hornik 1989), which is the anchor the acceptance target names.

2. **Zero-initialised linear bypass** (`use_world_encoder_skip`). The world encoder is
   `Linear(world_obs_dim, 64) -> ReLU -> Linear(64, world_dim)`, and a ReLU MLP can only
   APPROXIMATE the linear variance-preserving map PCA-32 realises exactly. V3-EXQ-1008's
   consumer-rung decomposition of the 0.1998 PCA-to-trained gap attributes -0.0779 to the
   encoder's nonlinear architecture measured AT RANDOM INIT (random orthonormal-32 0.7725 vs
   untrained encoder net 0.6946) -- an architectural cost, not an objective one. Zero-init
   (ReZero, Bachlechner et al. 2021) means enabling the flag is a no-op at step 0; only training
   moves it off zero. `world_path_parameters()` includes it, so it is actually trained -- and so
   the V3-EXQ-783 weight-delta readiness check still sees the whole world path.

### The bypass consumes NO RNG draws, and that is load-bearing

Found by the flag-inertness probe this landing adds, not by review. Every `nn.Module` built in
`SplitEncoder.__init__` consumes torch RNG draws, so a module inserted into that sequence shifts
the random initialisation of every module built after it. The first draft built the bypass
immediately after `world_encoder`, which meant an ON/OFF arm pair differing only in
`use_world_encoder_skip` ALSO differed in the init of `lateral_head`, `event_classifier`,
`resource_proximity_head`, `resource_field_head` and both topdown projections -- a confound
across the whole encoder rather than a bypass. (The same hazard `x1002._make_agent` documents
for `resource_field_head`.)

Moving it to the end of `__init__` is NOT sufficient: `LatentStack` constructs further modules
after the `SplitEncoder`, so one extra draw still shifts those. The fix is to save and restore
the RNG state around the construction -- free, because `nn.Linear`'s random weight is
immediately overwritten with zeros, so those draws cannot affect any value. The flag is
therefore EXACTLY inert rather than approximately so, and the ON and OFF arms of V3-EXQ-1023
are identical at initialisation except for the (zero) bypass.

`tests/test_flag_inertness.py::test_use_world_encoder_skip_is_bit_identical_off_and_live_on`
pins all three properties: OFF grows no `state_dict` key (existing checkpoints still load); ON
is `torch.equal` on `z_world` at step 0; and perturbing the bypass weight DOES move `z_world`,
so a module that were wired but never summed cannot pass the zero-init check vacuously.

### Measured effect, and the cost it carries

2 seeds, linear-decodable `world_obs` R^2 / `resource_field_view` R^2 (the 25-dim slice the
1010 autopsy reports decodes the oracle at 0.9735-0.9832 from raw, hence the proxy):

| recipe | obs R^2 (s42/s43) | resource-field R^2 (s42/s43) |
|---|---|---|
| PCA-32 **anchor** | 0.9983 / 0.9984 | **0.9878 / 0.9894** |
| shipped | 0.9432 / 0.9493 | 0.9264 / 0.9479 |
| preservation 50 + skip | 0.9962 / 0.9962 | 0.9813 / 0.9841 |
| **preservation 200 + skip** | **0.9974 / 0.9978** | **0.9857 / 0.9908** |

SD-070's anti-collapse gate survives: participation ratio 14.68 -> 13.86 against a >= 2.0 gate.

**THE HONEST COST, stated rather than buried: the grounding heads lose discriminativeness.**
At a realistic P0 step count (600 steps, 4000 buffered observations) mean held-out grounding
lift falls 0.5702 (shipped) -> 0.4313 (w=50) -> 0.4064 (w=200) -> 0.3982 (w=500). Lift stays
strongly positive so SD-070's gate holds, but roughly a quarter of the grounding signal is
traded for the preservation gain, and the resource-field gain above w=50 is small (0.9826 ->
0.9853 -> 0.9856). If a downstream consumer turns out to depend on grounding
discriminativeness rather than on preserved variance, w=50 is the better operating point and
this is the number to revisit.

**SD-106 PROMOTES NOTHING on its own.** The R^2 figures above are a PROXY. The acceptance
target is `>= 0.85` held-out oracle-action agreement at the consumer rung
(`x734.PPOPolicyNet` at `PPO_TRUNK_HIDDEN`) on a seed majority, measured by re-running
`experiments/v3_exq_1010_zworld_overcapacity_decoder_sweep.py` UNCHANGED -- the harness,
dataset recipe, calibration anchor and negative control all already exist.
