"""
Amygdala analogue substrate for REE-v3 (SD-035 cluster).

Two peer non-trainable arithmetic modules, mirroring the biological
division between basolateral (BLA) and central (CeA) amygdala. Both
consume z_harm_a (produced by AffectiveHarmEncoder in
ree_core/latent/stack.py, SD-011) and write to different downstream
consumers:

  - BLAAnalog (ree_core/amygdala/bla.py, SD-035 + MECH-074a/b/d):
    writes encoding_gain (multiplier on HippocampalModule write
    strength under arousal; Roozendaal 2011 inverted-U), retrieval_bias
    (per-trace weight vector, NOT a scalar; LaBar & Cabeza 2006), and
    remap_signal (per-code binary with predictor attribution; Moita 2004
    contextual/auditory dissociation requires the attribution gate).
  - CeAAnalog (ree_core/amygdala/cea.py, SD-035 + MECH-046 + MECH-074c):
    writes mode_prior (scalar, pre-softmax additive log-odds routed via
    SalienceCoordinator.update_signal("cea_mode_prior", ...); Mendez-
    Bertolo 2016 fast latency) and fast_prime (scalar candidate-prior
    pulse distinct from mode_prior; Pessoa & Adolphs 2010 override
    window). Also accepts an escapability_hint placeholder so MECH-219 /
    Q-036 can wire in without a module-interface refactor.

Non-trainable: pure arithmetic. No gradient flow. Reset per episode.

  - BLAAttributionHead (ree_core/amygdala/attribution_head.py, SD-035 +
    MECH-074d second pass, 2026-08-09): the ONE trainable component in
    this package. Supplies BLAAnalog's candidate_code_contributions from
    a learned, context-conditioned attribution over ContextMemory slots
    instead of the fixed non-trainable proxy. Off by default
    (REEConfig.bla_attribution_head = "contribution_threshold"); set
    "trainable" to enable. Built after V3-EXQ-894/894a showed the fixed
    rule's context-selectivity does not respond to PE-threshold
    recalibration. Its own optimiser, detached inputs -- no gradient
    reaches E1 or the latent stack, so the "no gradient flow" property
    above still holds for the substrate this package reads.

Master switch: REEConfig.use_amygdala_analog (default False) gates both
modules -- backward-compat requirement. Per-module switches
use_bla_analog / use_cea_analog give granular control.

See docs/architecture/sd_035_amygdala_analog.md.
"""

from ree_core.amygdala.attribution_head import (
    BLAAttributionHead,
    BLAAttributionHeadConfig,
)
from ree_core.amygdala.bla import BLAAnalog, BLAConfig, BLAOutput
from ree_core.amygdala.cea import CeAAnalog, CeAConfig, CeAOutput

__all__ = [
    "BLAAnalog",
    "BLAAttributionHead",
    "BLAAttributionHeadConfig",
    "BLAConfig",
    "BLAOutput",
    "CeAAnalog",
    "CeAConfig",
    "CeAOutput",
]
