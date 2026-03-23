"""
MECH-095: TPJ Agency Comparator — V3 Implementation

Biological basis: temporoparietal junction (TPJ) acts as a self/other boundary
comparator by comparing efference-copy predictions against observed reafference.

Mechanism:
  1. Before action: E2 generates efference-copy prediction
         z_self_pred = E2.predict_next_self(z_self_t, a_t)
  2. After action: observe actual z_self
         z_self_obs = LatentStack.encode(new_obs).z_self
  3. Compute mismatch:
         agency_mismatch = ||z_self_pred - z_self_obs||
  4. Agency signal:
         agency_signal = 1 / (1 + agency_mismatch)  -- high when self-caused
         attribution = "self" if agency_signal > threshold else "world"

Interpretation:
  - Low mismatch (agency_signal near 1): z_self changed as predicted by E2 ->
    self-caused change (motor-predictable).
  - High mismatch (agency_signal near 0): z_self changed unexpectedly ->
    world-caused change (motor-unpredictable) or action execution error.

Failure in schizophrenia (Frith et al. 2000):
  The comparator is non-functional -> self-generated actions appear world-caused.
  Passivity experiences (thought insertion, alien control) = attribution of
  self-generated movements to external agency. In REE without MECH-095: z_world_delta
  from self-caused actions is attributed to the environment, contaminating SD-003.

Self/world split (SD-005) is necessary but not sufficient:
  The split creates separate representations, but without the comparator, the
  LABELLING of changes (self-caused vs world-caused) is absent. The comparator
  provides the labelling signal.

Training risk ("schizophrenic drift"):
  If harm training uses z_world features correlated with z_self features, backprop
  can gradually blur the split boundary. The agency_signal output can be used as a
  regularisation signal to maintain the boundary:
    - self-caused events: z_self encoder should predict z_self_pred well (low loss)
    - world-caused events: z_self encoder should NOT be informative about world changes
  See AdversarialSplitHead in stack.py for the gradient reversal defense.

References:
  - Blakemore, Wolpert & Frith (2002). Abnormalities in the awareness of action.
    Trends in Cognitive Sciences.
  - Frith et al. (2000). Explaining the symptoms of schizophrenia: abnormalities
    in the awareness of action.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class TPJComparator(nn.Module):
    """
    MECH-095: Temporoparietal Junction agency comparator.

    Compares E2 efference-copy prediction against observed z_self.

    Usage in agent.py sense() loop:
        # Before action execution (store prediction):
        z_self_pred = e2.predict_next_self(z_self_current, action_taken)

        # After action + encode (observe actual):
        z_self_obs = new_latent_state.z_self

        # Compute agency signal:
        agency_signal, attribution = tpj.compare(z_self_pred, z_self_obs)

        # Feed to residue field and E3:
        residue.set_agency_context(attribution)
    """

    def __init__(
        self,
        self_dim: int = 32,
        agency_threshold: float = 0.5,
    ):
        """
        Args:
            self_dim:          dimensionality of z_self
            agency_threshold:  agency_signal threshold above which event is
                               classified as self-caused (default 0.5)
        """
        super().__init__()
        self.self_dim = self_dim
        self.agency_threshold = agency_threshold

    def compare(
        self,
        z_self_pred: torch.Tensor,
        z_self_obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compare efference-copy prediction against observed z_self.

        Args:
            z_self_pred: [batch, self_dim] -- E2's predicted z_self (efference copy)
            z_self_obs:  [batch, self_dim] -- observed z_self after action

        Returns:
            agency_signal: [batch] -- scalar in (0, 1); high = self-caused
            is_self_caused: [batch] bool -- True if agency_signal > threshold
        """
        # L2 mismatch between prediction and observation
        mismatch = (z_self_pred - z_self_obs).norm(dim=-1)          # [batch]

        # Map mismatch to agency signal: high when prediction was accurate (self-caused)
        agency_signal = 1.0 / (1.0 + mismatch)                      # [batch], in (0, 1)

        is_self_caused = agency_signal > self.agency_threshold       # [batch] bool

        return agency_signal, is_self_caused

    def compute_agency_loss(
        self,
        z_self_pred: torch.Tensor,
        z_self_obs: torch.Tensor,
        true_is_self_caused: torch.Tensor,
    ) -> torch.Tensor:
        """
        Training loss: encourages accurate agency attribution.

        Penalises high agency signal on world-caused events (where E2's prediction
        should NOT match -- the world changed the agent, not the agent's own action).

        Args:
            z_self_pred:          [batch, self_dim]
            z_self_obs:           [batch, self_dim]
            true_is_self_caused:  [batch] bool -- ground truth (from environment)

        Returns:
            scalar loss
        """
        agency_signal, _ = self.compare(z_self_pred, z_self_obs)

        # Binary cross entropy: predict 1 for self-caused, 0 for world-caused
        targets = true_is_self_caused.float()
        loss = nn.functional.binary_cross_entropy(
            agency_signal.clamp(1e-6, 1 - 1e-6),
            targets,
        )
        return loss

    def forward(
        self,
        z_self_pred: torch.Tensor,
        z_self_obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.compare(z_self_pred, z_self_obs)
