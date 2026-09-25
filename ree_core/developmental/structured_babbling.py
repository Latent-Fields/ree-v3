"""Structured babbling: a class-balanced, persistent-run motor generator (W2a).

Why this exists
---------------
The 2026-09-25 babbling probe (REE_assembly
``evidence/planning/babbling_e2_action_coverage_probe_20260925.md``, 0ac69c87446) found
that "Phase 0 is random-policy stepping" is false as a description of the generator the
curriculum actually runs: Phase 0 takes the agent's OWN native E3 selection
(``act_with_split_obs``, ``argmax % 4``), so the class distribution is whatever the
untrained policy prefers and the stay class (4) is never emitted (premise 1). The dose that
gave the E2 world head real action coverage was a REDESIGNED generator -- the L2 form: draw
a class uniformly, hold it for a run length drawn uniformly from {1..4}, repeat. The
coupled-loop-repair campaign (``evidence/planning/coupled_loop_repair_campaign_plan.md``
section 3, W2a; synthesis 7b/7d) adopts that form as a ree_core developmental-stage source,
widened from the probe's {0..3} to ALL env action classes (the 5-class env includes stay =
4).

What it is (and is not)
-----------------------
* A pure generator. ``next_class()`` returns the next action class; ``next_action()``
  returns it one-hot, shape ``[1, n_classes]``, the executed-action form every forward
  model in ree_core consumes.
* Its randomness comes ONLY from its own ``numpy.random.Generator`` seeded at construction.
  It never reads or advances the global torch / numpy / python RNG, so constructing or
  drawing from it cannot shift any act-path draw.
* It is NOT wired into any policy, scheduler or driver. ``REEAgent`` constructs it only
  when ``structured_babbling_enabled`` is True (default False: never constructed, nothing
  imported) and nothing in ree_core calls it. Its consumer (the E2 world-head member's
  babbling replay, campaign W3) is branch work.

Evidence domain: none by itself. It produces an action stream; that the stream improves any
learner is the W3 L2R gate's question, not this module's.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch


class StructuredBabbler:
    """Uniform class draw, held for a uniform run length in {1..max_run}, repeated."""

    def __init__(self, n_classes: int, max_run: int = 4, seed: int = 0) -> None:
        if int(n_classes) < 1:
            raise ValueError("StructuredBabbler needs n_classes >= 1, got %r" % (n_classes,))
        if int(max_run) < 1:
            raise ValueError("StructuredBabbler needs max_run >= 1, got %r" % (max_run,))
        self.n_classes = int(n_classes)
        self.max_run = int(max_run)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._current: Optional[int] = None
        self._remaining = 0
        self.n_emitted = 0
        self.n_runs = 0
        self.last_run_length = 0   # drawn length of the run in progress (telemetry)

    @classmethod
    def from_config(cls, config: Any) -> "StructuredBabbler":
        """Build from ``REEConfig``. ``structured_babbling_n_classes`` 0 -> the env action
        count the config was built for (``config.e2.action_dim``, which ``from_dims`` sets
        from its ``action_dim`` argument)."""
        n = int(getattr(config, "structured_babbling_n_classes", 0))
        if n <= 0:
            n = int(config.e2.action_dim)
        return cls(n_classes=n,
                   max_run=int(getattr(config, "structured_babbling_max_run", 4)),
                   seed=int(getattr(config, "structured_babbling_seed", 0)))

    def next_class(self) -> int:
        """The next action class. Starts a new run when the current one is exhausted."""
        if self._remaining <= 0:
            self._current = int(self._rng.integers(0, self.n_classes))
            self._remaining = int(self._rng.integers(1, self.max_run + 1))
            self.last_run_length = self._remaining
            self.n_runs += 1
        self._remaining -= 1
        self.n_emitted += 1
        assert self._current is not None
        return self._current

    def next_action(self, device: Any = None) -> torch.Tensor:
        """The next action as a one-hot ``[1, n_classes]`` float tensor (no RNG drawn)."""
        out = torch.zeros(1, self.n_classes, device=device)
        out[0, self.next_class()] = 1.0
        return out

    def reset(self) -> None:
        """Abandon the current run (episode boundary). The generator stream continues."""
        self._current = None
        self._remaining = 0
