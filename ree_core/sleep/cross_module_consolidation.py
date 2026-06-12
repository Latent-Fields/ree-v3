"""
CrossModuleConsolidator: module-tagged interleaved offline consolidation.

MECH-423 R3 readiness instrumentation (MECH-121 consolidation cluster).

The legacy MECH-121 offline pass (SelfModelAggregator.offline_gradient_pass,
ree_core/sleep/phase_manager.py) trains e2_harm_s ALONE over region-keyed replay
traces that carry NO module/stream identity. As a result two readouts the
EXP-0380 (MECH-423 cross-model super-additivity) acceptance check requires do not
exist in the live ree-v3 eval path:

  * cross-module replay share -- the fraction of replayed traces that update more
    than one module (with a single-module pass it is unmeasurable / trivially 0);
  * an E1<->E2 INTERLEAVED consolidation schedule -- the legacy pass is a blocked
    single-module schedule, so the integrated E1+E2 representation the
    super-additivity ablation tests cannot be acquired under interleaving.

This module supplies both. It is pure orchestration: it takes a set of named
modules, each with a loss closure (computing that module's loss over a fresh
replay draw) and its parameter list, runs a configurable schedule, tags each
replayed trace by which modules it actually updated, and reports a flat dict of
pure-arithmetic readouts.

MECH-094: this is the SAME explicit exception the e2_harm_s writeback already
relies on. It is a WEIGHT-update pass -- the optimisers are constructed LOCALLY
over only the named modules' parameters and nothing is written to residue /
anchors / memory. No hypothesis-tagged content is produced. A simulation_mode
guard is provided for symmetry with the other regulators (no-op when True); the
legitimate offline call site passes simulation_mode=False.

Biological grounding (the R3 thresholds derive from these): McClelland,
McNaughton & O'Reilly 1995 + Kumaran, Hassabis & McClelland 2016 -- interleaving
is a NECESSARY condition for integrating a shared representation; a blocked
schedule causes catastrophic interference, which would show up as a sub-additive
ARTEFACT rather than a refutation of super-additivity. The "blocked" schedule is
therefore the pre-registered control arm, not a bug.

Bit-identical OFF: the agent only builds a CrossModuleConsolidator when
use_cross_module_consolidation=True, and the SleepLoopManager hook is skipped
when the consolidator is None, so the default waking + sleep pipelines are
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import torch


VALID_SCHEDULES = ("interleaved", "blocked")


@dataclass
class CrossModuleConsolidatorConfig:
    """Configuration for the cross-module consolidation pass.

    schedule:
        "interleaved" -- each replayed trace runs one gradient step on EVERY
            module that has replay content this step, so a trace can touch >1
            module (the integration regime).
        "blocked"     -- modules are trained sequentially (all steps of module A,
            then all steps of module B); each trace touches exactly one module
            (the catastrophic-interference control; cross_module_replay_share 0).
    n_steps:
        Number of replay steps. For "interleaved" this is the number of traces;
        for "blocked" it is the number of steps PER module.
    lr:
        Learning rate of the locally-constructed per-module optimisers.
    """

    schedule: str = "interleaved"
    n_steps: int = 0
    lr: float = 1e-3

    def __post_init__(self) -> None:
        if self.schedule not in VALID_SCHEDULES:
            raise ValueError(
                "schedule must be one of "
                f"{VALID_SCHEDULES}; got {self.schedule!r}"
            )
        if self.n_steps < 0:
            raise ValueError(f"n_steps must be >= 0; got {self.n_steps}")
        if self.lr <= 0.0:
            raise ValueError(f"lr must be > 0; got {self.lr}")


class CrossModuleConsolidator:
    """Module-tagged interleaved offline consolidation pass (MECH-423 R3)."""

    def __init__(self, config: Optional[CrossModuleConsolidatorConfig] = None) -> None:
        self.config = config or CrossModuleConsolidatorConfig()
        self._last_metrics: Dict[str, float] = {}

    @property
    def last_metrics(self) -> Dict[str, float]:
        return dict(self._last_metrics)

    def consolidate(
        self,
        module_losses: Dict[str, Callable[[], torch.Tensor]],
        module_params: Dict[str, Iterable[torch.nn.Parameter]],
        n_steps: Optional[int] = None,
        schedule: Optional[str] = None,
        lr: Optional[float] = None,
        simulation_mode: bool = False,
    ) -> Dict[str, float]:
        """Run the consolidation pass and return a flat readout dict.

        Args:
            module_losses: name -> callable returning a scalar loss tensor over a
                FRESH replay draw. The callable MUST return an exactly-zero tensor
                (e.g. ``params.sum() * 0.0``) when it has no replay content; such
                a step does NOT count as touching that module (so the cross-module
                share genuinely reflects whether integration happened, not just
                the schedule label). This matches the existing agent loss
                convention (compute_prediction_loss / compute_e2_loss).
            module_params: name -> iterable of parameters for that module's locally
                constructed optimiser (MECH-094 explicit exception: scoped to the
                named module only; no residue / memory writes).
            n_steps / schedule / lr: override the config values for this call.
            simulation_mode: MECH-094 guard -- when True the pass is a no-op and
                returns the zeroed readout (replay/DMN callers must not consolidate
                model weights). The legitimate offline call site passes False.

        Returns:
            Flat dict of pure-arithmetic readouts:
                n_updates                  -- total gradient steps taken
                n_traces                   -- replay traces processed
                n_cross_module_traces      -- traces that updated > 1 module
                cross_module_replay_share  -- n_cross_module_traces / n_traces
                interleaved                -- 1.0 if schedule == "interleaved"
                updates_<name>             -- per-module gradient-step count
        """
        sched = schedule if schedule is not None else self.config.schedule
        steps = int(n_steps if n_steps is not None else self.config.n_steps)
        rate = float(lr if lr is not None else self.config.lr)
        if sched not in VALID_SCHEDULES:
            raise ValueError(
                f"schedule must be one of {VALID_SCHEDULES}; got {sched!r}"
            )

        names = list(module_losses.keys())
        per_module: Dict[str, int] = {name: 0 for name in names}
        n_updates = 0
        n_traces = 0
        n_cross = 0

        if simulation_mode or steps <= 0 or not names:
            return self._finalize(
                per_module, n_updates, n_traces, n_cross, sched
            )

        # MECH-094 explicit exception: per-module optimisers constructed LOCALLY
        # over only the named modules' parameters. Built fresh each call so the
        # pass owns no persistent optimiser state and touches nothing else.
        optimizers: Dict[str, torch.optim.Optimizer] = {}
        for name in names:
            params = [p for p in module_params.get(name, []) if p.requires_grad]
            if params:
                optimizers[name] = torch.optim.Adam(params, lr=rate)

        def _step_module(name: str) -> bool:
            """One gradient step on a module. Returns True iff it was touched."""
            nonlocal n_updates
            loss_fn = module_losses[name]
            opt = optimizers.get(name)
            if opt is None:
                return False
            loss = loss_fn()
            if not torch.is_tensor(loss) or not loss.requires_grad:
                return False
            # Exactly-zero sentinel == "no replay content" -> module not touched.
            if float(loss.detach().reshape(-1)[0].item() if loss.numel() == 1
                     else loss.detach().abs().sum().item()) == 0.0:
                return False
            opt.zero_grad()
            loss.backward()
            opt.step()
            per_module[name] += 1
            n_updates += 1
            return True

        if sched == "interleaved":
            # Each trace runs one step on every module; a trace touching >1
            # module is a cross-module integration event.
            for _ in range(steps):
                touched = 0
                for name in names:
                    if _step_module(name):
                        touched += 1
                n_traces += 1
                if touched > 1:
                    n_cross += 1
        else:  # "blocked"
            # Modules trained sequentially: each trace touches exactly one module.
            for name in names:
                for _ in range(steps):
                    n_traces += 1
                    _step_module(name)  # touched count is always <= 1 -> no cross

        return self._finalize(per_module, n_updates, n_traces, n_cross, sched)

    def _finalize(
        self,
        per_module: Dict[str, int],
        n_updates: int,
        n_traces: int,
        n_cross: int,
        schedule: str,
    ) -> Dict[str, float]:
        share = (float(n_cross) / float(n_traces)) if n_traces > 0 else 0.0
        metrics: Dict[str, float] = {
            "n_updates": float(n_updates),
            "n_traces": float(n_traces),
            "n_cross_module_traces": float(n_cross),
            "cross_module_replay_share": float(share),
            "interleaved": 1.0 if schedule == "interleaved" else 0.0,
        }
        for name, count in per_module.items():
            metrics[f"updates_{name}"] = float(count)
        self._last_metrics = dict(metrics)
        return metrics
