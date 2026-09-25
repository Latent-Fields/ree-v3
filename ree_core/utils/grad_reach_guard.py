"""Gradient-reach guard: a PURE INSTRUMENT that reports, per parameter tensor, whether the
optimizer(s) that hold it actually train it.

Why this exists
---------------
The 2026-09-25 gradient-reach census (REE_assembly
``evidence/planning/gradient_reach_census_20260925.md``, commit 940c690c9dd) found act-path
modules that sit INSIDE an optimizer and are never trained: the V3-EXQ-1078 recipe
``Adam(agent.parameters())`` on the E1+E2 losses never reaches
``hippocampal.action_object_decoder`` (nor the DR-13 GRU, the E2 world head, ...), and the
all-ON e2 optimizer (``experiments/_lib/allon_training.py``) holds
``e2.self_transition`` / ``e2.self_action_encoder`` with grad ``None`` on every step. The
check drivers use today -- "some parameter in the optimizer moved" -- PASSES both recipes.
This guard FAILS both, naming the dead tensors. Spec: REE_assembly
``evidence/planning/native_waking_trainer_design_20260925.md`` section 3 (commit
0c0f5b76ec); this module productionises that record's probe-only prototype.

Scope (deliberately narrow)
---------------------------
* Nothing in ree_core imports this module. It changes no default and no behaviour.
* It RAISES NOTHING on a FAIL: it returns a verdict. Whether a FAIL should stop a run or
  annotate a manifest is an open user decision (design record section 6, D4), so that
  policy is left to the caller.
* Calling it is byte-neutral: every observation runs under ``torch.random.fork_rng`` and
  only reads ``.grad`` / clones parameters; it never writes a gradient, a parameter or an
  RNG state. (The caller's ``run_steps`` of course trains whatever it trains.)

What it asserts (design record section 3a)
------------------------------------------
G1 REACH       every non-allowlisted tensor held by an observed optimizer gets a NONZERO,
               FINITE grad on at least one of that optimizer's steps in the window. grad
               ``None``, grad exactly 0 and a non-finite grad all count as "not reached".
               NOT "None at any step": the native zero-loss sentinel
               (``next(module.parameters()).sum() * 0.0``) legitimately yields None/0
               while replay buffers fill; the naive form false-alarmed on 30/30 healthy
               E1 tensors in the census recipe.
G2 MOVED       every reached tensor changed value over the window (catches lr 0, or a
               tensor that gets gradient but whose optimizer never steps it).
G3 ALLOWLIST   an allowlisted (frozen-by-design) tensor that DOES get gradient is a stale
               allowlist entry and FAILS -- an allowlist is itself a negative instrument
               and must not silently hide a module that has become trainable.
G4 NON-VACUOUS no optimizer observed, zero steps, no tensor checked, or fewer than
               ``min_steps`` observed steps -> CANNOT_DETERMINE, never PASS.
G5 LEAK        (report only) a tensor in ``named_parameters`` that no observed optimizer
               holds but that carried a nonzero grad at an observed step. Gradient that
               reaches a module no optimizer steps is discarded; whether that is a defect
               depends on the design, so it is reported, not failed. It can include a
               stale grad left by a DIFFERENT optimizer's backward in the same tick.

Verdict precedence: FAIL if any tensor with a sufficient window fails G1/G2, or any
allowlisted tensor is stale (positive evidence needs no window); otherwise
CANNOT_DETERMINE if any G4 condition holds; otherwise PASS.

Evidence domain: a PASS is D1 (reach), not influence. It says gradient arrives and the
weights move, not that the module's output matters to any consumer.

Usage
-----
    res = check_grad_reach(run_steps, optimizers=[opt],
                           named_parameters=agent.named_parameters())
    print(res.summary())            # ASCII
    if res.status == FAIL: ...      # caller's policy

``optimizers=None`` observes EVERY optimizer that steps while ``run_steps`` runs (via
``torch.optim.optimizer.register_optimizer_step_pre_hook``) -- for recipes that build their
optimizer internally (e.g. ``ZSelfP0Trainer.train``). For recorded backward passes with no
optimizer, build a ``GradReachGuard`` and call ``observe_params`` after each backward.
"""

from __future__ import annotations

import dataclasses
from collections import OrderedDict
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

PASS = "PASS"
FAIL = "FAIL"
CANNOT_DETERMINE = "CANNOT_DETERMINE"

# Per-tensor statuses.
T_REACHED = "reached"                      # nonzero finite grad on >=1 step, and moved
T_REACHED_NOT_MOVED = "reached_not_moved"  # G2 failure
T_DEAD = "dead"                            # G1 failure (window >= min_steps)
T_UNDETERMINED = "undetermined"            # never reached, but window < min_steps
T_ALLOWLISTED = "allowlisted"              # frozen by design, and got no gradient
T_STALE_ALLOWLIST = "stale_allowlist"      # G3 failure
T_NO_REQUIRES_GRAD = "requires_grad_false" # in an optimizer but requires_grad=False
T_OUTSIDE = "outside"                      # not held by any observed optimizer
T_OUTSIDE_LEAK = "outside_leak"            # G5: outside, but carried a nonzero grad

_FAILING = (T_REACHED_NOT_MOVED, T_DEAD, T_STALE_ALLOWLIST)

# Frozen-by-design parameter-name prefixes (over ``REEAgent.named_parameters()``) with the
# reason each is legitimately never trained. Source: the gradient-reach census, REE_assembly
# commit 940c690c9dd ("Rows that look random but are frozen by design"), plus the
# ContextMemory write path added by the native-waking-trainer design record, commit
# 0c0f5b76ec, sections 0 and 1b. file:line anchors re-verified at ree-v3 dbc6db8.
# Each entry is guarded by G3: if it ever receives gradient the guard FAILS it as stale.
# NOT on this list, on purpose: ``world_obs_encoder`` (an open user decision, design record
# section 6 D3) -- until that is decided an untrained world_obs_encoder must be visible.
FROZEN_BY_DESIGN: Dict[str, str] = OrderedDict([
    ("residue_field.rbf_field",
     "non-gradient harm-accumulation rule (census 940c690c9dd)"),
    ("e1.context_memory.write_gate",
     "write consumed under torch.no_grad (e1_deep.py:380-386); trained only by the opt-in "
     "write-addressing loss (contextmemory_write_addressing_loss_weight > 0)"),
    ("e1.context_memory.write_content",
     "write consumed under torch.no_grad (e1_deep.py:380-386), when gated_content_write"),
    ("lateral_pfc.delta_proj",
     "frozen-random, not trained in landing (lateral_pfc_analog.py:233-236)"),
    ("lateral_pfc.world_proj",
     "frozen-random, not trained in landing (lateral_pfc_analog.py:233-236)"),
    ("ofc.state_bias_head",
     "last Linear zeroed, output 0, when train_state_bias_head=False "
     "(ofc_analog.py:201-215); expected STALE (G3) when that flag is True"),
])


@dataclasses.dataclass
class TensorReach:
    """Per-tensor record over the observation window."""

    name: str
    numel: int
    in_optimizer: bool
    optimizers: Tuple[str, ...]
    n_obs: int          # optimizer steps (of optimizers holding it) observed
    n_none: int
    n_zero: int
    n_nonfinite: int
    n_nonzero: int
    moved: Optional[bool]   # None when not held by an observed optimizer
    allowlist_reason: Optional[str]
    status: str


@dataclasses.dataclass
class GradReachResult:
    """Verdict plus the per-tensor evidence it was derived from."""

    status: str
    reasons: List[str]
    min_steps: int
    optimizer_steps: Dict[str, int]
    tensors: List[TensorReach]

    def by_status(self, *statuses: str) -> List[TensorReach]:
        return [t for t in self.tensors if t.status in statuses]

    @property
    def n_checked(self) -> int:
        return sum(1 for t in self.tensors if t.in_optimizer and t.allowlist_reason is None
                   and t.status != T_NO_REQUIRES_GRAD)

    @property
    def dead_names(self) -> List[str]:
        return [t.name for t in self.by_status(T_DEAD)]

    @property
    def failing_names(self) -> List[str]:
        return [t.name for t in self.by_status(*_FAILING)]

    @property
    def leaked_names(self) -> List[str]:
        return [t.name for t in self.by_status(T_OUTSIDE_LEAK)]

    def modules(self, statuses: Sequence[str], depth: int = 2) -> Dict[str, Tuple[int, int]]:
        """Aggregate tensors with the given statuses to ``depth``-component module
        prefixes: {prefix: (n_tensors, n_params)}."""
        agg: Dict[str, List[int]] = OrderedDict()
        for t in self.by_status(*statuses):
            key = ".".join(t.name.split(".")[:depth])
            a = agg.setdefault(key, [0, 0])
            a[0] += 1
            a[1] += int(t.numel)
        return {k: (v[0], v[1]) for k, v in agg.items()}

    def summary(self, depth: int = 2) -> str:
        """ASCII-only multi-line summary."""
        lines = ["grad-reach: %s  checked=%d min_steps=%d steps=%s" % (
            self.status, self.n_checked, self.min_steps,
            ",".join("%s:%d" % kv for kv in self.optimizer_steps.items()) or "none")]
        for r in self.reasons:
            lines.append("  reason: %s" % r)
        for label, sts in (("DEAD", (T_DEAD,)),
                           ("REACHED-NOT-MOVED", (T_REACHED_NOT_MOVED,)),
                           ("STALE-ALLOWLIST", (T_STALE_ALLOWLIST,)),
                           ("UNDETERMINED", (T_UNDETERMINED,)),
                           ("LEAK(report-only)", (T_OUTSIDE_LEAK,))):
            for mod, (nt, npar) in self.modules(sts, depth).items():
                lines.append("  %s %s tensors=%d params=%d" % (label, mod, nt, npar))
        return "\n".join(lines)


def _allow_reason(name: str, allowlist: Mapping[str, str]) -> Optional[str]:
    for pre, why in allowlist.items():
        if name == pre or name.startswith(pre + "."):
            return why
    return None


def _classify_grad(g: Optional[torch.Tensor]) -> str:
    if g is None:
        return "none"
    g = g.detach()
    if not bool(torch.isfinite(g).all()):
        return "nonfinite"
    if not bool((g != 0).any()):
        return "zero"
    return "nonzero"


class GradReachGuard:
    """Accumulates per-tensor gradient observations; ``result()`` gives the verdict.

    Observation paths (any mix):
      * ``attach(optimizer)``: a step pre-hook on that optimizer (removed by ``detach``).
      * ``attach_global()``: observes every optimizer that steps (torch global hook).
      * ``observe_params(params, label)``: manual, for recorded backward passes.
    """

    def __init__(self,
                 named_parameters: Optional[Iterable[Tuple[str, torch.nn.Parameter]]] = None,
                 allowlist: Optional[Mapping[str, str]] = None,
                 min_steps: int = 8) -> None:
        self.allowlist: Dict[str, str] = dict(FROZEN_BY_DESIGN if allowlist is None
                                              else allowlist)
        self.min_steps = int(min_steps)
        self._name_of: Dict[int, str] = OrderedDict()
        self._outside_params: Dict[int, torch.nn.Parameter] = OrderedDict()
        for n, p in (named_parameters or []):
            if id(p) not in self._name_of:
                self._name_of[id(p)] = n
                self._outside_params[id(p)] = p
        # id(param) -> state
        self._held: Dict[int, dict] = OrderedDict()
        self._opt_label: Dict[object, str] = {}
        self._opt_steps: Dict[str, int] = OrderedDict()
        self._leak: Dict[int, int] = {}
        self._handles: list = []

    # -- registration -------------------------------------------------------------
    def _label_for(self, key: object, hint: str) -> str:
        if key not in self._opt_label:
            label = "%s#%d" % (hint, len(self._opt_label))
            self._opt_label[key] = label
            self._opt_steps[label] = 0
        return self._opt_label[key]

    def _register_params(self, label: str, params: Sequence[torch.nn.Parameter]) -> None:
        with torch.random.fork_rng(devices=[]):
            for j, p in enumerate(params):
                st = self._held.get(id(p))
                if st is None:
                    name = self._name_of.get(id(p), "%s.param%d" % (label, j))
                    st = {"p": p, "name": name, "opts": [], "n_obs": 0, "none": 0,
                          "zero": 0, "nonfinite": 0, "nonzero": 0,
                          "snap": p.detach().clone()}
                    self._held[id(p)] = st
                if label not in st["opts"]:
                    st["opts"].append(label)

    def register_optimizer(self, optimizer: torch.optim.Optimizer) -> str:
        label = self._label_for(id(optimizer), type(optimizer).__name__)
        params = [p for grp in optimizer.param_groups for p in grp["params"]]
        self._register_params(label, params)
        return label

    def attach(self, optimizer: torch.optim.Optimizer) -> None:
        self.register_optimizer(optimizer)
        self._handles.append(optimizer.register_step_pre_hook(
            lambda opt, args, kwargs: self.observe_optimizer(opt)))

    def attach_global(self) -> None:
        from torch.optim.optimizer import register_optimizer_step_pre_hook
        self._handles.append(register_optimizer_step_pre_hook(
            lambda opt, args, kwargs: self.observe_optimizer(opt)))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    # -- observation --------------------------------------------------------------
    def observe_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Record grads for this optimizer's params. Call BEFORE ``optimizer.step()``."""
        label = self.register_optimizer(optimizer)
        params = [p for grp in optimizer.param_groups for p in grp["params"]]
        self._observe(label, params)

    def observe_params(self, params: Sequence[torch.nn.Parameter], label: str = "manual") -> None:
        """Manual observation of one recorded backward pass over ``params``."""
        params = list(params)
        lab = self._label_for(("manual", label), label)
        self._register_params(lab, params)
        self._observe(lab, params)

    def _observe(self, label: str, params: Sequence[torch.nn.Parameter]) -> None:
        with torch.random.fork_rng(devices=[]):
            self._opt_steps[label] += 1
            seen = set()
            for p in params:
                if id(p) in seen:
                    continue
                seen.add(id(p))
                st = self._held[id(p)]
                st["n_obs"] += 1
                st[_classify_grad(p.grad)] += 1
            for pid, p in self._outside_params.items():
                if pid in self._held:
                    continue
                if _classify_grad(p.grad) == "nonzero":
                    self._leak[pid] = self._leak.get(pid, 0) + 1

    # -- verdict ------------------------------------------------------------------
    def result(self) -> GradReachResult:
        with torch.random.fork_rng(devices=[]):
            return self._result()

    def _result(self) -> GradReachResult:
        tensors: List[TensorReach] = []
        reasons: List[str] = []
        for pid, st in self._held.items():
            p = st["p"]
            why = _allow_reason(st["name"], self.allowlist)
            reached = st["nonzero"] > 0
            moved = bool(not torch.equal(p.detach(), st["snap"]))
            if not p.requires_grad:
                status = T_NO_REQUIRES_GRAD
            elif why is not None:
                status = T_STALE_ALLOWLIST if reached else T_ALLOWLISTED
            elif reached:
                status = T_REACHED if moved else T_REACHED_NOT_MOVED
            elif st["n_obs"] >= self.min_steps:
                status = T_DEAD
            else:
                status = T_UNDETERMINED
            tensors.append(TensorReach(
                name=st["name"], numel=int(p.numel()), in_optimizer=True,
                optimizers=tuple(st["opts"]), n_obs=st["n_obs"], n_none=st["none"],
                n_zero=st["zero"], n_nonfinite=st["nonfinite"], n_nonzero=st["nonzero"],
                moved=moved, allowlist_reason=why, status=status))
        for pid, p in self._outside_params.items():
            if pid in self._held:
                continue
            n_leak = self._leak.get(pid, 0)
            tensors.append(TensorReach(
                name=self._name_of[pid], numel=int(p.numel()), in_optimizer=False,
                optimizers=(), n_obs=0, n_none=0, n_zero=0, n_nonfinite=0,
                n_nonzero=n_leak, moved=None,
                allowlist_reason=_allow_reason(self._name_of[pid], self.allowlist),
                status=T_OUTSIDE_LEAK if n_leak else T_OUTSIDE))

        result = GradReachResult(status=PASS, reasons=reasons, min_steps=self.min_steps,
                                 optimizer_steps=dict(self._opt_steps), tensors=tensors)
        # G4 conditions (collected even when FAIL wins, so the record is complete).
        if not self._opt_steps:
            reasons.append("no optimizer observed")
        elif sum(self._opt_steps.values()) == 0:
            reasons.append("zero optimizer steps observed")
        short = [lab for lab, n in self._opt_steps.items() if n < self.min_steps]
        if short and self._opt_steps and sum(self._opt_steps.values()) > 0:
            reasons.append("window shorter than min_steps=%d for: %s"
                           % (self.min_steps, ", ".join(short)))
        if result.n_checked == 0:
            reasons.append("no tensor checked (empty group, or every tensor allowlisted / "
                           "requires_grad=False)")
        failing = result.failing_names
        if failing:
            result.status = FAIL
            reasons.insert(0, "%d failing tensor(s): %d dead, %d reached-not-moved, "
                              "%d stale-allowlist" % (
                                  len(failing), len(result.by_status(T_DEAD)),
                                  len(result.by_status(T_REACHED_NOT_MOVED)),
                                  len(result.by_status(T_STALE_ALLOWLIST))))
        elif reasons:
            result.status = CANNOT_DETERMINE
        return result


def check_grad_reach(run_steps: Callable[[], object],
                     optimizers: Optional[Sequence[torch.optim.Optimizer]] = None,
                     named_parameters: Optional[Iterable[Tuple[str, torch.nn.Parameter]]] = None,
                     allowlist: Optional[Mapping[str, str]] = None,
                     min_steps: int = 8) -> GradReachResult:
    """Run ``run_steps()`` while observing optimizer steps; return the reach verdict.

    optimizers       the optimizers to observe. ``None`` observes EVERY optimizer that
                     steps during ``run_steps`` (global hook) -- use it when the recipe
                     builds its optimizer internally. An empty sequence observes nothing
                     and returns CANNOT_DETERMINE.
    named_parameters names for the tensors (e.g. ``agent.named_parameters()``); tensors in
                     it that no observed optimizer holds are reported as outside / leak.
    allowlist        name-prefix -> reason; default ``FROZEN_BY_DESIGN``. Pass ``{}`` for
                     none.
    min_steps        the G4 window floor.

    Never raises on a FAIL; the verdict is the return value.
    """
    guard = GradReachGuard(named_parameters=named_parameters, allowlist=allowlist,
                           min_steps=min_steps)
    if optimizers is None:
        guard.attach_global()
    else:
        for opt in optimizers:
            guard.attach(opt)
    try:
        run_steps()
    finally:
        guard.detach()
    return guard.result()
