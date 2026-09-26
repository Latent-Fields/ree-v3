"""Contract: every stamped manifest records the torch thread counts it ran at.

chip-20260926-manifest-torch-thread-count. REE_assembly
evidence/planning/runner_multislot_design_spike_20260926.md (premise P4, hazards
H3/H4) measured that torch's intra-op thread count was an UNRECORDED run variable:
nothing in the runner / ree_core / experiments/_lib set it, so a machine_affinity
"any" item ran at 8 threads on a cx43 and 2 on a cpx22, and BLAS reduction order
depends on that count. manifest_core.stamp_recording_core now stamps:

  torch_num_threads, torch_num_interop_threads -- the TRUE values at stamp time
      (after any driver torch.set_num_threads), only when torch is imported;
  thread_env_requested -- {OMP_NUM_THREADS, MKL_NUM_THREADS} as requested (None
      when unset), so intent vs truth is visible.

Pinned here:
  (a) the recorded counts equal torch's live values after an explicit
      set_num_threads(k), for two different k (a hardcoded value cannot pass);
  (b) the env request is recorded verbatim, None when unset;
  (c) fill-only: an author-set value survives unless overwrite=True;
  (d) the end-to-end sanctioned writer (pack_writer.write_flat_manifest) carries
      the fields to disk -- the path "every manifest" actually takes;
  (e) manifest_core still imports and stamps WITHOUT importing torch, and then
      omits the torch keys rather than fabricating them;
  (f) record-only: NOT in ALWAYS_CORE_KEYS / MANDATORY_CORE_KEYS (legacy-corpus
      reason, same as every post-2026-07 field), and NOT folded into
      machine_class (open user decision D3 in the design doc).

ASCII-only. Run: pytest tests/contracts/test_manifest_thread_provenance.py -q
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from experiments._lib import manifest_core as mc
from experiments._lib import arm_fingerprint as afp
from experiments import pack_writer as pw

_REE_V3_ROOT = Path(__file__).resolve().parents[2]
_KEYS = ("torch_num_threads", "torch_num_interop_threads", "thread_env_requested")


@pytest.fixture(autouse=True)
def _restore_threads():
    prev = torch.get_num_threads()
    yield
    torch.set_num_threads(prev)


def _two_distinct_counts():
    prev = torch.get_num_threads()
    return (1, 3) if prev not in (1, 3) else ((2, 3) if prev == 1 else (1, 2))


def test_stamp_records_true_intraop_count_after_set_num_threads():
    for k in _two_distinct_counts():
        torch.set_num_threads(k)
        m = {"run_id": "v3_exq_threadprov_v3", "outcome": "PASS"}
        mc.stamp_recording_core(m, config={"x": 1}, seeds=[0], elapsed_seconds=1.0)
        assert m["torch_num_threads"] == k == torch.get_num_threads()
        assert m["torch_num_interop_threads"] == torch.get_num_interop_threads()
        assert isinstance(m["torch_num_interop_threads"], int)


def test_env_request_recorded_verbatim_none_when_unset(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    m = {}
    mc.stamp_recording_core(m, seeds=[0], elapsed_seconds=1.0)
    assert m["thread_env_requested"] == {"OMP_NUM_THREADS": "3", "MKL_NUM_THREADS": None}


def test_fill_only_respects_author_value_unless_overwrite():
    torch.set_num_threads(_two_distinct_counts()[0])
    m = {"torch_num_threads": 99}
    mc.stamp_recording_core(m, seeds=[0], elapsed_seconds=1.0)
    assert m["torch_num_threads"] == 99
    mc.stamp_recording_core(m, seeds=[0], elapsed_seconds=1.0, overwrite=True)
    assert m["torch_num_threads"] == torch.get_num_threads()


def test_sanctioned_writer_carries_fields_to_disk(tmp_path):
    k = _two_distinct_counts()[1]
    torch.set_num_threads(k)
    manifest = {
        "run_id": "v3_exq_threadprov_writer_v3",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "outcome": "PASS",
    }
    out = pw.write_flat_manifest(manifest, tmp_path, config={"x": 1}, seeds=[0],
                                 elapsed_seconds=1.0)
    doc = json.loads(Path(out).read_text())
    for key in _KEYS:
        assert key in doc, key
    assert doc["torch_num_threads"] == k


def test_no_torch_import_and_torch_keys_omitted_when_torch_absent():
    code = (
        "import sys\n"
        "from experiments._lib import manifest_core as mc\n"
        "d = mc.torch_thread_provenance()\n"
        "assert 'torch' not in sys.modules, 'manifest_core imported torch'\n"
        "assert 'torch_num_threads' not in d and 'torch_num_interop_threads' not in d, d\n"
        "assert set(d['thread_env_requested']) == {'OMP_NUM_THREADS', 'MKL_NUM_THREADS'}\n"
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(_REE_V3_ROOT),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout + r.stderr


def test_record_only_not_core_not_machine_class():
    for key in _KEYS:
        assert key not in mc.ALWAYS_CORE_KEYS
        assert key not in mc.MANDATORY_CORE_KEYS
    assert "thread" not in json.dumps(afp.machine_class(), default=str).lower()
