"""
Content-addressed sink for bulky per-step trace arrays (Q-081 recording harness).

Why this exists
---------------
The Q-081 cross-stream telemetry audit
(REE_assembly/evidence/planning/q081_cross_stream_telemetry_audit.md, section 4 item 2)
requires that a per-step multi-stream trace be stored BY REFERENCE, not inline in the
experiment manifest. At 32-D x ~6 streams x steps x seeds the payload is orders of
magnitude larger than any manifest, and the git-tracked coordination plane -- shared by
four concurrent phase3 writers -- must not absorb it. The Experimental Recording Standard
(experimental_recording_standard_2026-07-12.md section 3d, "Large artifacts by reference")
already mandates a content-addressed pointer for exactly this class of payload.

Contract
--------
`TraceStore.put(arrays, meta)` writes ONE compressed .npz named by a CONTENT digest, and
returns a small plain-JSON pointer dict. The pointer is what goes in the manifest; the
blob never does. Any consumer can verify the trace it fetched is the trace the manifest
named.

The digest is taken over the canonical CONTENT (each array's name, dtype, shape and raw
bytes, in sorted order, plus the canonical meta json) -- NOT over the .npz file bytes.
This matters: numpy's savez writes a zip archive whose entry headers embed the local
wallclock time, so two runs producing bit-identical arrays would produce different file
bytes and a file-byte hash would defeat the point of content-addressing. Digesting the
content makes a re-run of identical arrays land on the same path (idempotent) and makes
verification a statement about the data rather than about the packaging.

Storage root
------------
`$REE_TRACE_ROOT` if set, else `<ree-v3>/traces/`, which is gitignored. Traces are
therefore MACHINE-LOCAL: a run executed on a cloud worker leaves its trace on that
worker, and the pointer records `machine` so an analyst knows where to fetch it from.
This is deliberate -- keeping the coordination plane untouched was the requirement;
retrieval is a separate concern and is not solved here.

ASCII-only output (repo rule). numpy + stdlib only; no torch, no ree_core.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import socket
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

TRACE_STORE_SCHEMA = "trace_ptr/v1"

# Bytes. A pointer dict that exceeds this has array payload smuggled into it, which is
# precisely what this module exists to prevent. Enforced by a contract test.
MAX_POINTER_BYTES = 8192


def default_trace_root() -> Path:
    """Storage root: $REE_TRACE_ROOT, else <ree-v3>/traces/ (gitignored)."""
    env = os.environ.get("REE_TRACE_ROOT")
    if env:
        return Path(env)
    # experiments/_lib/trace_store.py -> experiments/_lib -> experiments -> ree-v3
    return Path(__file__).resolve().parents[2] / "traces"


def content_digest(arrays: Mapping[str, np.ndarray], meta_json: str) -> str:
    """sha256 over the canonical CONTENT of a trace (not over the .npz file bytes).

    Digested, in sorted key order: name, dtype string, shape, then the array's raw
    C-contiguous bytes; finally the canonical meta json. Independent of archive
    packaging, so identical data always yields the same digest.
    """
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(name.encode("utf-8"))
        h.update(str(a.dtype).encode("utf-8"))
        h.update(repr(tuple(int(d) for d in a.shape)).encode("utf-8"))
        h.update(a.tobytes())
    h.update(meta_json.encode("utf-8"))
    return h.hexdigest()


class TraceStore:
    """Content-addressed .npz store. One blob per finalized trace."""

    def __init__(self, root: Optional[Path] = None, machine: Optional[str] = None):
        self.root = Path(root) if root is not None else default_trace_root()
        self.machine = machine if machine is not None else socket.gethostname()

    def put(
        self,
        arrays: Mapping[str, np.ndarray],
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Write arrays as one content-addressed .npz; return a manifest pointer.

        `meta` is small JSON-serialisable side-car data (schema version, stream
        descriptors, config declaration). It is embedded INSIDE the blob as a
        `__meta__` json string so a fetched trace is self-describing, and it therefore
        participates in the content hash.

        Returns a plain-JSON dict; no numpy types leak out.
        """
        if "__meta__" in arrays:
            raise ValueError("'__meta__' is reserved by TraceStore")
        meta_json = json.dumps(meta or {}, sort_keys=True, separators=(",", ":"))
        sha = content_digest(arrays, meta_json)

        payload = dict(arrays)
        payload["__meta__"] = np.array(meta_json)
        buf = io.BytesIO()
        np.savez_compressed(buf, **{k: payload[k] for k in sorted(payload)})
        blob = buf.getvalue()

        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{sha}.npz"
        if not path.exists():
            # Write-then-rename so a concurrent reader never sees a partial blob.
            tmp = self.root / f".{sha}.npz.tmp.{os.getpid()}"
            tmp.write_bytes(blob)
            tmp.replace(path)

        streams = {
            k: {"shape": [int(d) for d in v.shape], "dtype": str(v.dtype)}
            for k, v in arrays.items()
        }
        return {
            "pointer_schema": TRACE_STORE_SCHEMA,
            "sha256": sha,
            "bytes": int(len(blob)),
            "filename": path.name,
            "root_hint": str(self.root),
            "machine": self.machine,
            "storage": "machine_local_content_addressed",
            "n_streams": len(arrays),
            "streams": streams,
        }

    def get(self, pointer: Mapping[str, Any]) -> Dict[str, Any]:
        """Load a trace this store holds. Verifies the content against its digest.

        Raises FileNotFoundError if the blob is not on THIS machine (the normal case
        when the run executed on a worker), and ValueError on a digest mismatch.
        """
        sha = pointer["sha256"]
        path = self.root / f"{sha}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"trace blob {sha} not under {self.root} "
                f"(recorded on machine={pointer.get('machine')})"
            )
        with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if k != "__meta__"}
            meta_json = str(z["__meta__"]) if "__meta__" in z.files else "{}"
        actual = content_digest(arrays, meta_json)
        if actual != sha:
            raise ValueError(f"trace content digest mismatch: expected {sha}, got {actual}")
        return {"arrays": arrays, "meta": json.loads(meta_json)}


def pointer_is_lean(pointer: Mapping[str, Any]) -> bool:
    """True if the pointer carries no array payload (manifest-safe)."""
    try:
        encoded = json.dumps(pointer, sort_keys=True)
    except TypeError:
        return False
    return len(encoded.encode("utf-8")) <= MAX_POINTER_BYTES
