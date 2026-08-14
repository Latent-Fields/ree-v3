"""Physical-token exporter for a ReconstructionRecord.

Produces the artifacts you can put on durable physical media (an engraved metal
key / jewellery, a QR, an M-DISC or a Piql/Svalbard deposit) so a preserved
organism survives an all-digital or all-company failure. See the "Storage &
durability" section of the plan doc.

For a record it writes, into an output directory:

  * <id>.key.txt      -- the ~180-byte KEY LINE: the minimal locator + verifier
                         (id, seed, code-commit, machine-class, sha256). This is
                         what you ENGRAVE. It does not contain the config; it
                         points at, and authenticates, the record.
  * <id>.record.json.gz -- the full ReconstructionRecord, gzipped (~10 KB). The
                         self-sufficient payload to deposit in an object store,
                         on M-DISC, or with Piql.
  * <id>.manifest.json -- machine-readable: the key fields, the record's integrity
                         hash, and the sha256 of the .gz FILE (so you can verify
                         the deposited blob), plus the file list.
  * <id>.README.txt   -- human-readable: what these are, how to VERIFY (the hash),
                         how to RECONSTRUCT (checkout the commit, load, rebuild).
  * <id>.key.svg / .png -- a QR of the key line, IF a QR backend is available
                         (`segno`, pure-python zero-deps: `pip install segno`).
                         SVG is vector -> ideal for a laser engraver.

DEPENDENCY DISCIPLINE: stdlib only at import time. The QR backend is discovered
lazily and is optional; without it every non-QR artifact is still produced. A
backend can also be injected (for tests, or a custom renderer).

ASCII-only output.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from typing import Any, Dict, Optional

from ree_core.preservation.reconstruction_record import (
    ReconstructionRecord,
    to_json,
)

KEY_SCHEME = "REE-PRESERVE/1"


# --------------------------------------------------------------------------- #
# The key line (what you engrave)                                             #
# --------------------------------------------------------------------------- #

def _sha_hex(record: ReconstructionRecord) -> str:
    return (record.integrity or "").split(":", 1)[-1]


def record_key_line(record: ReconstructionRecord) -> str:
    """The compact, self-describing locator+verifier string for a record.

    Pipe-delimited so it survives OCR/engraving and parses trivially. Fields:
    id, seed, commit (code at capture), machine (fidelity class), sha256 (the
    record's integrity hash -- authenticates any recovered copy).
    """
    prov = record.provenance or {}
    return "|".join([
        KEY_SCHEME,
        "id=%s" % record.record_id,
        "seed=%s" % record.seed,
        "commit=%s" % (prov.get("substrate_commit") or "-"),
        "machine=%s" % (prov.get("machine_class") or "-"),
        "sha256=%s" % _sha_hex(record),
    ])


def parse_key_line(line: str) -> Dict[str, str]:
    """Inverse of record_key_line. Raises ValueError on a non-matching scheme."""
    parts = line.strip().split("|")
    if not parts or parts[0] != KEY_SCHEME:
        raise ValueError("not a %s key line" % KEY_SCHEME)
    out: Dict[str, str] = {"scheme": parts[0]}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
# QR backend (optional, lazily discovered, injectable)                        #
# --------------------------------------------------------------------------- #

class _SegnoBackend:
    name = "segno"

    def write(self, data: str, svg_path: str, png_path: str) -> None:
        import segno  # lazy
        qr = segno.make(data, error="h")  # high ECC -> robust to a damaged surface
        qr.save(svg_path, scale=8)
        qr.save(png_path, scale=8)


class _QrcodeBackend:
    name = "qrcode"

    def write(self, data: str, svg_path: str, png_path: str) -> None:
        import qrcode  # lazy
        import qrcode.image.svg as svgmod
        qrcode.make(data, image_factory=svgmod.SvgImage).save(svg_path)
        qrcode.make(data).save(png_path)


def _default_qr_backend() -> Optional[Any]:
    try:
        import segno  # noqa: F401
        return _SegnoBackend()
    except ImportError:
        pass
    try:
        import qrcode  # noqa: F401
        return _QrcodeBackend()
    except ImportError:
        return None


# --------------------------------------------------------------------------- #
# The exporter                                                                #
# --------------------------------------------------------------------------- #

_README = """REE preservation token -- {record_id}

This is a physical-recovery token for one preserved REE organism. It exists so the
organism can be reconstructed later even if every online copy is gone.

FILES
  {record_id}.key.txt        The KEY LINE. Small enough to engrave. It LOCATES and
                             AUTHENTICATES the record; it does not itself contain
                             the organism.
  {record_id}.record.json.gz The full record (gzipped). Config + seed + environment
                             + provenance + the understanding we had at the time.
  {record_id}.manifest.json  Machine-readable index + hashes.
  {record_id}.key.svg/.png   A QR of the key line (present only if a QR backend was
                             installed at export time).

VERIFY (that a recovered record is authentic and unaltered)
  1. gunzip {record_id}.record.json.gz
  2. sha256 over the record's canonical content must equal the sha256 in the key
     line. The record module's verify_integrity() does exactly this check.

RECONSTRUCT (birth-replay; valid WITHIN the stated machine class)
  1. Obtain the code at commit {commit} (e.g. via Software Heritage) with a matching
     {machine} environment.
  2. Load the record and rebuild from birth:
       from ree_core.preservation import load_record, reconstruct_config
       from experiments._lib.arm_fingerprint import seeded_construct
       from ree_core.agent import REEAgent
       rec = load_record("{record_id}/reconstruction_record.json")
       agent = seeded_construct(rec.seed, lambda: REEAgent(reconstruct_config(rec)))
  This re-derives the whole life deterministically from birth. Multi-step replay is
  bit-exact only within the machine class stamped above.

KEY LINE
  {key_line}
"""


def export_token(
    record: ReconstructionRecord,
    out_dir: str,
    *,
    gzip_record: bool = True,
    make_qr: bool = True,
    qr_backend: Optional[Any] = None,
) -> Dict[str, Any]:
    """Write the token artifacts for `record` into `out_dir`. Returns a summary.

    Zero-dep artifacts (key line, record, manifest, README) are always written.
    QR is written only if `make_qr` and a backend is available/injected.
    """
    os.makedirs(out_dir, exist_ok=True)
    rid = record.record_id
    prov = record.provenance or {}
    key_line = record_key_line(record)
    written: Dict[str, str] = {}

    def _path(name: str) -> str:
        return os.path.join(out_dir, name)

    # key line
    key_path = _path("%s.key.txt" % rid)
    with open(key_path, "w", encoding="utf-8") as fh:
        fh.write(key_line + "\n")
    written["key"] = key_path

    # full record (gz or plain)
    record_bytes = to_json(record).encode("utf-8")
    if gzip_record:
        rec_path = _path("%s.record.json.gz" % rid)
        blob = gzip.compress(record_bytes)
    else:
        rec_path = _path("%s.record.json" % rid)
        blob = record_bytes
    with open(rec_path, "wb") as fh:
        fh.write(blob)
    written["record"] = rec_path
    gz_sha = "sha256:" + hashlib.sha256(blob).hexdigest()

    # QR of the key line (optional)
    qr_written = False
    qr_note = None
    if make_qr:
        backend = qr_backend or _default_qr_backend()
        if backend is not None:
            svg_path = _path("%s.key.svg" % rid)
            png_path = _path("%s.key.png" % rid)
            backend.write(key_line, svg_path, png_path)
            written["key_svg"] = svg_path
            written["key_png"] = png_path
            qr_written = True
        else:
            qr_note = "QR not written: no QR backend. `pip install segno` (pure-python) to enable."

    # manifest
    manifest = {
        "scheme": KEY_SCHEME,
        "record_id": rid,
        "seed": record.seed,
        "substrate_commit": prov.get("substrate_commit"),
        "machine_class": prov.get("machine_class"),
        "record_integrity": record.integrity,      # authenticates record CONTENT
        "record_file": os.path.basename(rec_path),
        "record_file_sha256": gz_sha,              # authenticates the deposited BLOB
        "key_line": key_line,
        "files": [os.path.basename(p) for p in written.values()],
    }
    man_path = _path("%s.manifest.json" % rid)
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    written["manifest"] = man_path

    # README
    readme_path = _path("%s.README.txt" % rid)
    with open(readme_path, "w", encoding="utf-8") as fh:
        fh.write(_README.format(
            record_id=rid,
            commit=prov.get("substrate_commit") or "(uncommitted/unknown)",
            machine=prov.get("machine_class") or "(unknown)",
            key_line=key_line,
        ))
    written["readme"] = readme_path

    return {
        "record_id": rid,
        "out_dir": out_dir,
        "key_line": key_line,
        "files": written,
        "qr_written": qr_written,
        "qr_note": qr_note,
        "manifest": manifest,
    }


def _main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="ree_core.preservation.token",
        description="Export a physical-recovery token for a ReconstructionRecord.",
    )
    ap.add_argument("--record", required=True,
                    help="path to a reconstruction_record.json or .json.gz")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--no-qr", action="store_true", help="skip QR generation")
    ap.add_argument("--no-gzip", action="store_true", help="write the record uncompressed")
    args = ap.parse_args(argv)

    from ree_core.preservation.reconstruction_record import from_json, verify_integrity

    raw = open(args.record, "rb").read()
    if args.record.endswith(".gz"):
        raw = gzip.decompress(raw)
    record = from_json(raw.decode("utf-8"))
    if not verify_integrity(record):
        print("ERROR: record integrity check failed; refusing to export.")
        return 1
    result = export_token(record, args.out, gzip_record=not args.no_gzip,
                          make_qr=not args.no_qr)
    print("exported token for %s ->" % result["record_id"])
    for name, path in result["files"].items():
        print("  %-10s %s" % (name, path))
    if result["qr_note"]:
        print("  note: %s" % result["qr_note"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
