"""Content-addressed, optionally-encrypted archive backends for ReconstructionRecords.

See the "Storage & durability" section of
REE_assembly/evidence/planning/preservation_snapshot_plan.md for the tiered
European-sovereign storage plan this implements. This module provides:

  * Encryptor          -- client-side encryption interface (you hold the key; the
                          store sees only ciphertext). NoEncryption (default) and
                          AesGcmEncryptor (AES-256-GCM).
  * LocalArchive       -- stdlib filesystem backend (a local hot copy, and what the
                          tests exercise).
  * S3Archive          -- any S3-compatible object store. The intended "node 1" is
                          HETZNER OBJECT STORAGE (German/EU jurisdiction) -- see
                          S3Archive for the endpoint/credential convention.

CONTENT-ADDRESSING + APPEND-ONLY. An object's key embeds the record's sha256
integrity hash, so identical content always lands at the same immutable key and a
DIFFERENT record can never overwrite an existing one -- append-only by
construction, matching the immutability the record primitive already guarantees.

DEPENDENCY DISCIPLINE. Top-level imports are stdlib only. `cryptography` is imported
lazily inside AesGcmEncryptor; `boto3` is imported lazily inside S3Archive and ONLY
when no client is injected -- so importing this module, using LocalArchive, or
injecting a client needs neither package. ree_core stays import-light.

ASCII-only in printed output.
"""

from __future__ import annotations

import base64
import os
from typing import Any, Callable, Dict, Optional

from ree_core.preservation.reconstruction_record import (
    ReconstructionRecord,
    IntegrityError,
    RecordExistsError,
    to_json,
    from_json,
    verify_integrity,
)


# --------------------------------------------------------------------------- #
# Client-side encryption                                                      #
# --------------------------------------------------------------------------- #

class NoEncryption:
    """Identity 'encryptor' -- store plaintext JSON. The default."""

    suffix = ".json"
    alg = "none"

    def encrypt(self, data: bytes) -> bytes:
        return data

    def decrypt(self, data: bytes) -> bytes:
        return data


class AesGcmEncryptor:
    """AES-256-GCM authenticated encryption. You hold the 32-byte key.

    Envelope: b"REEG1" + 12-byte random nonce + AESGCM ciphertext(+tag). Losing the
    key means the record is unrecoverable -- that is the point of client-side
    encryption, and why the key must be backed up independently of the archive.
    """

    suffix = ".json.enc"
    alg = "aes-256-gcm"
    _MAGIC = b"REEG1"
    _NONCE_LEN = 12

    def __init__(self, key: bytes):
        if not isinstance(key, (bytes, bytearray)) or len(key) != 32:
            raise ValueError("AES-256-GCM requires a 32-byte key")
        self._key = bytes(key)

    @staticmethod
    def generate_key() -> bytes:
        """A fresh 32-byte key. Store it safely and independently of the archive."""
        return os.urandom(32)

    @classmethod
    def from_env(cls, var: str = "REE_PRESERVATION_KEY") -> "AesGcmEncryptor":
        """Load a base64-encoded 32-byte key from an environment variable."""
        raw = os.environ.get(var)
        if not raw:
            raise ValueError("environment variable %r is not set" % var)
        return cls(base64.b64decode(raw))

    def _aesgcm(self):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # lazy
        return AESGCM(self._key)

    def encrypt(self, data: bytes) -> bytes:
        nonce = os.urandom(self._NONCE_LEN)
        ct = self._aesgcm().encrypt(nonce, data, None)
        return self._MAGIC + nonce + ct

    def decrypt(self, data: bytes) -> bytes:
        if not data.startswith(self._MAGIC):
            raise ValueError("not a REE AES-GCM envelope")
        body = data[len(self._MAGIC):]
        nonce, ct = body[: self._NONCE_LEN], body[self._NONCE_LEN:]
        return self._aesgcm().decrypt(nonce, ct, None)


# --------------------------------------------------------------------------- #
# Archive backends                                                            #
# --------------------------------------------------------------------------- #

class _BaseArchive:
    """Shared content-addressing, append-only, and encrypt/verify logic.

    Subclasses implement the three storage primitives `_exists/_write/_read`.
    """

    def __init__(self, encryptor: Optional[Any] = None, prefix: str = ""):
        self.encryptor = encryptor or NoEncryption()
        self.prefix = prefix

    def object_key(self, record: ReconstructionRecord) -> str:
        """`<prefix><record_id>/<sha256-hex>.json[.enc]` -- content-addressed.

        The integrity hash in the key is what makes the store append-only: a
        changed record has a different hash, hence a different key, and can never
        land on top of the original.
        """
        if not record.integrity:
            raise IntegrityError("record has no integrity hash; build it via capture()")
        sha = record.integrity.split(":", 1)[-1]
        return "%s%s/%s%s" % (self.prefix, record.record_id, sha, self.encryptor.suffix)

    def put(self, record: ReconstructionRecord) -> str:
        """Encrypt + store the record immutably; return its object key.

        Refuses to overwrite an existing key (RecordExistsError) -- append-only.
        """
        if not verify_integrity(record):
            raise IntegrityError("refusing to archive a record with a missing/stale integrity hash")
        key = self.object_key(record)
        if self._exists(key):
            raise RecordExistsError("object already exists at %r; records are immutable" % key)
        self._write(key, self.encryptor.encrypt(to_json(record).encode("utf-8")), record)
        return key

    def load(self, key: str) -> ReconstructionRecord:
        """Read + decrypt + re-verify integrity. Raises on tamper/corruption."""
        record = from_json(self.encryptor.decrypt(self._read(key)).decode("utf-8"))
        if not verify_integrity(record):
            raise IntegrityError("integrity check failed for %r (corruption or tampering)" % key)
        return record

    def exists(self, record: ReconstructionRecord) -> bool:
        return self._exists(self.object_key(record))

    # --- storage primitives (subclass) ---
    def _exists(self, key: str) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    def _write(self, key: str, body: bytes, record: ReconstructionRecord) -> None:  # pragma: no cover
        raise NotImplementedError

    def _read(self, key: str) -> bytes:  # pragma: no cover - abstract
        raise NotImplementedError


class LocalArchive(_BaseArchive):
    """Filesystem backend -- a local hot copy, and what the contract tests use."""

    def __init__(self, root: str, encryptor: Optional[Any] = None, prefix: str = ""):
        super().__init__(encryptor, prefix)
        self.root = str(root)

    def _path(self, key: str) -> str:
        return os.path.join(self.root, key)

    def _exists(self, key: str) -> bool:
        return os.path.exists(self._path(key))

    def _write(self, key: str, body: bytes, record: ReconstructionRecord) -> None:
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(body)
        os.replace(tmp, path)  # atomic within the directory

    def _read(self, key: str) -> bytes:
        with open(self._path(key), "rb") as fh:
            return fh.read()


# 404-ish error codes an S3 head/get raises when the key is absent.
_S3_NOT_FOUND = {"404", "NoSuchKey", "NotFound", "NoSuchBucket"}


def _s3_error_code(exc: Exception) -> Optional[str]:
    resp = getattr(exc, "response", None)
    if isinstance(resp, dict):
        return str(resp.get("Error", {}).get("Code", "")) or None
    return None


class S3Archive(_BaseArchive):
    """S3-compatible object store. Intended node 1: HETZNER OBJECT STORAGE (EU).

    Wiring for Hetzner (German/EU jurisdiction):
      * endpoint_url = "https://<location>.your-objectstorage.com"
        (location is fsn1 / nbg1 / hel1 -- all EU: Falkenstein, Nuremberg, Helsinki)
      * credentials via the standard boto3 chain -- set AWS_ACCESS_KEY_ID /
        AWS_SECRET_ACCESS_KEY (Hetzner's S3 access key + secret) in the environment.
        This module never reads, stores, or logs credentials itself.
      * pair with AesGcmEncryptor so Hetzner only ever sees ciphertext.

    `client` is injectable: pass a preconfigured boto3 client (e.g. path-style
    addressing) or, in tests, a fake -- when a client is given, boto3 is never
    imported. Object-lock (WORM) is opt-in via `object_lock_days`; the bucket must
    have been created with object lock enabled.
    """

    def __init__(
        self,
        bucket: str,
        endpoint_url: Optional[str] = None,
        encryptor: Optional[Any] = None,
        prefix: str = "",
        client: Optional[Any] = None,
        region_name: str = "eu-central-1",
        object_lock_days: Optional[int] = None,
        put_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(encryptor, prefix)
        self.bucket = bucket
        self.object_lock_days = object_lock_days
        self.put_kwargs = dict(put_kwargs or {})
        if client is None:
            import boto3  # lazy -- only when we must build a real client
            client = boto3.client(
                "s3", endpoint_url=endpoint_url, region_name=region_name
            )
        self.client = client

    def _exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception as exc:  # noqa: BLE001 - inspect the S3 error code
            if _s3_error_code(exc) in _S3_NOT_FOUND:
                return False
            raise

    def _write(self, key: str, body: bytes, record: ReconstructionRecord) -> None:
        args: Dict[str, Any] = dict(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType=(
                "application/octet-stream" if self.encryptor.alg != "none" else "application/json"
            ),
            Metadata={
                "record-id": record.record_id,
                "integrity": record.integrity or "",
                "enc-alg": self.encryptor.alg,
            },
        )
        if self.object_lock_days is not None:
            # Bucket must have object lock enabled; RetainUntilDate is computed by
            # the caller's clock, so it is passed through put_kwargs rather than
            # stamped here (this module takes no wall-clock dependency).
            args["ObjectLockMode"] = "COMPLIANCE"
        args.update(self.put_kwargs)
        self.client.put_object(**args)

    def _read(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
