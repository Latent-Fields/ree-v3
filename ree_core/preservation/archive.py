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


# --------------------------------------------------------------------------- #
# Fan-out to >1 backend (the >1-copy rule) + env-driven construction          #
# --------------------------------------------------------------------------- #

class MultiArchive:
    """Store a record in EVERY backend, so no single provider holds the only copy.

    This is the operational form of the ">1 copy" rule in the plan doc: give it
    e.g. [Hetzner S3, Scaleway S3] and one `put` lands the record in both. It is
    IDEMPOTENT -- a backend that already holds the record (append-only) is reported
    as "exists" rather than raising -- so re-running a deposit is safe.
    """

    def __init__(self, archives, names=None):
        self.archives = list(archives)
        if not self.archives:
            raise ValueError("MultiArchive needs at least one backend")
        self.names = list(names) if names else [
            getattr(a, "name", "archive%d" % i) for i, a in enumerate(self.archives)
        ]

    def put(self, record) -> Dict[str, Dict[str, str]]:
        """Deposit into every backend. Returns {name: {status, key}} per backend."""
        out: Dict[str, Dict[str, str]] = {}
        for name, a in zip(self.names, self.archives):
            key = a.object_key(record)
            if a.exists(record):
                out[name] = {"status": "exists", "key": key}
            else:
                out[name] = {"status": "written", "key": a.put(record)}
        return out

    def verify(self, record) -> Dict[str, bool]:
        """Read the record back from every backend and confirm integrity + identity."""
        out: Dict[str, bool] = {}
        for name, a in zip(self.names, self.archives):
            try:
                rec = a.load(a.object_key(record))
                out[name] = rec.integrity == record.integrity
            except Exception:  # noqa: BLE001 - any failure is a failed verify
                out[name] = False
        return out


def s3_archive_from_env(group: str, *, encryptor: Optional[Any] = None,
                        client_factory: Optional[Callable[..., Any]] = None) -> "S3Archive":
    """Build an S3Archive from a group of environment variables.

    Reads (for GROUP=e.g. HETZNER / SCALEWAY, so two providers get INDEPENDENT
    credentials -- one shared AWS_* env cannot serve two providers):

        REE_PRESERVE_<GROUP>_ENDPOINT   (required)  e.g. https://fsn1.your-objectstorage.com
        REE_PRESERVE_<GROUP>_BUCKET     (required)
        REE_PRESERVE_<GROUP>_KEY_ID     (required)  the provider's S3 access key
        REE_PRESERVE_<GROUP>_SECRET     (required)  the provider's S3 secret
        REE_PRESERVE_<GROUP>_REGION     (default eu-central-1)
        REE_PRESERVE_<GROUP>_PREFIX     (default "")

    Credentials are read from the environment and handed straight to the S3 client;
    this function never logs or persists them. Pass one shared `encryptor` (e.g.
    AesGcmEncryptor.from_env()) so every provider stores the same ciphertext scheme.
    `client_factory` is injectable for testing; the default builds a boto3 client
    (imported lazily).
    """
    g = group.upper()

    def _env(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
        key = "REE_PRESERVE_%s_%s" % (g, name)
        val = os.environ.get(key, default)
        if required and not val:
            raise ValueError("environment variable %r is not set" % key)
        return val

    endpoint = _env("ENDPOINT", required=True)
    bucket = _env("BUCKET", required=True)
    key_id = _env("KEY_ID", required=True)
    secret = _env("SECRET", required=True)
    region = _env("REGION", "eu-central-1")
    prefix = _env("PREFIX", "") or ""

    if client_factory is None:
        def client_factory(**kw):  # lazy boto3
            import boto3
            return boto3.client("s3", **kw)

    client = client_factory(
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
    )
    archive = S3Archive(bucket, endpoint_url=endpoint, client=client,
                        encryptor=encryptor, prefix=prefix)
    archive.name = group.lower()  # for MultiArchive reporting
    return archive
