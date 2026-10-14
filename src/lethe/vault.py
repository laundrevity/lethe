"""Opt-in encrypted conversation persistence.

Design (see README "Logless by default"):
  * Nothing here runs unless the user asks (`chat --save`, `/save`, `chat --resume`).
  * Passphrase -> scrypt (n=2^17) -> 32-byte key -> AES-256-GCM. The key is never stored;
    the passphrase lives in process memory for the session only.
  * Blob = b"LETHE1" + salt(16) + nonce(12) + ciphertext(+tag). Everything descriptive (title,
    model, timestamps, messages) is inside the ciphertext. Filenames are random ids.
  * Plaintext never touches disk: serialize -> encrypt -> write. The atomic write goes through a
    temp file that only ever holds ciphertext.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from .paths import state_dir

MAGIC = b"LETHE1"
SALT_LEN, NONCE_LEN = 16, 12
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**17, 8, 1


class WrongPassphrase(Exception):
    pass


def _key(passphrase: str, salt: bytes) -> bytes:
    return Scrypt(salt=salt, length=32, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P).derive(passphrase.encode("utf-8"))


def encrypt(data: bytes, passphrase: str) -> bytes:
    salt = secrets.token_bytes(SALT_LEN)
    nonce = secrets.token_bytes(NONCE_LEN)
    ct = AESGCM(_key(passphrase, salt)).encrypt(nonce, data, MAGIC)
    return MAGIC + salt + nonce + ct


def decrypt(blob: bytes, passphrase: str) -> bytes:
    if not blob.startswith(MAGIC):
        raise ValueError("not a lethe vault file")
    off = len(MAGIC)
    salt = blob[off : off + SALT_LEN]
    nonce = blob[off + SALT_LEN : off + SALT_LEN + NONCE_LEN]
    ct = blob[off + SALT_LEN + NONCE_LEN :]
    try:
        return AESGCM(_key(passphrase, salt)).decrypt(nonce, ct, MAGIC)
    except InvalidTag as e:
        raise WrongPassphrase("wrong passphrase (or corrupted file)") from e


# ------------------------------------------------------------------------------- conversations


@dataclass
class Conversation:
    id: str
    model: str  # canonical spec, e.g. owner/repo:QUANT
    messages: list[dict] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    title: str = ""
    interrupted: bool = False  # the last turn was stopped with Ctrl-C; the next prompt says so

    @classmethod
    def new(cls, model: str) -> "Conversation":
        return cls(id=secrets.token_hex(4), model=model)

    def to_bytes(self) -> bytes:
        self.updated = time.time()
        if not self.title:
            first = next((m["content"] for m in self.messages if m["role"] == "user"), "")
            self.title = (first if isinstance(first, str) else "").strip().splitlines()[0][:80] if first else ""
        return json.dumps(asdict(self), ensure_ascii=False).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "Conversation":
        d = json.loads(data.decode("utf-8"))
        return cls(**d)

    @property
    def turns(self) -> int:
        return sum(1 for m in self.messages if m["role"] == "user")


def vault_dir() -> Path:
    d = state_dir() / "conversations"
    d.mkdir(parents=True, exist_ok=True)
    return d


def path_for(conv_id: str) -> Path:
    return vault_dir() / f"{conv_id}.lethe"


def save(conv: Conversation, passphrase: str) -> Path:
    blob = encrypt(conv.to_bytes(), passphrase)
    dest = path_for(conv.id)
    tmp = dest.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    return dest


def load(conv_id: str, passphrase: str) -> Conversation:
    p = path_for(conv_id)
    if not p.exists():
        raise FileNotFoundError(f"no saved conversation {conv_id}")
    return Conversation.from_bytes(decrypt(p.read_bytes(), passphrase))


@dataclass
class VaultEntry:
    id: str
    path: Path
    size: int
    mtime: float


def list_entries() -> list[VaultEntry]:
    out = []
    for p in sorted(vault_dir().glob("*.lethe"), key=lambda p: p.stat().st_mtime, reverse=True):
        st = p.stat()
        out.append(VaultEntry(id=p.stem, path=p, size=st.st_size, mtime=st.st_mtime))
    return out


def delete(conv_id: str) -> bool:
    p = path_for(conv_id)
    if not p.exists():
        return False
    # overwrite before unlink so a casual undelete gets noise, not ciphertext
    size = p.stat().st_size
    with open(p, "r+b") as f:
        f.write(secrets.token_bytes(size))
        f.flush()
        os.fsync(f.fileno())
    p.unlink()
    return True
