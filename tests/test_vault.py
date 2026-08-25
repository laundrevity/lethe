import pytest

from lethe import vault


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("LETHE_ROOT", str(tmp_path))
    monkeypatch.setattr(vault, "SCRYPT_N", 2**12)  # fast for tests
    return tmp_path


def test_roundtrip_and_wrong_passphrase():
    blob = vault.encrypt(b"hello", "pw")
    assert blob.startswith(vault.MAGIC) and b"hello" not in blob
    assert vault.decrypt(blob, "pw") == b"hello"
    with pytest.raises(vault.WrongPassphrase):
        vault.decrypt(blob, "nope")
    with pytest.raises(ValueError):
        vault.decrypt(b"garbage", "pw")


def test_same_plaintext_different_ciphertext():
    assert vault.encrypt(b"x", "pw") != vault.encrypt(b"x", "pw")  # fresh salt + nonce each time


def test_save_load_list_delete(root):
    conv = vault.Conversation.new("owner/repo:Q8")
    conv.messages = [{"role": "user", "content": "secret question\nmore"}, {"role": "assistant", "content": "secret answer"}]
    p = vault.save(conv, "pw")
    assert p.parent == root / ".lethe" / "conversations"
    assert p.name == f"{conv.id}.lethe"
    raw = p.read_bytes()
    assert b"secret" not in raw and b"owner/repo" not in raw  # nothing descriptive in the clear
    assert not list(p.parent.glob("*.tmp"))
    assert oct(p.stat().st_mode & 0o777) == "0o600"

    entries = vault.list_entries()
    assert [e.id for e in entries] == [conv.id]

    back = vault.load(conv.id, "pw")
    assert back.messages == conv.messages and back.model == "owner/repo:Q8"
    assert back.title == "secret question" and back.turns == 1

    with pytest.raises(vault.WrongPassphrase):
        vault.load(conv.id, "wrong")

    assert vault.delete(conv.id)
    assert not p.exists() and not vault.delete(conv.id)
