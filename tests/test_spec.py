import pytest

from lethe import spec as S


@pytest.fixture
def models_root(tmp_path, monkeypatch):
    monkeypatch.setenv("LETHE_ROOT", str(tmp_path))
    d = tmp_path / "models" / "HauhauCS--Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF"
    d.mkdir(parents=True)
    (d / "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf").write_bytes(b"x" * 10)
    (d / "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf").write_bytes(b"x" * 5)
    (d / "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf").write_bytes(b"x")
    (d / "mmproj-BF16.gguf").write_bytes(b"x")
    u = tmp_path / "models" / "unsloth--MiniMax-M2.7-GGUF"
    u.mkdir()
    for i in (1, 2, 3):
        (u / f"MiniMax-M2.7-UD-IQ4_XS-0000{i}-of-00003.gguf").write_bytes(b"x" * 7)
    return tmp_path


def test_parse_forms():
    s = S.ModelSpec.parse("HauhauCS/Foo-GGUF:Q8_K_P")
    assert s.repo == "HauhauCS/Foo-GGUF" and s.quant == "Q8_K_P" and s.is_hub
    assert str(s) == "HauhauCS/Foo-GGUF:Q8_K_P"
    s = S.ModelSpec.parse("HauhauCS/Foo-GGUF:q8_k_p")
    assert s.quant == "Q8_K_P"
    s = S.ModelSpec.parse("HauhauCS/Foo-GGUF")
    assert s.repo and s.quant is None
    s = S.ModelSpec.parse("q8_k_p")
    assert not s.is_hub and s.path is None and s.raw == "q8_k_p"


def test_local_models_group_shards_and_hide_sidecars(models_root):
    ms = S.local_models()
    specs = sorted(m.spec for m in ms)
    assert specs == [
        "HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF:Q4_K_P",
        "HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF:Q8_K_P",
        "unsloth/MiniMax-M2.7-GGUF:IQ4_XS",
    ]
    mm = next(m for m in ms if m.repo == "unsloth/MiniMax-M2.7-GGUF")
    assert len(mm.files) == 3 and mm.size == 21 and mm.path.name.endswith("00001-of-00003.gguf")
    assert len(S.local_models(include_sidecars=True)) == 5


def test_resolve(models_root):
    lm = S.resolve("HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF:Q8_K_P")
    assert lm.quant == "Q8_K_P" and lm.size == 10
    assert S.resolve("q8_k_p").quant == "Q8_K_P"
    assert S.resolve("unsloth/MiniMax-M2.7-GGUF").quant == "IQ4_XS"  # only one quant locally
    assert S.resolve(str(lm.path)).files == lm.files


def test_resolve_errors(models_root):
    with pytest.raises(S.ModelNotFound) as e:
        S.resolve("HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF:Q6_K_P")
    assert e.value.spec.quant == "Q6_K_P" and "Q8_K_P" in str(e.value)
    with pytest.raises(S.ModelNotFound):
        S.resolve("HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF")  # ambiguous: two quants
    with pytest.raises(S.ModelNotFound):
        S.resolve("nonexistent")
    with pytest.raises(S.ModelNotFound):
        S.resolve("Aggressive")  # ambiguous substring
