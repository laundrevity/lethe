from pathlib import Path

import numpy as np
import pytest
from gguf import GGUFWriter

from lethe.ggufmeta import NeedMoreData, parse_kv, parse_local, quant_from_filename, shape_from_kv


def write_gguf(path: Path, arch: str, **kv) -> Path:
    w = GGUFWriter(str(path), arch)
    for k, v in kv.items():
        key = f"{arch}.{k}" if not k.startswith(("general.", "tokenizer.")) else k
        if isinstance(v, bool):
            w.add_bool(key, v)
        elif isinstance(v, int):
            (w.add_uint64 if v > 0xFFFFFFFF else w.add_uint32)(key, v)
        elif isinstance(v, float):
            w.add_float32(key, v)
        elif isinstance(v, str):
            w.add_string(key, v)
        elif isinstance(v, list):
            w.add_array(key, v)
    # one tiny tensor so the file is well-formed
    w.add_tensor("token_embd.weight", np.zeros((4, 4), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


@pytest.fixture
def hybrid_gguf(tmp_path):
    """Qwen3.5-style hybrid: 8 layers, 1 in 4 with a KV cache, 248K vocab summarized by length."""
    return write_gguf(
        tmp_path / "Foo-27B-Q8_K_P.gguf",
        "qwen35",
        **{"general.name": "Foo 27B", "general.file_type": 7, "general.parameter_count": 27_000_000_000},
        context_length=262144,
        block_count=8,
        embedding_length=5120,
        full_attention_interval=4,
        **{"attention.head_count": 24, "attention.head_count_kv": 4, "attention.key_length": 256, "attention.value_length": 256},
        **{"tokenizer.ggml.tokens": [f"t{i}" for i in range(1000)]},
    )


def test_parse_local_hybrid(hybrid_gguf):
    kv = parse_local(hybrid_gguf)
    assert kv["general.architecture"] == "qwen35"
    assert kv["qwen35.block_count"] == 8
    # big arrays are summarized, not materialized
    assert kv["tokenizer.ggml.tokens"] == {"__array_len__": 1000}

    s = shape_from_kv(kv, hybrid_gguf.name)
    assert s.n_layer == 8
    assert s.n_attn_layers == 2  # 8 // full_attention_interval
    assert s.n_head_kv == 4 and s.key_len == 256 and s.val_len == 256
    assert s.n_ctx_train == 262144
    assert s.n_vocab == 1000
    assert s.quant == "Q8_K_P"  # filename wins over file_type (7 == Q8_0)
    assert s.params_b == pytest.approx(27.0)
    # KV per token = layers_with_kv * heads * (k+v) * 2 bytes
    assert s.kv_bytes_per_token() == 2 * 4 * 512 * 2
    assert s.kv_bytes(262144) == 2 * 4 * 512 * 2 * 262144


def test_dense_and_per_layer_heads(tmp_path):
    p = write_gguf(
        tmp_path / "dense-Q4_K_M.gguf",
        "llama",
        context_length=8192,
        block_count=4,
        embedding_length=1024,
        **{"attention.head_count": 8, "attention.head_count_kv": [2, 2, 0, 2]},
    )
    s = shape_from_kv(parse_local(p), p.name)
    assert s.n_attn_layers == 3  # layers with kv heads > 0
    assert s.n_head_kv == 2
    assert s.key_len == 128  # embedding / n_head
    assert s.quant == "Q4_K_M"


def test_max_ctx_for(hybrid_gguf):
    s = shape_from_kv(parse_local(hybrid_gguf), hybrid_gguf.name)
    per = s.kv_bytes_per_token()
    assert s.max_ctx_for(budget_bytes=per * 1000 + 10, model_bytes=10) == 1000
    assert s.max_ctx_for(budget_bytes=5, model_bytes=10) == 0


def test_truncated_header_raises(hybrid_gguf):
    buf = hybrid_gguf.read_bytes()[:64]
    with pytest.raises(NeedMoreData):
        parse_kv(buf)


def test_not_gguf():
    with pytest.raises(ValueError):
        parse_kv(b"NOPE" + b"\0" * 64)


@pytest.mark.parametrize(
    "name,quant",
    [
        ("Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf", "Q8_K_P"),
        ("Qwen3.5-122B-A10B-UD-IQ4_XS.gguf", "IQ4_XS"),
        ("gemma-4-31B-it-Q4_K_M.gguf", "Q4_K_M"),
        ("MiniMax-M2.7-UD-IQ4_XS-00003-of-00004.gguf", "IQ4_XS"),
        ("model-BF16-00001-of-00002.gguf", "BF16"),
        ("mmproj-BF16.gguf", "BF16"),
        ("weird-name.gguf", None),
    ],
)
def test_quant_from_filename(name, quant):
    assert quant_from_filename(name) == quant


def test_swa_default_pattern_gpt_oss(tmp_path):
    """gpt-oss GGUFs carry sliding_window=128 but no pattern key; llama.cpp alternates window/full (period 2)."""
    path = write_gguf(
        tmp_path / "gpt-oss-120b-F16.gguf",
        "gpt-oss",
        **{"general.name": "Gpt-Oss-120B", "general.file_type": 1},
        context_length=131072,
        block_count=36,
        embedding_length=2880,
        **{"attention.head_count": 64, "attention.head_count_kv": 8, "attention.key_length": 64, "attention.value_length": 64},
        **{"attention.sliding_window": 128},
    )
    s = shape_from_kv(parse_local(path), path.name)
    assert s.n_swa_layers == 18 and s.n_attn_layers == 18
    assert s.kv_bytes_per_token() == 18 * 8 * (64 + 64) * 2  # 36 KiB
    assert s.kv_bytes(131072) == s.kv_bytes_per_token() * 131072 + 18 * 8 * 128 * 128 * 2
