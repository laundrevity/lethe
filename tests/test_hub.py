from lethe.hub import SHARD_RE, HubGguf, match_quant


def test_shard_regex():
    m = SHARD_RE.match("Qwen3.6-35B-A3B-BF16-00002-of-00002.gguf")
    assert m and m["base"] == "Qwen3.6-35B-A3B-BF16" and m["idx"] == "00002" and m["n"] == "00002"
    assert SHARD_RE.match("Qwen3.6-35B-A3B-BF16.gguf") is None


def _g(base):
    return HubGguf(repo_id="x/y", base=base, files=[base], size=1)


def test_match_quant_is_token_bounded():
    ggufs = [_g("m-Q8_K_P.gguf"), _g("m-Q8_0.gguf"), _g("m-IQ4_XS.gguf"), _g("m-Q4_K_M.gguf"), _g("m-UD-Q4_K_XL.gguf")]
    assert [g.base for g in match_quant(ggufs, "Q8_K_P")] == ["m-Q8_K_P.gguf"]
    assert [g.base for g in match_quant(ggufs, "q4_k_m")] == ["m-Q4_K_M.gguf"]
    # "Q4_K" must not match Q4_K_M / Q4_K_XL partially
    assert match_quant(ggufs, "Q4_K") == []


def test_sidecar_detection():
    assert _g("mmproj-BF16.gguf").is_sidecar
    assert _g("Foo-FastMTP-32K.gguf").is_sidecar
    assert not _g("Foo-Q8_K_P.gguf").is_sidecar
