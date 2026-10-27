"""Native tool-schema regression: no weights, server, SSH, or conversation files.

The Python tool tests cannot catch a schema that llama.cpp turns into a grammar
with no arguments. Compile a small helper when the local native build is present
and send templates, the actual advertised tool, and synthetic calls over stdin.
"""

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from lethe import config, ggufmeta, paths, spec, tools
from lethe.shell import FullShell, ShellMode


@pytest.fixture(scope="module")
def native_schema_probe(tmp_path_factory):
    compiler = shutil.which("c++")
    source = Path(__file__).parent / "helpers" / "tool_schema.cpp"
    vendor = paths.llama_dir()
    binary_dir = paths.llama_bin("llama-server").parent
    if not compiler or not any((binary_dir / f"libllama-common.{ext}").exists() for ext in ("dylib", "so")):
        pytest.skip("native schema regression requires a built llama.cpp and C++ compiler")
    executable = tmp_path_factory.mktemp("native-schema") / "tool-schema"
    command = [compiler, "-std=c++17", str(source), "-o", str(executable)]
    for directory in (".", "common", "include", "ggml/include", "src"):
        command += ["-I", str(vendor / directory)]
    command += [
        "-L", str(binary_dir), "-l", "llama-common", "-l", "llama", "-l", "ggml-base", f"-Wl,-rpath,{binary_dir}",
    ]
    compiled = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert compiled.returncode == 0, compiled.stderr

    def probe(template, tool, cases):
        result = subprocess.run(
            [str(executable)], input=json.dumps({"template": template, "tool": tool, "cases": cases}),
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    return probe


@pytest.fixture(params=["vendored", "default-model"])
def qwen_template(request):
    if request.param == "vendored":
        return (paths.llama_dir() / "models" / "templates" / "Qwen3.5-4B.jinja").read_text()
    model_name = config.default_model()
    if not model_name:
        pytest.skip("no configured default model")
    try:
        model = spec.resolve(model_name)
    except (spec.ModelNotFound, OSError):
        pytest.skip("default model is not downloaded")
    # Metadata only, with a fixed bound: never read the weight tensors.
    with model.path.open("rb") as source:
        try:
            metadata = ggufmeta.parse_kv(source.read(16 << 20))
        except ggufmeta.NeedMoreData:
            pytest.skip("default model metadata exceeds the 16 MiB inspection limit")
    template = metadata.get("tokenizer.chat_template", "")
    if not isinstance(template, str) or "<parameter=" not in template:
        pytest.skip("configured default does not use Qwen-style tool parameters")
    return template


def _call(tool_name="run_ssh", **arguments):
    fields = "".join(
        f"<parameter={name}>\n{value if isinstance(value, str) else json.dumps(value)}\n</parameter>\n"
        for name, value in arguments.items()
    )
    return f"<tool_call>\n<function={tool_name}>\n{fields}</function>\n</tool_call>"


def test_real_qwen_grammar_requires_host_and_preserves_payloads(native_schema_probe, qwen_template, tmp_path, monkeypatch):
    monkeypatch.setattr(FullShell, "check_available", lambda self: None)
    tool = tools.build_registry(["shell"], shell_dir=tmp_path, shell_mode=ShellMode.FULL)["run_ssh"].spec()
    arguments = [
        {"host": "oswald", "bash": ["cd ~/project", "printf '%s\\n' \"quoted value\""]},
        {"host": "oswald", "bash": ["for value in 'literal \\\\ text'; do\n  printf '%s\\n' \"$value\"\ndone"]},
        {"host": "oswald", "python": 'import json\nprint(json.dumps({"message": "café 👋", "path": "a\\\\b"}))\n'},
    ]
    cases = [_call(), *(_call(**argument) for argument in arguments)]
    fixed = native_schema_probe(qwen_template, tool, cases)
    assert set(fixed["parameters"]) == set(tool["function"]["parameters"]["properties"])
    assert not fixed["cases"][0]["grammar_accepts"]
    for parsed, expected in zip(fixed["cases"][1:], arguments):
        assert parsed["grammar_accepts"], parsed
        assert parsed["calls"] == [{"name": "run_ssh", "arguments": expected}]
    strings = native_schema_probe(qwen_template, tool, [_call(host="oswald", bash="pwd; uname -s")])
    assert not strings["cases"][0]["grammar_accepts"]

    # Reintroduce the original failure: root oneOf becomes a non-object schema,
    # so llama.cpp's foreach_parameter sees none of the sibling properties.
    broken = copy.deepcopy(tool)
    broken["function"]["parameters"]["oneOf"] = [{"required": ["bash"]}, {"required": ["python"]}]
    regression = native_schema_probe(qwen_template, broken, cases)
    assert regression["parameters"] == []
    assert regression["cases"][0]["grammar_accepts"]
    assert regression["cases"][0]["calls"] == [{"name": "run_ssh", "arguments": {}}]
    assert all(not case["grammar_accepts"] for case in regression["cases"][1:])


@pytest.mark.parametrize("name,arguments", [
    ("run_python", {"code": 'import json\nprint(json.dumps({"message": "café", "path": "a\\\\b"}))\n'}),
    ("run_bash", {"commands": ["cd ~/project", "printf '%s\\n' \"quoted value\""]}),
])
def test_local_code_grammar_preserves_native_source(native_schema_probe, qwen_template, tmp_path, monkeypatch, name, arguments):
    monkeypatch.setattr(FullShell, "check_available", lambda self: None)
    tool = tools.build_registry(["shell"], shell_dir=tmp_path, shell_mode=ShellMode.FULL)[name].spec()
    parsed = native_schema_probe(qwen_template, tool, [_call(name), _call(name, **arguments)])
    assert not parsed["cases"][0]["grammar_accepts"]
    assert parsed["cases"][1]["grammar_accepts"]
    assert parsed["cases"][1]["calls"] == [{"name": name, "arguments": arguments}]
