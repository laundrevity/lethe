import json
import os
import sys

import pytest

from lethe import shell, ssh


@pytest.fixture
def capture(monkeypatch):
    calls = []
    result = {"exit_code": 0, "stdout": "ok", "stderr": "", "timed_out": False, "truncated": False}

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(shell, "_run_bounded", run)
    return calls, result


@pytest.fixture
def fake_remote(tmp_path, monkeypatch):
    """Run SSH's remote command locally; never connect to a real machine."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    home = tmp_path / "remote home"
    home.mkdir()
    local = tmp_path / "local workspace"
    local.mkdir()
    executable = bindir / "ssh"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "args = sys.argv[1:]\n"
        "split = args.index('--')\n"
        "assert args[split + 1] == 'oswald'\n"
        "assert len(args) == split + 3\n"
        "env = {**os.environ, 'HOME': os.environ['LETHE_FAKE_REMOTE_HOME']}\n"
        "os.chdir(env['HOME'])\n"
        "os.execve('/bin/sh', ['sh', '-c', args[-1]], env)\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("LETHE_FAKE_REMOTE_HOME", str(home))
    return ssh.SSHRunner(local), home


@pytest.mark.parametrize("host", [
    "", "-oProxyCommand=evil", "user@-host", "a b", "a\nb", "a\0b", "a\tb", "a;id", "a$(id)",
    "a`id`", "a'", 'a"', "a|id", "a>file", "user@host@extra", "@host", "user@", "host:2222",
    "ssh://host", "[invalid]", "[::1", "::1]", "::1%bad;id", "x" * 513, None, 5,
])
def test_invalid_hosts_never_start_process(tmp_path, capture, host):
    with pytest.raises(ValueError, match="host"):
        ssh.SSHRunner(tmp_path).execute(host=host, python="print('ok')")
    assert capture[0] == []


@pytest.mark.parametrize("host", [
    "oswald", "user@oswald", "build_user@host-name.example", "user+label@host", "10.0.0.1",
    "::1", "[::1]", "user@[2001:db8::1]", "user@fe80::1%en0", "host_alias",
])
def test_aliases_users_and_ip_addresses_are_accepted(tmp_path, capture, host):
    ssh.SSHRunner(tmp_path).execute(host=host, bash="true")
    assert capture[0][0][0][0][-2] == host


@pytest.mark.parametrize("arguments", [
    {}, {"bash": "true", "python": "pass"}, {"bash": []}, {"bash": [""]}, {"bash": [" "]},
    {"bash": ["true", 3]}, {"bash": 3}, {"bash": ("true",)}, {"python": 3}, {"python": ["pass"]},
    {"bash": ""}, {"python": "\0"}, {"bash": ["true", "\0"]}, {"python": "a" * 8001},
    {"python": "é" * 4001}, {"bash": ["x" * 4000, "x" * 4001]},
    {"bash": "true", "timeout_seconds": 0}, {"bash": "true", "timeout_seconds": 601},
    {"bash": "true", "timeout_seconds": True}, {"bash": "true", "timeout_seconds": 2.5},
    {"bash": "true", "cwd": ""}, {"bash": "true", "cwd": "\0"}, {"bash": "true", "cwd": 3},
    {"bash": "true", "cwd": "~another"},
])
def test_invalid_source_directory_and_limits_never_start_process(tmp_path, capture, arguments):
    with pytest.raises(ValueError):
        ssh.SSHRunner(tmp_path).execute(host="oswald", **arguments)
    assert capture[0] == []


def test_source_stays_on_stdin_and_ssh_is_noninteractive_and_nonpersistent(tmp_path, monkeypatch, capture):
    monkeypatch.setenv("LETHE_TEST_CREDENTIAL", "inherited")
    source = 'import json\nprint(json.dumps({"quote": "a\\\"b", "text": "café"}))\n'
    runner = ssh.SSHRunner(tmp_path)
    assert json.loads(runner.run(host="oswald", python=source)) == capture[1]
    (argv, stdin, cwd, env, timeout), kwargs = capture[0][0]
    assert stdin == source.encode("utf-8")
    assert all(source not in arg and "import json" not in arg for arg in argv)
    assert argv[:4] == ["ssh", "-T", "-S", "none"]
    assert argv[-3:-1] == ["--", "oswald"]
    options = set(argv[5:-3:2])
    assert {
        "BatchMode=yes", "StrictHostKeyChecking=yes", "CheckHostIP=no", "UpdateHostKeys=no",
        "ControlMaster=no", "ControlPersist=no", "ClearAllForwardings=yes", "ForwardAgent=no",
        "ForwardX11=no", "Tunnel=no", "PermitLocalCommand=no", "AddKeysToAgent=no",
        "ForkAfterAuthentication=no", "SessionType=default", "StdinNull=no", "RemoteCommand=none",
    } <= options
    assert "-F" not in argv  # aliases and identities still come from the user's SSH config
    assert cwd == tmp_path.resolve() and timeout == 120
    assert env["LETHE_TEST_CREDENTIAL"] == "inherited" and env["HISTFILE"] == "/dev/null"
    assert kwargs == {"stop_on_output_limit": False}
    assert list(tmp_path.iterdir()) == []


def test_array_size_counts_user_source_not_wrapper_bytes(tmp_path, capture):
    ssh.SSHRunner(tmp_path).execute(host="oswald", bash=["x" * 4000, "y" * 4000], timeout_seconds=600)
    (argv, source, cwd, env, timeout), kwargs = capture[0][0]
    assert len(source) > 8000 and timeout == 600


def test_python_quotes_newlines_unicode_and_stderr_round_trip(fake_remote):
    runner, home = fake_remote
    payload = 'a"b\'c\\d\nsecond line café 👋\n'
    result = runner.execute(host="oswald", python=(
        "import sys\n"
        f"sys.stdout.write({payload!r})\n"
        "sys.stderr.write('error detail')\n"
        "sys.exit(7)\n"
    ))
    assert result == {
        "stdout": payload, "stderr": "error detail", "exit_code": 7, "timed_out": False, "truncated": False,
    }
    assert list(home.iterdir()) == []


@pytest.mark.parametrize("directory", ["dir ' \" $(touch unexpected); &", "-", "-P", "line\nbreak", "space dir"])
def test_remote_cwd_is_literal_and_not_local_or_shell_interpolation(fake_remote, directory):
    runner, home = fake_remote
    expected = home / directory
    expected.mkdir()
    result = runner.execute(host="oswald", cwd=directory, python="import os; print(os.getcwd())")
    assert result["exit_code"] == 0 and result["stdout"] == str(expected) + "\n"
    assert not (home / "unexpected").exists()


def test_remote_home_expansion_and_missing_directory(fake_remote):
    runner, home = fake_remote
    child = home / "sub ' dir"
    child.mkdir()
    for cwd, expected in [(None, home), ("~", home), ("~/sub ' dir", child), (str(child), child)]:
        result = runner.execute(host="oswald", cwd=cwd, bash="pwd")
        assert result["exit_code"] == 0 and result["stdout"] == str(expected) + "\n"
    result = runner.execute(host="oswald", cwd="missing", bash="printf should-not-run")
    assert result["exit_code"] != 0 and result["stdout"] == ""


def test_bash_list_shares_variables_and_cwd_and_handles_comments_and_heredocs(fake_remote):
    runner, home = fake_remote
    (home / "subdir").mkdir()
    result = runner.execute(host="oswald", bash=[
        "value='hello world'\n# trailing comment",
        "cd subdir",
        'printf "%s:%s\\n" "$value" "${PWD##*/}" # another comment',
        "cat <<'END'\nquotes ' \" and $literal\nEND",
    ])
    assert result["exit_code"] == 0
    assert result["stdout"] == 'hello world:subdir\nquotes \' " and $literal\n'


@pytest.mark.parametrize("bash", [
    ["printf before", "false", "printf after"],
    "printf before\nfalse\nprintf after\n",
])
def test_bash_stops_on_failed_command(fake_remote, bash):
    runner, home = fake_remote
    result = runner.execute(host="oswald", bash=bash)
    assert result["exit_code"] == 1 and result["stdout"] == "before"


@pytest.mark.parametrize("commands,expected", [
    (["false; printf recovered"], "recovered"),
    (["false; printf first", "false; printf last"], "firstlast"),
    (["false; printf first", "false", "printf skipped"], "first"),
])
def test_bash_list_uses_each_compound_commands_final_status(fake_remote, commands, expected):
    runner, home = fake_remote
    result = runner.execute(host="oswald", bash=commands)
    assert result["stdout"] == expected
    assert result["exit_code"] == (1 if "false" in commands else 0)


def test_python_imports_do_not_create_bytecode(fake_remote):
    runner, home = fake_remote
    (home / "helper.py").write_text("value = 'loaded'\n")
    result = runner.execute(host="oswald", python="import helper; print(helper.value)")
    assert result["exit_code"] == 0 and result["stdout"] == "loaded\n"
    assert {path.name for path in home.iterdir()} == {"helper.py"}


def test_output_is_bounded_but_command_finishes_and_timeout_works(fake_remote):
    runner, home = fake_remote
    result = runner.execute(host="oswald", python="import sys; sys.stdout.write('x' * 40000); sys.exit(3)")
    assert result["exit_code"] == 3 and result["truncated"] and not result["timed_out"]
    assert len(result["stdout"].encode()) == shell.MAX_OUTPUT_BYTES
    result = runner.execute(host="oswald", python="import time; time.sleep(30)", timeout_seconds=1)
    assert result["timed_out"]


def test_preview_validates_and_shows_complete_source_without_connecting(tmp_path, capture):
    runner = ssh.SSHRunner(tmp_path)
    source = "import os\nprint('literal source')\n"
    preview = runner.preview(host="oswald", python=source, cwd="~/subdir", timeout_seconds=45)
    header, _, body = preview.partition("\n")
    assert "oswald" in header and "python" in header and "~/subdir" in header and "45s" in header
    assert body == source
    preview = runner.preview(host="oswald", bash=["pwd", "printf second"])
    assert "2 commands, &&" in preview and "pwd" in preview and "printf second" in preview
    with pytest.raises(ValueError, match="exactly one"):
        runner.preview(host="oswald")
    assert capture[0] == []
