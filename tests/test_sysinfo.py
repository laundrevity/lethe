from types import SimpleNamespace

import pytest

from lethe import paths, sysinfo


@pytest.mark.parametrize('device', ['MTL0: Apple M5 Max', 'Vulkan0: AMD Radeon 8060S', 'ROCm0: AMD Radeon 8060S'])
def test_backend_memory_reports(monkeypatch, tmp_path, device):
    exe = tmp_path / 'llama-server'
    exe.touch()
    monkeypatch.setattr(paths, 'llama_bin', lambda name: exe)
    monkeypatch.setattr(sysinfo.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=f'{device} (102400 MiB, 81920 MiB free)', stderr=''))
    assert sysinfo._gpu_device_memory() == (100 * sysinfo.GiB, 80 * sysinfo.GiB)


def test_linux_budget_respects_host_and_gpu_free(monkeypatch):
    g = sysinfo.GiB
    monkeypatch.setattr(sysinfo.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(sysinfo.psutil, 'virtual_memory', lambda: SimpleNamespace(available=70 * g))
    m = sysinfo.Machine(128 * g, 120 * g, 32, 'Strix Halo', False, 100 * g)
    assert m.available_now() == 66 * g
    m.gpu_free = 50 * g
    assert m.available_now() == 46 * g
    assert m.available_now(excluding_rss=10 * g) == 56 * g


def test_linux_no_gpu_does_not_assume_all_ram(monkeypatch):
    monkeypatch.setattr(sysinfo.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(sysinfo, '_gpu_device_memory', lambda: None)
    assert sysinfo.machine().usable_for_model == 0


@pytest.mark.parametrize('backend,build', [('metal', 'build'), ('vulkan', 'build-vulkan'), ('hip', 'build-hip')])
def test_backend_binary_selection(monkeypatch, backend, build):
    monkeypatch.setenv('LETHE_BACKEND', backend)
    assert paths.llama_bin('llama-server') == paths.llama_dir() / build / 'bin' / 'llama-server'


@pytest.mark.parametrize('platform,build', [('linux', 'build-hip'), ('darwin', 'build')])
def test_platform_default_backend(monkeypatch, platform, build):
    monkeypatch.delenv('LETHE_BACKEND', raising=False)
    monkeypatch.setattr(paths.sys, 'platform', platform)
    assert paths.llama_bin('llama-server') == paths.llama_dir() / build / 'bin' / 'llama-server'
