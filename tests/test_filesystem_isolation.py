from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_filesystem_isolation.py"


def _load():
    spec = importlib.util.spec_from_file_location("filesystem_isolation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_same_device_fails_closed_before_any_write(tmp_path):
    module = _load()
    work = tmp_path / "work"
    artifacts = tmp_path / "artifacts"
    work.mkdir()
    artifacts.mkdir()

    with pytest.raises(ValueError, match="different filesystem"):
        module.require_distinct_filesystems(work, artifacts)
    assert list(work.iterdir()) == []
    assert list(artifacts.iterdir()) == []


def test_descriptor_device_comparison_and_evidence(monkeypatch, tmp_path):
    module = _load()
    work = tmp_path / "work"
    artifacts = tmp_path / "artifacts"
    work.mkdir()
    artifacts.mkdir()
    real_fstat = os.fstat

    class DifferentDevice:
        def __init__(self, original, device):
            for name in dir(original):
                if name.startswith("st_"):
                    try:
                        setattr(self, name, getattr(original, name))
                    except AttributeError:
                        pass
            self.st_dev = device

    artifact_inode = artifacts.stat().st_ino

    def fake_fstat(fd):
        value = real_fstat(fd)
        if value.st_ino == artifact_inode:
            return DifferentDevice(value, value.st_dev + 1)
        return value

    monkeypatch.setattr(module.os, "fstat", fake_fstat)
    report = module.require_distinct_filesystems(work, artifacts)
    assert report["work_device"] != report["artifact_device"]
    assert report["work_root"] == str(work)
    assert report["artifact_root"] == str(artifacts)


def test_component_wise_nofollow_rejects_symlink_and_unnormalized_path(tmp_path):
    module = _load()
    physical = tmp_path / "physical"
    physical.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(physical, target_is_directory=True)

    with pytest.raises(OSError):
        fd = module.open_physical_directory(alias, where="test root")
        os.close(fd)
    with pytest.raises(ValueError, match="normalized absolute"):
        module.open_physical_directory(
            Path(f"{tmp_path}/physical/../physical"), where="test root"
        )
