from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/verify_git_repository.py"
REF = "refs/heads/codex/position-identity-v1"


def _load():
    spec = importlib.util.spec_from_file_location("git_repository_binding", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_git(tmp_path: Path, body: str) -> tuple[Path, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "git"
    path.write_text("#!/bin/bash\nset -euo pipefail\n" + body)
    path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_public_ref_query_returns_canonical_binding(tmp_path):
    module = _load()
    commit = "a" * 40
    git, digest = _fake_git(
        tmp_path,
        f"[[ \"$*\" == \"ls-remote --refs {module.REPOSITORY_URL} {REF}\" ]]\n"
        f"printf '%s\\t%s\\n' '{commit}' '{REF}'\n",
    )
    binding = module.query_exact_repository(
        git=git, git_sha256=digest, expected_commit_sha=commit
    )
    assert binding["repository_url"] == module.REPOSITORY_URL
    assert binding["repository_ref"] == REF
    assert binding["commit_sha"] == commit
    assert binding["repository_identity_sha256"] == module.repository_identity_sha256(
        commit
    )
    assert module.validate_repository_binding(
        binding, expected_commit_sha=commit, expected_git_sha256=digest
    ) == binding


def test_query_ignores_ambient_git_config_and_uses_fixed_nonrepo_cwd(
    tmp_path, monkeypatch
):
    module = _load()
    commit = "a" * 40
    malicious = tmp_path / "malicious.gitconfig"
    malicious.write_text(
        '[url "file:///tmp/fake.git"]\n'
        f"\tinsteadOf = {module.REPOSITORY_URL}\n"
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(malicious))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv(
        "GIT_CONFIG_KEY_0",
        "url.file:///tmp/fake.git.insteadOf",
    )
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", module.REPOSITORY_URL)
    git, digest = _fake_git(
        tmp_path / "fakebin",
        '[[ "$PWD" == "/" ]]\n'
        '[[ "${GIT_CONFIG_NOSYSTEM:-}" == "1" ]]\n'
        '[[ "${GIT_CONFIG_GLOBAL:-}" == "/dev/null" ]]\n'
        '[[ "${GIT_CONFIG_COUNT:-}" == "0" ]]\n'
        '[[ "${GIT_ALLOW_PROTOCOL:-}" == "https" ]]\n'
        '[[ -z "${HOME:-}" ]]\n'
        '[[ -z "${GIT_CONFIG_KEY_0:-}" && -z "${GIT_CONFIG_VALUE_0:-}" ]]\n'
        f"printf '%s\\t%s\\n' '{commit}' '{REF}'\n",
    )
    binding = module.query_exact_repository(
        git=git,
        git_sha256=digest,
        expected_commit_sha=commit,
    )
    assert binding["commit_sha"] == commit


@pytest.mark.parametrize(
    "body,match",
    [
        ("exit 17\n", "query failed"),
        (f"printf '%s\\t%s\\n' '{'b' * 40}' '{REF}'\n", "does not advertise"),
        (f"printf '%s\\t%s\\n%s\\t%s\\n' '{'a' * 40}' '{REF}' '{'a' * 40}' 'refs/heads/other'\n", "does not advertise"),
    ],
)
def test_failed_unadvertised_or_extra_ref_output_is_rejected(tmp_path, body, match):
    module = _load()
    git, digest = _fake_git(tmp_path, body)
    with pytest.raises(ValueError, match=match):
        module.query_exact_repository(
            git=git, git_sha256=digest, expected_commit_sha="a" * 40
        )


def test_query_is_timeout_and_output_bounded(tmp_path, monkeypatch):
    module = _load()
    slow, slow_digest = _fake_git(tmp_path / "slow", "sleep 2\n")
    monkeypatch.setattr(module, "QUERY_TIMEOUT_SECONDS", 0.05)
    with pytest.raises(ValueError, match="timed out"):
        module.query_exact_repository(
            git=slow, git_sha256=slow_digest, expected_commit_sha="a" * 40
        )

    noisy, noisy_digest = _fake_git(
        tmp_path / "noisy",
        "head -c 70000 /dev/zero | tr '\\000' x\n",
    )
    with pytest.raises(ValueError, match="byte ceiling"):
        module.query_exact_repository(
            git=noisy, git_sha256=noisy_digest, expected_commit_sha="a" * 40
        )


def test_git_digest_and_symlink_substitution_fail_before_query(tmp_path):
    module = _load()
    git, digest = _fake_git(tmp_path, "exit 99\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        module.query_exact_repository(
            git=git, git_sha256="0" * 64, expected_commit_sha="a" * 40
        )
    alias = tmp_path / "git-alias"
    alias.symlink_to(git)
    with pytest.raises(ValueError, match="no-follow"):
        module.query_exact_repository(
            git=alias, git_sha256=digest, expected_commit_sha="a" * 40
        )


def test_fifo_git_is_rejected_without_blocking(tmp_path):
    fifo = tmp_path / "git"
    os.mkfifo(fifo)
    code = (
        "import importlib.util,pathlib,sys\n"
        "spec=importlib.util.spec_from_file_location('fifo_git_repository',sys.argv[1])\n"
        "module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)\n"
        "module.query_exact_repository(git=pathlib.Path(sys.argv[2]),"
        "git_sha256='0'*64,expected_commit_sha='a'*40)\n"
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(SCRIPT), str(fifo)],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert completed.returncode != 0
    assert "bounded executable file" in completed.stderr
