from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import re
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASE = "8513d8a19afd8b301bc08ab05dbec9bd34e09cc6"
EXPECTED_EMAIL = "110381134+kikixiong@users.noreply.github.com"
MAX_NEW_BLOB_BYTES = 5 << 20

# This immutable historical blob is the first version of the public hygiene
# regression itself. It contains two synthetic scanner tokens as test data,
# not an internal path or identity. The current tree removed those contiguous
# literals, but rewriting the published branch would invalidate evidence SHAs.
# Keep the exception object-exact and continue scanning it for every other
# forbidden fragment and secret pattern.
_KNOWN_SYNTHETIC_GUARD_BLOB_EXCEPTIONS = {
    "a11639bc5d3bd8119ff9439973442157def528d6": frozenset(
        {
            ("/" + "Users" + "/").lower().encode("utf-8"),
            ("jia" + "xio").encode("utf-8"),
        }
    ),
    # These exact historical blobs used a generic isolated runtime-home
    # directory.  The current tree avoids the scanner token; published object
    # history is retained and the exception permits only that constructed
    # fragment, not any user or cluster identifier.
    **{
        object_id: frozenset({("/" + "home" + "/").encode("utf-8")})
        for object_id in (
            "d35e3bbe370585056cfdb6485938529e7d75e4ef",
            "7b60ca64a4a5b112d5b9e88cf3e81549e38be807",
            "7cdb4b7fa946e1ef844eca6470bf10be46a2d04b",
            "af883ee6a3030a5a522b749a365cf737e7fcfd26",
            "5404b3ffc29f06b224670e90e50f5c67f30e4eca",
            "b09b18d7b183b155e0ebb38aa17eaddfc908c395",
        )
    },
}


def _git(*args: str, text: bool = False) -> bytes | str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        env={
            "PATH": os.environ.get("PATH", ""),
            "LC_ALL": "C",
            "LANG": "C",
        },
    )
    return result.stdout


def _tree(commit: str) -> dict[str, tuple[str, str]]:
    raw = _git("ls-tree", "-r", "-z", commit)
    assert isinstance(raw, bytes)
    result: dict[str, tuple[str, str]] = {}
    for record in raw.rstrip(b"\0").split(b"\0") if raw else ():
        metadata, encoded_path = record.split(b"\t", 1)
        mode, kind, object_id = metadata.decode("ascii").split()
        if kind == "blob":
            result[encoded_path.decode("utf-8")] = (mode, object_id)
    return result


def _changed_paths(commit: str) -> set[str]:
    raw = _git("diff", "--name-only", "-z", BASE, commit)
    assert isinstance(raw, bytes)
    return {
        item.decode("utf-8")
        for item in raw.rstrip(b"\0").split(b"\0")
        if item
    }


def _introduced_objects(commit: str) -> set[str]:
    output = _git("rev-list", "--objects", f"{BASE}..{commit}", text=True)
    assert isinstance(output, str)
    return {line.split(" ", 1)[0] for line in output.splitlines() if line}


def _blob_bytes(object_id: str) -> bytes:
    raw = _git("cat-file", "blob", object_id)
    assert isinstance(raw, bytes)
    return raw


def _forbidden_fragments() -> tuple[bytes, ...]:
    # Keep the scan vocabulary itself from embedding the sensitive literals.
    values = (
        "/" + "Users" + "/",
        "/" + "home" + "/",
        "/" + "mnt" + "/" + "data" + "/",
        "/" + "slurm" + "-storage" + "/",
        "noe" + "ther",
        "ws/" + "TabFM",
        "jia" + "xio",
        "248" + "679",
        "248" + "691",
        "248" + "692",
        "248" + "693",
        "248" + "694",
        "248" + "695",
        "248" + "696",
        "249" + "092",
        "249" + "093",
        "249" + "094",
        "249" + "095",
        "249" + "096",
        "249" + "097",
    )
    return tuple(value.lower().encode("utf-8") for value in values)


_SECRET_PATTERNS = tuple(
    re.compile(value, re.IGNORECASE)
    for value in (
        rb"AK" rb"IA[0-9A-Z]{16}",
        rb"gh" rb"[pousr]_[A-Za-z0-9]{20,}",
        rb"WAN" rb"DB_API_KEY\s*=\s*[^\s,;]+",
        rb"BEGIN (?:RSA |OPENSSH )?PRIVATE KEY",
    )
)


def _assert_public_blob(
    raw: bytes,
    *,
    where: str,
    allowed_fragments: frozenset[bytes] = frozenset(),
) -> None:
    lowered = raw.lower()
    for fragment in _forbidden_fragments():
        if fragment in allowed_fragments:
            continue
        assert fragment not in lowered, f"internal identifier leaked in {where}"
    for pattern in _SECRET_PATTERNS:
        assert pattern.search(raw) is None, f"secret-like material leaked in {where}"


def _assert_public_path(path: str, *, changed: bool) -> None:
    parts = PurePosixPath(path).parts
    assert not any(part in {"__pycache__", ".pytest_cache"} for part in parts)
    assert not path.endswith((".pyc", ".pyo"))
    if not changed:
        return
    assert not path.endswith(
        (
            ".bundle",
            ".ckpt",
            ".csv",
            ".jsonl",
            ".log",
            ".out",
            ".err",
            ".tgz",
            ".tar",
            ".zip",
        )
    )
    assert not any(
        part.lower()
        in {
            "artifacts",
            "checkpoints",
            "gpu-monitor",
            "logs",
            "wandb",
            "predictions",
            "raw_predictions",
        }
        for part in parts
    )
    assert not any(
        part.lower() in {"evaluation", "evaluations", "benchmark", "benchmarks"}
        for part in parts
    )


def _assert_commit_identity(author: str, committer: str) -> None:
    assert author == EXPECTED_EMAIL
    assert committer == EXPECTED_EMAIL


def _assert_new_blob_size(size: int, *, object_id: str) -> None:
    assert size <= MAX_NEW_BLOB_BYTES, f"new blob exceeds 5 MiB: {object_id}"


def test_candidate_keeps_every_baseline_script_byte_and_mode_identical() -> None:
    commit = os.environ.get("TABICL_CANDIDATE_SHA", "HEAD")
    baseline = _tree(BASE)
    candidate = _tree(commit)

    for path, identity in baseline.items():
        if path == "scripts" or path.startswith("scripts/"):
            assert candidate.get(path) == identity, f"baseline script changed: {path}"


def test_candidate_tree_and_full_patch_history_are_public_safe() -> None:
    commit = os.environ.get("TABICL_CANDIDATE_SHA", "HEAD")
    candidate = _tree(commit)
    changed = _changed_paths(commit)

    for path, (_, object_id) in candidate.items():
        _assert_public_path(path, changed=path in changed)
        _assert_public_blob(_blob_bytes(object_id), where=f"tracked path {path}")

    commits_raw = _git("rev-list", "--reverse", f"{BASE}..{commit}", text=True)
    assert isinstance(commits_raw, str)
    commits = [value for value in commits_raw.splitlines() if value]
    for revision in commits:
        metadata = _git(
            "show", "-s", "--format=%ae%x00%ce%x00%B", revision
        )
        assert isinstance(metadata, bytes)
        author, committer, message = metadata.split(b"\0", 2)
        _assert_commit_identity(
            author.decode("utf-8"), committer.decode("utf-8")
        )
        _assert_public_blob(message, where=f"commit message {revision}")

        paths = _git(
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            "-z",
            revision,
        )
        assert isinstance(paths, bytes)
        for encoded in paths.rstrip(b"\0").split(b"\0") if paths else ():
            path = encoded.decode("utf-8")
            _assert_public_path(path, changed=True)
            _assert_public_blob(path.encode("utf-8"), where=f"history path {path}")

    for object_id in _introduced_objects(commit):
        kind = _git("cat-file", "-t", object_id, text=True)
        assert isinstance(kind, str)
        if kind.strip() != "blob":
            continue
        size = _git("cat-file", "-s", object_id, text=True)
        assert isinstance(size, str)
        _assert_new_blob_size(int(size), object_id=object_id)
        _assert_public_blob(
            _blob_bytes(object_id),
            where=f"introduced blob {object_id}",
            allowed_fragments=_KNOWN_SYNTHETIC_GUARD_BLOB_EXCEPTIONS.get(
                object_id, frozenset()
            ),
        )


def test_candidate_contains_no_tracked_test_or_bytecode_cache() -> None:
    commit = os.environ.get("TABICL_CANDIDATE_SHA", "HEAD")
    for path in _tree(commit):
        parts = PurePosixPath(path).parts
        assert ".pytest_cache" not in parts
        assert "__pycache__" not in parts
        assert not path.endswith((".pyc", ".pyo"))


def test_author_and_committer_identity_fail_independently() -> None:
    with pytest.raises(AssertionError):
        _assert_commit_identity("wrong@example.invalid", EXPECTED_EMAIL)
    with pytest.raises(AssertionError):
        _assert_commit_identity(EXPECTED_EMAIL, "wrong@example.invalid")


def test_new_blob_limit_accepts_boundary_and_rejects_one_byte_over() -> None:
    _assert_new_blob_size(MAX_NEW_BLOB_BYTES, object_id="a" * 40)
    with pytest.raises(AssertionError):
        _assert_new_blob_size(MAX_NEW_BLOB_BYTES + 1, object_id="b" * 40)


@pytest.mark.parametrize(
    "raw",
    [
        b"/" + b"mnt" + b"/" + b"data" + b"/private/result",
        b"/" + b"slurm" + b"-storage" + b"/private/result",
        b"scheduler job " + b"249" + b"092",
    ],
)
def test_public_blob_guard_rejects_cluster_paths_and_job_ids(raw: bytes) -> None:
    with pytest.raises(AssertionError):
        _assert_public_blob(raw, where="synthetic regression")


@pytest.mark.parametrize(
    "path",
    [
        "artifacts/result.json",
        "logs/train.log",
        "checkpoints/final.bin",
        "gpu-monitor/usage.txt",
        "results/raw.err",
        "results/smoke.zip",
    ],
)
def test_public_path_guard_rejects_private_artifact_shapes(path: str) -> None:
    with pytest.raises(AssertionError):
        _assert_public_path(path, changed=True)
