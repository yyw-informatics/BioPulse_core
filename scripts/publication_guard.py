from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


BLOCKED_PREFIXES = (
    ".agents/",
    ".codex/",
    ".cursor/",
    ".idea/",
    ".mypy_cache/",
    ".nox/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".tox/",
    ".venv/",
    ".vscode/",
    "__pypackages__/",
    "benchmark_packs/",
    "data/",
    "dev-docs/",
    "external/",
    "htmlcov/",
    "logs/",
    "op_cache/",
    "runs/",
    "temp/",
    "tmp/",
    "venv/",
)

BLOCKED_NAMES = {
    ".coverage",
    ".env",
    ".envrc",
    ".python-version",
    ".python_history",
    ".tool-versions",
    "Thumbs.db",
    "coverage.xml",
    "environment.yml",
    "environment.yaml",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "junit.xml",
}

BLOCKED_SUFFIXES = (
    ".age",
    ".asc",
    ".bam",
    ".bed",
    ".cer",
    ".crt",
    ".csv",
    ".db",
    ".egg-info/PKG-INFO",
    ".env",
    ".fastq",
    ".feather",
    ".fq",
    ".gpg",
    ".gtf",
    ".gz",
    ".h5",
    ".h5ad",
    ".h5mu",
    ".hdf5",
    ".jks",
    ".jsonl",
    ".kdbx",
    ".keystore",
    ".key",
    ".loom",
    ".log",
    ".mtx",
    ".npy",
    ".npz",
    ".p12",
    ".p8",
    ".parquet",
    ".pem",
    ".pfx",
    ".pub",
    ".pyc",
    ".rda",
    ".rds",
    ".sam",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".tgz",
    ".token",
    ".tsv",
    ".whl",
    ".zip",
)

TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".R",
    ".r",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

SECRET_PATTERNS = (
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("aws access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}\b")),
    ("github fine-grained token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("openai-style api key", re.compile(r"\bsk-[A-Za-z0-9_-]{32,}\b")),
    (
        "assigned secret-like value",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|password|access[_-]?token|auth[_-]?token)\b"
            r"\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"
        ),
    ),
)

LOCAL_PATH_PATTERNS = (
    ("mac user path", re.compile(r"/" r"Users/[^/\s]+/")),
    ("mac private temp path", re.compile(r"/" r"private/(?:tmp|var)/")),
    ("windows user path", re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+")),
)


def publication_candidates() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted({Path(line) for line in result.stdout.splitlines() if line.strip()})


def path_violations(path: Path) -> list[str]:
    normalized = path.as_posix()
    name = path.name
    violations: list[str] = []

    if normalized.endswith(".md") and normalized != "README.md":
        violations.append("extra tracked Markdown file")
    if name in BLOCKED_NAMES:
        violations.append("blocked env/local file name")
    if any(normalized == prefix.rstrip("/") or normalized.startswith(prefix) for prefix in BLOCKED_PREFIXES):
        violations.append("blocked generated/local path")
    if any(normalized.endswith(suffix) for suffix in BLOCKED_SUFFIXES):
        violations.append("blocked generated/data/secret suffix")
    if ".egg-info/" in normalized or "__pycache__/" in normalized:
        violations.append("blocked build/cache artifact")
    return violations


def read_text_for_scan(path: Path) -> str | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    if stat.st_size > 1_000_000 or path.suffix not in TEXT_SUFFIXES:
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data:
        return None
    return data.decode("utf-8", errors="ignore")


def content_violations(path: Path) -> list[str]:
    text = read_text_for_scan(path)
    if text is None:
        return []
    violations: list[str] = []
    for label, pattern in LOCAL_PATH_PATTERNS:
        if pattern.search(text):
            violations.append(label)
    for label, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            violations.append(label)
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail if Git-visible files are unsafe for public publication.")
    parser.parse_args()

    failures: list[str] = []
    for path in publication_candidates():
        if not path.exists():
            continue
        for reason in path_violations(path):
            failures.append(f"{path}: {reason}")
        for reason in content_violations(path):
            failures.append(f"{path}: {reason}")

    if failures:
        print("Publication guard failed:", file=sys.stderr)
        for item in failures:
            print(f"- {item}", file=sys.stderr)
        return 1

    print("Publication guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
