"""Секреты и служебные файлы не попадают в Git; Docker не публикует лишнего."""

import shutil
import subprocess

import pytest
import yaml

from experimentlab.config import PROJECT_ROOT

WEAK_VALUES = ("lab_" + "password", "change_" + "me")


def _tracked() -> list[str]:
    if shutil.which("git") is None or not (PROJECT_ROOT / ".git").exists():
        pytest.skip("нет git-репозитория")
    out = subprocess.run(
        ["git", "ls-files"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [line for line in out.splitlines() if line]


def test_secret_files_are_not_tracked():
    tracked = _tracked()
    assert ".env" not in tracked
    assert not any(p.startswith("data/simulation/") for p in tracked)
    assert ".env.example" in tracked


def test_gitignore_covers_secrets_and_generated_data():
    lines = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for required in (".env", "data/simulation/", "logs/", ".venv/"):
        assert required in lines, required


def test_no_weak_default_secrets_in_tracked_files():
    this_file = "tests/test_repo_hygiene.py"
    bad = []
    for rel in _tracked():
        if rel == this_file or rel.endswith((".png", ".ico")):
            continue
        content = (PROJECT_ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        bad += [f"{rel}: {w}" for w in WEAK_VALUES if w in content]
    assert bad == []


def test_env_example_has_no_password():
    lines = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    assert "POSTGRES_PASSWORD=" in lines


def test_compose_has_no_literal_password_and_binds_to_loopback():
    text = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    for line in text.splitlines():
        if "POSTGRES_PASSWORD" in line:
            assert "${" in line, line
    services = yaml.safe_load(text)["services"]
    for name, spec in services.items():
        for port in spec.get("ports", []):
            assert str(port).startswith("127.0.0.1:"), f"{name}: {port}"
