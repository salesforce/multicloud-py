"""Exercise the merge-gate shell and validate trusted workflow boundaries."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def workflow(name: str) -> dict:
    return yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text())


@pytest.mark.parametrize(
    "failed_job",
    ["QUALITY_RESULT", "TEST_RESULT", "BENCHMARK_RESULT", "PACKAGE_RESULT"],
)
@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped", ""])
def test_merge_gate_rejects_each_unsuccessful_dependency(
    failed_job: str, result: str
) -> None:
    gate = workflow("ci.yml")["jobs"]["test"]
    env = {
        **os.environ,
        "QUALITY_RESULT": "success",
        "TEST_RESULT": "success",
        "BENCHMARK_RESULT": "success",
        "PACKAGE_RESULT": "success",
        failed_job: result,
    }
    completed = subprocess.run(["bash", "-e", "-c", gate["steps"][0]["run"]], env=env)
    assert completed.returncode != 0


def test_merge_gate_accepts_all_successful_dependencies() -> None:
    gate = workflow("ci.yml")["jobs"]["test"]
    env = {
        **os.environ,
        "QUALITY_RESULT": "success",
        "TEST_RESULT": "success",
        "BENCHMARK_RESULT": "success",
        "PACKAGE_RESULT": "success",
    }
    completed = subprocess.run(["bash", "-e", "-c", gate["steps"][0]["run"]], env=env)
    assert completed.returncode == 0
    assert set(gate["needs"]) == {
        "quality",
        "python-tests",
        "benchmark-smoke",
        "package-validation",
    }
    assert gate["if"] == "always()"


def test_privileged_title_trigger_executes_only_trusted_base_script() -> None:
    config = workflow("pr-title-trusted.yml")
    events = config.get("on", config.get(True))
    assert set(events) == {"pull_request_target"}
    assert events["pull_request_target"]["branches"] == ["main"]
    assert config["permissions"] == {"contents": "read"}
    job = config["jobs"]["trusted-conventional-title"]
    checkout, validate = job["steps"]
    assert checkout["with"]["ref"] == "${{ github.event.pull_request.base.sha }}"
    assert checkout["with"]["persist-credentials"] is False
    assert checkout["with"]["sparse-checkout"] == "scripts/check_pr_title.py"
    assert validate["env"]["TITLE"] == "${{ github.event.pull_request.title }}"
    assert "${{" not in validate["run"]
    assert validate["run"] == 'python3 scripts/check_pr_title.py "$TITLE"'


def test_bootstrap_title_check_remains_available_on_pull_requests() -> None:
    config = workflow("pr-title.yml")
    events = config.get("on", config.get(True))
    assert "pull_request" in events
    assert "conventional-title" in config["jobs"]


def test_workflow_security_scan_is_fork_compatible_and_has_no_write_token() -> None:
    config = workflow("zizmor.yml")
    events = config.get("on", config.get(True))
    assert events["pull_request"]["branches"] == ["main"]
    assert config["permissions"] == {}
    job = config["jobs"]["zizmor"]
    assert job["permissions"] == {"contents": "read"}
    scan = job["steps"][-1]
    assert scan["with"]["advanced-security"] is False
    assert scan["with"]["annotations"] is True
    assert len(scan["uses"].split("@")[1]) == 40


def test_docs_and_rehearsal_checkouts_do_not_persist_tokens() -> None:
    for name, job_id in [
        ("docs.yml", "publish"),
        ("test-pypi.yml", "build-and-validate"),
    ]:
        for step in workflow(name)["jobs"][job_id]["steps"]:
            if step.get("uses", "").startswith("actions/checkout@"):
                assert step["with"]["persist-credentials"] is False


def test_generated_release_refs_are_environment_data_not_shell_source() -> None:
    jobs = workflow("release.yml")["jobs"]
    for job_id in ["sync-release-pr", "validate-release"]:
        for step in jobs[job_id]["steps"]:
            assert "${{" not in step.get("run", "")
