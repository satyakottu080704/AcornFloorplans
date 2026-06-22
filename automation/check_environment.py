#!/usr/bin/env python3
"""Environment consistency checks for plan-generation deployment.

The script reports missing or risky configuration without printing secrets.
It is intended for local pre-deploy checks and server smoke checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


REQUIRED_SECRET_NAMES = (
    "SP_TENANT_ID",
    "SP_CLIENT_ID",
    "SP_CLIENT_SECRET",
    "SP_DRIVE_ID",
)

REQUIRED_RUNTIME_NAMES = (
    "PLAN_LAYOUT_PROVIDERS",
    "PLAN_PUBLISH_MODE",
)

SYNCED_FILE_PAIRS = (
    ("utils/layout_extractor.py", "automation/container/layout_extractor.py"),
    ("utils/vision_client.py", "automation/container/vision_client.py"),
)


@dataclass
class Check:
    name: str
    status: str
    detail: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _merged_env(env_file: Path | None) -> Dict[str, str]:
    merged = dict(os.environ)
    if env_file:
        merged.update(_load_env_file(env_file))
    return merged


def check_environment(repo_root: Path, env: Dict[str, str]) -> List[Check]:
    checks: List[Check] = []

    for name in REQUIRED_SECRET_NAMES:
        checks.append(
            Check(
                name=f"env:{name}",
                status="pass" if env.get(name) else "fail",
                detail="set" if env.get(name) else "missing",
            )
        )

    providers = env.get("PLAN_LAYOUT_PROVIDERS", "")
    provider_list = [p.strip().lower() for p in providers.split(",") if p.strip()]
    checks.append(
        Check(
            name="env:PLAN_LAYOUT_PROVIDERS",
            status="pass" if provider_list and provider_list[0] == "openai" else "warn",
            detail=providers or "missing; expected openai first",
        )
    )

    publish_mode = (env.get("PLAN_PUBLISH_MODE") or "review").strip().lower()
    checks.append(
        Check(
            name="env:PLAN_PUBLISH_MODE",
            status="pass" if publish_mode == "review" else "warn",
            detail=f"{publish_mode} (review is required until acceptance gates pass)",
        )
    )

    if publish_mode == "production":
        report = env.get("PLAN_ACCEPTANCE_REPORT", "")
        checks.append(
            Check(
                name="env:PLAN_ACCEPTANCE_REPORT",
                status="pass" if report else "fail",
                detail=report or "required for production mode",
            )
        )

    for left, right in SYNCED_FILE_PAIRS:
        left_path = repo_root / left
        right_path = repo_root / right
        if not left_path.exists() or not right_path.exists():
            checks.append(Check(f"sync:{left}<->{right}", "fail", "one or both files missing"))
            continue
        left_hash = _sha256(left_path)
        right_hash = _sha256(right_path)
        checks.append(
            Check(
                name=f"sync:{left}<->{right}",
                status="pass" if left_hash == right_hash else "warn",
                detail="in sync" if left_hash == right_hash else "files differ; keep provider copies aligned",
            )
        )

    return checks


def summarize(checks: Iterable[Check]) -> Dict[str, object]:
    rows = [check.__dict__ for check in checks]
    return {
        "ok": all(row["status"] != "fail" for row in rows),
        "failures": sum(1 for row in rows if row["status"] == "fail"),
        "warnings": sum(1 for row in rows if row["status"] == "warn"),
        "checks": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check plan-generation environment consistency.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--env-file", default="", help="Optional .env file to inspect without printing values")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    env = _merged_env(Path(args.env_file).resolve() if args.env_file else None)
    report = summarize(check_environment(repo_root, env))
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 2)


if __name__ == "__main__":
    main()
