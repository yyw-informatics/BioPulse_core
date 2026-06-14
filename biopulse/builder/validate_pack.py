from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def validate_pack(pack_dir: Path | str) -> dict:
    pack = Path(pack_dir)
    violations: list[str] = []
    warnings: list[str] = []

    if not pack.exists():
        return {"valid": False, "violations": [f"Pack does not exist: {pack}"], "warnings": []}

    task = _read_task_yaml(pack / "task.yaml", warnings, violations)
    task_type = task.get("task_type")

    required = [
        "task.yaml",
        "public/instruction.md",
        "public/output_schema.md",
        "hidden/ground_truth/solution.h5ad",
        "metrics/score.py",
        "metrics/metric_config.yaml",
        "evidence_spec.yaml",
        "source_manifest.json",
    ]
    if not task_type:
        violations.append("task.yaml is missing task_type")
    else:
        from biopulse.tasks.registry import get as get_task

        try:
            record = get_task(task_type)
        except ValueError:
            violations.append(f"Unsupported task_type in task.yaml: {task_type}")
        else:
            required.extend(f"public/input/{spec.filename}" for spec in record.inputs)

    for rel in required:
        if not (pack / rel).exists():
            violations.append(f"Missing required file: {rel}")

    public_dir = pack / "public"
    if public_dir.exists():
        for path in public_dir.rglob("*"):
            parts = {part.lower() for part in path.parts}
            if "hidden" in parts:
                violations.append(f"Forbidden hidden path under public/: {path.relative_to(pack)}")
            if path.name.lower() == "solution.h5ad":
                violations.append(f"Forbidden solution file under public/: {path.relative_to(pack)}")
    else:
        violations.append("Missing public/ directory")

    hidden_dir = pack / "hidden" / "ground_truth"
    if not hidden_dir.exists():
        violations.append("Missing hidden/ground_truth/ directory")

    manifest_path = pack / "source_manifest.json"
    if manifest_path.exists():
        try:
            json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            violations.append(f"Invalid source_manifest.json: {exc}")

    return {"valid": not violations, "violations": sorted(set(violations)), "warnings": warnings}


def _read_task_yaml(path: Path, warnings: list[str], violations: list[str]) -> Dict[str, Any]:
    if not path.exists():
        violations.append("Missing required file: task.yaml")
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        loaded = yaml.safe_load(text)
        return loaded if isinstance(loaded, dict) else {}
    except Exception as exc:
        warnings.append(f"Could not parse task.yaml with PyYAML, using simple parser: {exc}")
        parsed: Dict[str, str] = {}
        for line in text.splitlines():
            if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            parsed[key.strip()] = value.strip().strip('"')
        return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a BioPulse benchmark pack")
    parser.add_argument("pack_dir", type=Path)
    parser.add_argument("--json", action="store_true", help="Print machine-readable validation result")
    args = parser.parse_args(argv)

    result = validate_pack(args.pack_dir)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        status = "VALID" if result["valid"] else "INVALID"
        print(f"{status}: {args.pack_dir}")
        for violation in result["violations"]:
            print(f"VIOLATION: {violation}")
        for warning in result["warnings"]:
            print(f"WARNING: {warning}")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
