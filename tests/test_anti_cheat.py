from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


ENTRY = os.environ.get("REFRACTION_SOLVER", "/app/refraction.py")
VARIANTS = (
    (1.7, 4.2, 2.6, 24, 5171, 0.001),
    (2.5, 3.4, 1.8, 30, 5172, 0.001),
    (3.1, 6.7, 4.1, 36, 5173, 0.002),
)


def _load_output_helpers():
    path = Path(__file__).with_name("test_outputs.py")
    spec = importlib.util.spec_from_file_location("yel020_output_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPERS = _load_output_helpers()


def _signature(result: dict[str, object]) -> str:
    raw = json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _run_variants(tmp_path: Path) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for index, (v1, v2, thickness, size, seed, noise) in enumerate(VARIANTS):
        payload = HELPERS._profile(v1, v2, thickness, size, seed, noise)
        source = tmp_path / f"anti-input-{index}.json"
        output = tmp_path / f"anti-output-{index}.json"
        source.write_text(json.dumps(payload), encoding="utf-8")
        completed = HELPERS._run(source, output)
        actual = json.loads(output.read_text(encoding="utf-8")) if output.exists() else None
        results.append(
            {
                "returncode": completed.returncode,
                "v1": actual["v1_km_s"] if actual else None,
                "v2": actual["v2_km_s"] if actual else None,
                "thickness": actual["thickness_km"] if actual else None,
                "rmse": actual["rmse_s"] if actual else None,
                "signature": _signature(actual) if actual else None,
            }
        )
    return results


def test_anti_cheat_varied_profiles_cannot_reuse_one_velocity(tmp_path: Path) -> None:
    """题面：「若样本无法支持一个满足该速度关系的两层模型，就不发布剖面。」

    三组未公开测线分别具有不同速度、层厚、炮点数和噪声。只回放某一组固定剖面
    会在其余两组的速度或厚度上失真，无法同时解释全部直达与折射支路。
    """

    variants = _run_variants(tmp_path)
    assert [item["returncode"] for item in variants] == [0, 0, 0]
    assert [item["v1"] for item in variants] == pytest.approx([1.7, 2.5, 3.1], abs=0.18)
    assert [item["v2"] for item in variants] == pytest.approx([4.2, 3.4, 6.7], abs=0.25)
    assert [item["thickness"] for item in variants] == pytest.approx([2.6, 1.8, 4.1], abs=0.45)
    assert [item["rmse"] for item in variants] == pytest.approx([0.000638, 0.000660, 0.001413], abs=0.001)
    assert len({item["signature"] for item in variants}) == len(VARIANTS)
