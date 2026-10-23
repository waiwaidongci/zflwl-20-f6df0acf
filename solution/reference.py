#!/usr/bin/env python3
"""Reference solver for two-layer seismic refraction inversion."""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any


INPUT_KEYS = {"schema", "noise_sigma", "shots"}
SHOT_KEYS = {"id", "offset_km", "travel_s"}
SHOT_ID = re.compile(r"SG-[A-Z0-9]{4}\Z")


class Refused(ValueError):
    pass


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Refused("duplicate JSON key")
        result[key] = value
    return result


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Refused("expected finite number")
    result = float(value)
    if not math.isfinite(result):
        raise Refused("expected finite number")
    return result


def parse_input(path: Path) -> tuple[float, list[tuple[float, float]]]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs,
        )
    except OSError as exc:
        raise Refused("input cannot be read") from exc
    except json.JSONDecodeError as exc:
        raise Refused("input is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != INPUT_KEYS:
        raise Refused("top-level key set is invalid")
    if payload.get("schema") != "refraction-v1":
        raise Refused("unsupported schema")
    noise = _number(payload.get("noise_sigma"))
    if not 0.0 <= noise <= 0.01:
        raise Refused("noise_sigma is outside the public interval")
    raw_shots = payload.get("shots")
    if not isinstance(raw_shots, list) or not 10 <= len(raw_shots) <= 48:
        raise Refused("shots must contain 10 to 48 records")
    identifiers: set[str] = set()
    shots: list[tuple[float, float]] = []
    previous = -math.inf
    for item in raw_shots:
        if not isinstance(item, dict) or set(item) != SHOT_KEYS:
            raise Refused("shot key set is invalid")
        identifier = item.get("id")
        if not isinstance(identifier, str) or SHOT_ID.fullmatch(identifier) is None:
            raise Refused("shot id is invalid")
        if identifier in identifiers:
            raise Refused("shot id is duplicated")
        identifiers.add(identifier)
        offset = _number(item.get("offset_km"))
        travel = _number(item.get("travel_s"))
        if not 0.1 <= offset <= 40.0 or offset <= previous:
            raise Refused("offsets must increase within the public interval")
        if not 0.0 <= travel <= 30.0:
            raise Refused("travel time is outside the public interval")
        shots.append((offset, travel))
        previous = offset
    return noise, shots


def _travel(offset: float, v1: float, v2: float, thickness: float) -> float:
    direct = offset / v1
    head = 2.0 * thickness * math.sqrt(max(1e-12, 1.0 / (v1 * v1) - 1.0 / (v2 * v2))) + offset / v2
    return min(direct, head)


def _cost(shots: list[tuple[float, float]], v1: float, v2: float, thickness: float) -> float:
    return math.fsum(
        (observed - _travel(offset, v1, v2, thickness)) ** 2
        for offset, observed in shots
    )


def _clamp(v1: float, v2: float, thickness: float) -> tuple[float, float, float]:
    bounded_v1 = min(5.0, max(0.8, v1))
    bounded_v2 = min(8.0, max(1.5, v2, bounded_v1 + 0.05))
    bounded_thickness = min(20.0, max(0.1, thickness))
    return bounded_v1, bounded_v2, bounded_thickness


def _refine(
    shots: list[tuple[float, float]],
    start: tuple[float, float, float],
) -> tuple[tuple[float, float, float], float]:
    v1, v2, thickness = _clamp(*start)
    best_cost = _cost(shots, v1, v2, thickness)
    step_v1 = 0.5
    step_v2 = 0.8
    step_h = 1.5
    for _ in range(80):
        improved = False
        for index in range(3):
            for direction in (-1.0, 1.0):
                trial = [v1, v2, thickness]
                trial[index] += direction * (step_v1, step_v2, step_h)[index]
                candidate = _clamp(*trial)
                candidate_cost = _cost(shots, *candidate)
                if candidate_cost < best_cost - 1e-15:
                    v1, v2, thickness = candidate
                    best_cost = candidate_cost
                    improved = True
        if not improved:
            step_v1 *= 0.5
            step_v2 *= 0.5
            step_h *= 0.5
            if max(step_v1, step_v2, step_h) < 1e-5:
                break
    return (v1, v2, thickness), best_cost


def _line_slope(points: list[tuple[float, float]], intercept: bool) -> tuple[float, float]:
    if not intercept:
        denominator = math.fsum(x * x for x, _ in points)
        return math.fsum(x * y for x, y in points) / denominator, 0.0
    size = len(points)
    mean_x = math.fsum(x for x, _ in points) / size
    mean_y = math.fsum(y for _, y in points) / size
    denominator = math.fsum((x - mean_x) ** 2 for x, _ in points)
    slope = math.fsum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
    return slope, mean_y - slope * mean_x


def recover(shots: list[tuple[float, float]]) -> dict[str, Any]:
    near_count = max(4, len(shots) // 3)
    far_count = max(5, len(shots) // 2)
    near = shots[:near_count]
    far = shots[-far_count:]
    near_slope, _ = _line_slope(near, False)
    far_slope, far_intercept = _line_slope(far, True)
    v1_guess = 1.0 / max(1e-6, near_slope)
    v2_guess = 1.0 / max(1e-6, far_slope)
    if v2_guess <= v1_guess + 0.05:
        v1_guess, v2_guess = 1.5, 4.0
    radicand = max(1e-9, 1.0 / (v1_guess * v1_guess) - 1.0 / (v2_guess * v2_guess))
    thickness_guess = far_intercept / (2.0 * math.sqrt(radicand))
    starts = [
        (v1_guess, v2_guess, thickness_guess),
        (v1_guess * 0.9, v2_guess * 1.08, max(0.2, thickness_guess * 0.8)),
        (v1_guess * 1.08, v2_guess * 0.92, thickness_guess * 1.2),
        (1.8, 4.5, 3.0),
        (2.5, 6.0, 6.0),
    ]
    best: tuple[tuple[float, float, float], float] | None = None
    for start in starts:
        candidate = _refine(shots, start)
        if best is None or candidate[1] < best[1]:
            best = candidate
    if best is None:
        raise Refused("inversion did not produce a candidate")
    (v1, v2, thickness), total = best
    rmse = math.sqrt(total / len(shots))
    if rmse > 0.2:
        raise Refused("sample does not support a two-layer refraction model")
    return {
        "rmse_s": round(rmse, 6),
        "status": "ok",
        "thickness_km": round(thickness, 6),
        "v1_km_s": round(v1, 6),
        "v2_km_s": round(v2, 6),
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("REFUSED: expected input and output paths", file=sys.stderr)
        return 2
    source = Path(sys.argv[1])
    target = Path(sys.argv[2])
    try:
        _, shots = parse_input(source)
        result = recover(shots)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    target.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
