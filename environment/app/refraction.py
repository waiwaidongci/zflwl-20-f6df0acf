#!/usr/bin/env python3
"""Starter seismic refraction inversion entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("REFUSED: expected input and output paths", file=sys.stderr)
        return 2
    source = Path(sys.argv[1])
    target = Path(sys.argv[2])
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print("REFUSED: unreadable input", file=sys.stderr)
        return 2

    shots = payload.get("shots", []) if isinstance(payload, dict) else []
    far = shots[-1] if shots else {"offset_km": 1.0, "travel_s": 1.0}
    near = shots[0] if shots else {"offset_km": 1.0, "travel_s": 0.5}
    far_slope = float(far.get("travel_s", 1.0)) / max(float(far.get("offset_km", 1.0)), 0.1)
    near_slope = float(near.get("travel_s", 0.5)) / max(float(near.get("offset_km", 1.0)), 0.1)
    v1 = max(near_slope, 0.8)
    result = {
        "rmse_s": 0.0,
        "status": "ok",
        "thickness_km": 1.0,
        "v1_km_s": v1,
        "v2_km_s": max(far_slope, v1 + 0.05),
    }
    target.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
