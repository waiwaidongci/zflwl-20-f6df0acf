from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


ENTRY = os.environ.get("REFRACTION_SOLVER", "/app/refraction.py")

SAFE_ENV = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/tmp",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
}


def _run(payload: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-S", "-P", "-B", ENTRY, str(payload), str(output)],
        text=True,
        capture_output=True,
        env=SAFE_ENV,
        timeout=15,
        check=False,
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _travel(offset: float, v1: float, v2: float, thickness: float) -> float:
    direct = offset / v1
    head = 2.0 * thickness * math.sqrt(1.0 / v1**2 - 1.0 / v2**2) + offset / v2
    return min(direct, head)


def _profile(
    v1: float,
    v2: float,
    thickness: float,
    size: int,
    seed: int,
    noise: float = 0.0,
) -> dict[str, object]:
    shots: list[dict[str, object]] = []
    for index in range(size):
        offset = round(0.2 + index * 35.0 / (size - 1), 6)
        arrival = _travel(offset, v1, v2, thickness)
        arrival += noise * math.sin(index * 0.619 + seed)
        code = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        value = (seed * 997 + index * 7919) % (36**4)
        digits: list[str] = []
        current = value
        while current:
            current, remainder = divmod(current, 36)
            digits.append(code[remainder])
        identifier = "".join(reversed(digits or ["0"])).rjust(4, "0")
        shots.append(
            {
                "id": f"SG-{identifier}",
                "offset_km": offset,
                "travel_s": round(arrival, 8),
            }
        )
    return {"schema": "refraction-v1", "noise_sigma": noise, "shots": shots}


def test_gate_recovers_two_layer_solution(tmp_path: Path) -> None:
    """题面：
    「遇到结构或数值域错误时，处理同样停止：旧结果保留，退出码 2，标准错误中的拒绝行以 `REFUSED: ` 开头。」
    """

    # CASES: QUAKE-CMD-01 QUAKE-OUT-01 QUAKE-OUT-02
    payload = _profile(2.1, 4.6, 3.2, 20, 401)
    source = tmp_path / "profile.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["v1_km_s"] == pytest.approx(2.1, abs=0.15)
    assert result["v2_km_s"] == pytest.approx(4.6, abs=0.2)
    assert result["thickness_km"] == pytest.approx(3.2, abs=0.25)


def test_clean_profile_recovers_direct_wave_velocity(tmp_path: Path) -> None:
    """题面：
    「`shots` 的条数落在 10 至 48 之间；每条记录不接受 `id`、`offset_km`、`travel_s` 之外的键。每个炮点号由 `SG-` 和四位大写字母数字组成，同一测线内只能出现一次。`offset_km` 位于 0.1 至 40，并沿记录顺序单调增大。`travel_s` 位于 0 至 30。」
    """

    # CASES: QUAKE-SCENARIO-01 QUAKE-CORE-01
    payload = _profile(2.1, 4.6, 3.2, 22, 402)
    source = tmp_path / "profile.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["v1_km_s"] == pytest.approx(2.1, abs=0.12)
    assert result["rmse_s"] < 0.02


def test_near_contrast_reveals_refractor_velocity(tmp_path: Path) -> None:
    """题面：
    「结果文件恰好包含 status、v1_km_s、v2_km_s、thickness_km、rmse_s 五个键。`status` 为 `"ok"`，四个数值保留六位小数。`rmse_s` 用未取整参数计算所有炮点残差的均方根，保留六位小数。」
    """

    # CASES: QUAKE-CORE-02 QUAKE-BRANCH-01
    payload = _profile(2.4, 3.2, 2.2, 26, 403)
    source = tmp_path / "profile.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["v1_km_s"] == pytest.approx(2.4, abs=0.12)
    assert result["v2_km_s"] == pytest.approx(3.2, abs=0.15)
    assert result["thickness_km"] == pytest.approx(2.2, abs=0.3)


def test_noisy_long_line_recovers_global_layer_model(tmp_path: Path) -> None:
    """题面：
    「若两层速度关系无法由这批样本支撑，就不发布剖面。命令退出码为 2，标准错误含一行以 `REFUSED: ` 开头的拒绝原因，且输出文件不会出现。」
    """

    # CASES: QUAKE-CORE-03 QUAKE-SCENARIO-02
    payload = _profile(1.8, 5.2, 4.5, 48, 404, noise=0.004)
    source = tmp_path / "profile.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["v1_km_s"] == pytest.approx(1.8, abs=0.15)
    assert result["v2_km_s"] == pytest.approx(5.2, abs=0.25)
    assert result["thickness_km"] == pytest.approx(4.5, abs=0.4)


def test_public_upper_shot_count_finishes_within_deadline(tmp_path: Path) -> None:
    """题面：
    「公开上限的 48 个炮点必须在 8 秒内处理完成。反演只使用传入测线，禁止读取测试目录，更不许内置样例速度或厚度。」
    """

    # CASES: QUAKE-PERF-01 QUAKE-HINT-01
    payload = _profile(2.2, 5.8, 5.0, 48, 405, noise=0.003)
    source = tmp_path / "profile.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, "-S", "-P", "-B", ENTRY, str(source), str(output)],
        text=True,
        capture_output=True,
        env=SAFE_ENV,
        timeout=15,
        check=False,
    )
    elapsed = time.monotonic() - started
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["status"] == "ok"
    assert elapsed < 8


def test_minimum_shot_count_is_accepted(tmp_path: Path) -> None:
    """题面：
    「`shots` 的条数落在 10 至 48 之间」
    """

    # CASES: QUAKE-RANGE-01 QUAKE-SCHEMA-01
    payload = _profile(2.0, 5.0, 3.0, 10, 406)
    source = tmp_path / "profile.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert set(result) == {"status", "v1_km_s", "v2_km_s", "thickness_km", "rmse_s"}


def test_shots_require_exact_three_keys(tmp_path: Path) -> None:
    """题面：
    「每个炮点记录的是两者中较早到达的一支，再加上不超过公开噪声范围的测量残差。每个点究竟归直达支还是折射支没有单独给出，拟合时必须一起恢复。」
    """

    # CASES: QUAKE-SCHEMA-03
    payload = _profile(2.0, 5.0, 3.0, 18, 412)
    payload["shots"][0].pop("travel_s")
    source = tmp_path / "profile.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_too_few_shots_are_refused(tmp_path: Path) -> None:
    """题面：
    「河道勘察队重新处理一条旧测线时，只剩下炮点距和初至时间，炮班当时记下的分层速度与覆盖层厚度已经找不到。下一孔位要参考这条剖面，技术员必须先确认有没有稳定的折射界面，而不是把所有点拉成一根直线。」
    """

    # CASES: QUAKE-RANGE-02
    payload = _profile(2.0, 5.0, 3.0, 9, 407)
    source = tmp_path / "profile.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_duplicate_shot_id_is_refused(tmp_path: Path) -> None:
    """题面：
    「遇到结构或数值域错误时，处理同样停止：旧结果保留，退出码 2，标准错误中的拒绝行以 `REFUSED: ` 开头。」
    """

    # CASES: QUAKE-SCHEMA-06
    payload = _profile(2.0, 5.0, 3.0, 18, 408)
    payload["shots"][2]["id"] = payload["shots"][1]["id"]
    source = tmp_path / "profile.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_repeated_offset_is_refused(tmp_path: Path) -> None:
    """题面：
    「遇到结构或数值域错误时，处理同样停止：旧结果保留，退出码 2，标准错误中的拒绝行以 `REFUSED: ` 开头。」
    """

    # CASES: QUAKE-SCHEMA-04
    payload = _profile(2.0, 5.0, 3.0, 18, 413)
    payload["shots"][4]["offset_km"] = payload["shots"][3]["offset_km"]
    source = tmp_path / "profile.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_unknown_schema_is_refused(tmp_path: Path) -> None:
    """题面：
    「遇到结构或数值域错误时，处理同样停止：旧结果保留，退出码 2，标准错误中的拒绝行以 `REFUSED: ` 开头。」
    """

    # CASES: QUAKE-SCHEMA-02
    payload = _profile(2.0, 5.0, 3.0, 18, 411)
    payload["schema"] = "refraction-v2"
    source = tmp_path / "profile.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_extra_input_key_keeps_existing_result_bytes(tmp_path: Path) -> None:
    """题面：
    「不合规输入必须拒收，不得留下半成品。」
    """

    # CASES: QUAKE-SCHEMA-01 QUAKE-FAIL-01
    payload = _profile(2.0, 5.0, 3.0, 18, 409)
    payload["surveyor"] = "north-line"
    source = tmp_path / "profile.json"
    output = tmp_path / "result.json"
    old_payload = b'{"previous":true}\n'
    output.write_bytes(old_payload)
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert output.read_bytes() == old_payload


def test_success_writes_silent_canonical_json(tmp_path: Path) -> None:
    """题面：
    「同一输入重跑应得到相同字节。」
    """

    # CASES: QUAKE-SERIAL-01 QUAKE-SERIAL-02 QUAKE-OUTPUT-EMPTY-01
    payload = _profile(2.1, 4.6, 3.2, 20, 410)
    source = tmp_path / "profile.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    second_output = tmp_path / "result-copy.json"
    second_run = _run(source, second_output)
    raw = output.read_bytes()
    parsed = json.loads(raw)
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    assert completed.stdout == ""
    assert completed.stderr == ""
    assert second_run.returncode == 0
    assert second_output.read_bytes() == raw
    assert raw == canonical
