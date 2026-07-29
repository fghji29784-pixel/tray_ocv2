"""판정 영향 정량화 — docs/01_analysis_plan.md I 그룹 (발표 클라이맥스).

I2: 위치별 불량 판정률 지도. I3: spike-in 시뮬레이션 — 유일한 비순환 성능지표
(현행 라벨은 현행 로직 산물이라 순환이지만, 인공 결함 검출률은 그렇지 않다).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema
from .fields import mode_estimate, position_field


def position_failure_rate(df: pd.DataFrame, grade_col: str = "grade") -> dict:
    """[I2] ★위치별 불량(E) 판정률 + 기대 대비 배수 지도."""
    if grade_col not in df.columns:
        return {"ok": False, "reason": "등급 컬럼 없음"}
    d = df.copy()
    d["_is_fail"] = (d[grade_col].astype(str) == schema.GRADE_FAIL).astype(float)
    field = position_field(d, "_is_fail", agg="mean")
    overall_rate = float(d["_is_fail"].mean())
    if overall_rate <= 0:
        return {"ok": False, "reason": "불량 0건 — 배수 계산 불가"}
    ratio_field = field / overall_rate
    arr = ratio_field.to_numpy()
    if not np.isfinite(arr).any():
        return {"ok": False, "reason": "유효 위치 없음"}
    max_idx = np.unravel_index(np.nanargmax(arr), arr.shape)
    return {
        "ok": True, "overall_fail_rate": overall_rate,
        "field": field, "ratio_field": ratio_field,
        "max_ratio": float(np.nanmax(arr)),
        "max_ratio_position": f"{chr(ord('A') + max_idx[1])}{max_idx[0] + 1}",
        "note": "ratio_field 값이 N이면 그 자리는 평균보다 N배 더 자주 불량 판정됨",
    }


def spike_in_test(df: pd.DataFrame, docv7_col: str = "docv7_raw",
                  magnitudes_mv=(0.5, 0.8, 1.5),
                  offset_mv: float = schema.JUDGE_OFFSET_MV,
                  bin_mv: float = schema.JUDGE_MODE_BIN_MV,
                  seed: int = 0) -> dict:
    """[I3] ★★ 유일한 비순환 성능지표 — 인공 결함 주입 후 위치별 검출률.

    트레이당 정상 셀 1개(★여러 개 주입 금지 — mode 자체가 이동해 판정기준이
    흔들린다, docs/04_logic_audit.md I3 주의)에 magnitude_mv 를 더해 '똑같은 크기의
    불량'을 만들고, 현 판정 로직(트레이 mode+offset)으로 검출되는지 테두리거리별로
    집계한다. 검출력이 위치에 무관하게 평평해야 정상 — 특정 위치에서만 낮으면
    그 자리 진짜 불량을 놓치고 있다는 뜻(발표의 결정적 증거).
    """
    if docv7_col not in df.columns or "tray_id" not in df.columns:
        return {"ok": False, "reason": "필요 컬럼 없음"}
    rng = np.random.default_rng(seed)
    d = df[["tray_id", "row", "col", docv7_col]].dropna().copy()
    if len(d) < 200:
        return {"ok": False, "reason": "표본 부족"}

    results = []
    for mag in magnitudes_mv:
        spiked = d.copy()
        chosen = spiked.groupby("tray_id", group_keys=False).apply(
            lambda g: g.sample(n=1, random_state=rng))
        chosen_idx = chosen.index
        spiked.loc[chosen_idx, docv7_col] = spiked.loc[chosen_idx, docv7_col] + mag

        tray_mode = spiked.groupby("tray_id")[docv7_col].transform(
            lambda s: mode_estimate(s, bin_mv))
        judged_fail = (spiked[docv7_col] - tray_mode) > offset_mv
        detected = judged_fail.loc[chosen_idx]

        r = spiked.loc[chosen_idx, "row"].to_numpy()
        c = spiked.loc[chosen_idx, "col"].to_numpy()
        edge_d = np.minimum.reduce([r - 1, 12 - r, c - 1, 12 - c])
        tmp = pd.DataFrame({"edge_d": edge_d, "detected": detected.to_numpy()})
        by_edge = tmp.groupby("edge_d")["detected"].agg(["mean", "count"])

        results.append({
            "magnitude_mv": mag, "overall_detection_rate": float(detected.mean()),
            "n_injected": int(len(chosen_idx)),
            "detection_by_edge_distance": {
                int(k): {"rate": float(row["mean"]), "n": int(row["count"])}
                for k, row in by_edge.iterrows()},
        })

    return {
        "ok": True, "results": results,
        "note": ("magnitude 별 overall_detection_rate 와 edge_distance(0=테두리)별 "
                "검출률을 본다. edge_d=0 의 검출률이 안쪽보다 뚜렷이 낮으면 "
                "'가장자리 불량을 놓친다'는 것이 실측으로 확인된 것"),
    }
