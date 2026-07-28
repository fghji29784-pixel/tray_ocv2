"""데이터 신뢰성 검증 — docs/01_analysis_plan.md A 그룹.

가장 먼저 실행한다. 여기서 걸리면 이후 분석이 전부 무의미해진다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema
from .fields import mode_estimate


def lsb_estimate(series) -> float:
    """[A1] 분해능(LSB) 추정. 정렬된 값의 최소 반복 간격을 찾는다 (휴리스틱)."""
    x = pd.to_numeric(pd.Series(series), errors="coerce").dropna().sort_values().unique()
    if x.size < 3:
        return np.nan
    diffs = np.diff(x)
    diffs = diffs[diffs > 1e-9]
    if diffs.size == 0:
        return 0.0
    d = np.round(diffs, 6)
    vals, counts = np.unique(d, return_counts=True)
    thresh = counts.max() * 0.05
    cand = vals[counts >= thresh]
    return float(cand.min()) if cand.size else float(vals.min())


def missing_rate(df: pd.DataFrame, value_col: str) -> float:
    """[A2] 전체 결측률."""
    if value_col not in df.columns:
        return np.nan
    return float(df[value_col].isna().mean())


def missing_by_position(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """[A2] 위치별 결측률 (12,12) — style.tray_heatmap 으로 바로 시각화 가능."""
    from .fields import position_field

    d = df.copy()
    d["_isna"] = d[value_col].isna().astype(float)
    return position_field(d, "_isna", agg="mean")


def duplicate_cells(df: pd.DataFrame) -> pd.DataFrame:
    """[A5] 같은 트레이·같은 위치에 행이 2개 이상 있는 경우."""
    key_cols = [c for c in ("tray_id", "row", "col") if c in df.columns]
    if len(key_cols) < 3:
        return pd.DataFrame()
    dup_mask = df.duplicated(subset=key_cols, keep=False)
    return df.loc[dup_mask, key_cols].sort_values(key_cols)


def stuck_temp_positions(df: pd.DataFrame, temp_col: str, max_unique: int = 3,
                         min_trays: int = 30) -> pd.DataFrame:
    """[A3] 위치별로 온도값의 고유값 개수가 비정상적으로 적은 곳 = 고장 의심 핀."""
    if temp_col not in df.columns:
        return pd.DataFrame()
    rows = []
    for (r, c), sub in df.groupby(["row", "col"]):
        v = pd.to_numeric(sub[temp_col], errors="coerce").dropna()
        if v.size < min_trays:
            continue
        n_unique = v.nunique()
        if n_unique <= max_unique:
            rows.append({"row": r, "col": c, "n": int(v.size),
                        "n_unique": int(n_unique), "value": v.iloc[0]})
    return pd.DataFrame(rows)


def grade_consistency(df: pd.DataFrame, docv7_col: str = "docv7_raw",
                      offset_mv: float = schema.JUDGE_OFFSET_MV,
                      bin_mv: float = schema.JUDGE_MODE_BIN_MV,
                      grade_col: str = "grade") -> dict:
    """[A7] ★ '등급'이 현행 규칙(mode+offset)의 산물인지 실측 검증.

    100% 일치 = 완전 순환. 불일치가 있으면 그 부분은 순환 밖 정보
    (docs/04_logic_audit.md A7). 반드시 C2 보다 먼저 실행한다.
    """
    if docv7_col not in df.columns or grade_col not in df.columns:
        return {"ok": False, "reason": "필요 컬럼 없음"}
    d = df[["tray_id", docv7_col, grade_col]].dropna(subset=["tray_id", docv7_col])
    tray_mode = d.groupby("tray_id")[docv7_col].transform(
        lambda s: mode_estimate(s, bin_mv))
    computed_fail = (d[docv7_col] - tray_mode) > offset_mv
    actual_fail = d[grade_col].astype(str) == schema.GRADE_FAIL

    both_fail = int((computed_fail & actual_fail).sum())
    only_actual = int((~computed_fail & actual_fail).sum())   # 규칙 밖 E (★핵심)
    only_computed = int((computed_fail & ~actual_fail).sum())  # 규칙은 걸렸는데 A로 남음
    n_actual_fail = int(actual_fail.sum())

    return {
        "ok": True,
        "n_cells": int(len(d)),
        "n_actual_fail": n_actual_fail,
        "n_computed_fail": int(computed_fail.sum()),
        "both_fail": both_fail,
        "only_in_grade": only_actual,
        "only_in_computed": only_computed,
        "pct_grade_explained_by_rule": (
            both_fail / n_actual_fail * 100 if n_actual_fail else np.nan),
        "note": ("100% 면 완전 순환. 100% 미만이면 'only_in_grade' 만큼은 "
                "규칙 밖 정보(다른 판정 기준 혼입) — 순환 아님"),
    }


def docv7_unit_check(df: pd.DataFrame, docv7_col: str = "docv7_raw",
                     from_col: str = "v_prvt1", to_col: str = "v_prvt3") -> dict:
    """[A9] `docv7_raw` 컬럼과 (PRVT1−PRVT3) 직접계산이 같은 정의·단위인지 확인.

    후보 스케일 {1, 1000, 0.001} 중 잔차가 가장 작은 것을 채택 후보로 제시한다.
    """
    if not all(c in df.columns for c in (docv7_col, from_col, to_col)):
        return {"ok": False, "reason": "필요 컬럼 없음"}
    computed_raw_unit = df[from_col] - df[to_col]   # v_prvt 원 단위 기준
    docv7 = df[docv7_col]
    ok = computed_raw_unit.notna() & docv7.notna()
    if ok.sum() < 10:
        return {"ok": False, "reason": "유효 표본 부족"}

    best = None
    for scale in (1.0, 1000.0, 0.001, -1.0, -1000.0):
        resid = docv7[ok] - computed_raw_unit[ok] * scale
        mae = float(np.mean(np.abs(resid)))
        corr = float(np.corrcoef(docv7[ok], computed_raw_unit[ok])[0, 1])
        cand = {"scale": scale, "mean_abs_error": mae, "corr": corr}
        if best is None or mae < best["mean_abs_error"]:
            best = cand
    return {"ok": True, "n": int(ok.sum()), "best_scale_candidate": best,
           "note": "scale 은 docv7_raw ≈ (v_prvt1-v_prvt3)*scale 를 만족하는 값"}


def score_card(df: pd.DataFrame) -> pd.DataFrame:
    """[A1~A5 요약] 주요 컬럼의 신뢰성 스코어카드 1장."""
    targets = {
        "docv7 (판정값)": "docv7_raw",
        "전용OCV1": "v_prvt1", "전용OCV3": "v_prvt3",
        "OCV1": "v_ocv1", "OCV2": "v_ocv2",
        "OCV1 온도": "t_ocv1", "OCV2 온도": "t_ocv2",
    }
    rows = []
    for label, col in targets.items():
        if col not in df.columns:
            rows.append({"항목": label, "상태": "컬럼 없음"})
            continue
        rows.append({
            "항목": label,
            "결측률(%)": round(missing_rate(df, col) * 100, 2),
            "분해능(LSB)": round(lsb_estimate(df[col]), 5),
            "표본수": int(df[col].notna().sum()),
        })
    return pd.DataFrame(rows)
