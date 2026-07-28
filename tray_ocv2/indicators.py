"""자기방전 3지표 삼각검증 — docs/01_analysis_plan.md C 그룹 (진단의 최우선 관문).

⚠️ docs/04_logic_audit.md 오류 1: S_A·S_B 는 충전 "직후" 값의 차이라 이완(relaxation)이
섞여 있을 수 있다. compute() 가 반환하는 rest_before_from 컬럼으로 반드시 함께 확인한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema
from .fields import pairwise_corr, percentile_rank, tray_delta
from .timing import indicator_hours, rate_mv_per_day, rest_before_ocv


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """S_A/S_B/S_C(mV) + 구간시간(h) + 율(mV/day) + 직전 휴지시간(h) 추가."""
    out = df.copy()
    for ind in schema.INDICATORS:
        vf, vt = f"v_{ind.from_stage}", f"v_{ind.to_stage}"
        if vf in out.columns and vt in out.columns:
            out[ind.key] = out[vf] - out[vt]
        hours = indicator_hours(out, ind.key)
        out[f"{ind.key}_hours"] = hours
        if ind.key in out.columns:
            out[f"{ind.key}_rate"] = rate_mv_per_day(out[ind.key], hours)
        out[f"{ind.key}_rest_before_h"] = rest_before_ocv(out, ind.from_stage)
    return out


def tray_relative(df: pd.DataFrame, keys: list[str] | None = None) -> pd.DataFrame:
    """[C1 입력] 트레이 내 중앙값 제거 — 위치 편차만 남긴다 (within-tray)."""
    keys = keys or [i.key for i in schema.INDICATORS]
    out = df.copy()
    for k in keys:
        if k in out.columns:
            out[f"{k}_dev"] = tray_delta(out, k, stat="median")
    return out


def rest_time_report(df: pd.DataFrame) -> pd.DataFrame:
    """[K10] ★ S_A/S_B 가 이완 지배인지 판별하는 전제 조건 확인.

    from_stage(OCV2/OCV4) 직전 휴지시간의 분포. 짧으면(예: 중앙값 < 수 시간)
    해당 지표는 자기방전이 아니라 이완을 재고 있을 가능성이 크다 (오류 1).
    """
    rows = []
    for ind in schema.INDICATORS:
        col = f"{ind.key}_rest_before_h"
        if col not in df.columns:
            continue
        h = pd.to_numeric(df[col], errors="coerce").dropna()
        if h.empty:
            rows.append({"지표": ind.key, "구간": f"{ind.from_stage}→{ind.to_stage}",
                        "상태": "휴지시간 컬럼 없음(매핑 미확인)"})
            continue
        rows.append({
            "지표": ind.key, "구간": f"{ind.from_stage}→{ind.to_stage}",
            "직전휴지_중앙값(h)": round(float(h.median()), 2),
            "직전휴지_최소(h)": round(float(h.min()), 2),
            "판정": ("이완 지배 의심 (짧음)" if h.median() < 3 else
                    "이완 영향 제한적 (충분히 김)"),
        })
    return pd.DataFrame(rows)


def triangulate(df: pd.DataFrame) -> pd.DataFrame:
    """[C1] ★★ 최우선 관문 — S_A/S_B/S_C 의 within-tray 상관행렬.

    있으면: 자기방전은 셀 고유 재현 특성 → 현 판정은 실체를 본다.
    없으면: docv7 은 구간 특이 잡음일 수 있다 — 단, rest_time_report 로 오류1 배제 후 판단.
    """
    keys = [f"{i.key}_dev" for i in schema.INDICATORS if f"{i.key}_dev" in df.columns]
    if len(keys) < 2:
        return pd.DataFrame()
    return pairwise_corr(df, keys)


def cross_check_grade(df: pd.DataFrame, grade_col: str = "grade") -> dict:
    """[C2] ★★ 현재 유일한 실측 기반 비순환 검증.

    S_A·S_B 는 라벨(등급) 생성에 쓰이지 않았다. 등급 E 셀이 S_A/S_B 에서도
    상위 꼬리에 있으면, 분해 없이 순환 라벨을 반쯤 깨는 증거가 된다.
    """
    if grade_col not in df.columns:
        return {"ok": False, "reason": "등급 컬럼 없음"}
    out = {}
    is_fail = df[grade_col].astype(str) == schema.GRADE_FAIL
    if is_fail.sum() == 0:
        return {"ok": False, "reason": "불량(E) 셀 없음"}

    for key in ("S_A_dev", "S_B_dev"):
        if key not in df.columns:
            continue
        pct = percentile_rank(df[key].abs())
        fail_pct = pct[is_fail].dropna()
        if fail_pct.empty:
            continue
        out[key] = {
            "n_fail": int(len(fail_pct)),
            "median_percentile": float(fail_pct.median()),
            "frac_above_p90": float((fail_pct > 0.90).mean()),
            "note": ("median_percentile 이 0.5보다 훨씬 크고 frac_above_p90 이 0.1보다 "
                    "크면(무작위 기대치 이상) 등급 E 가 S_A/S_B 에서도 튄다는 뜻"),
        }
    return {"ok": True, "results": out}


def grade_composition(df: pd.DataFrame) -> dict:
    """[A7] 래퍼 — qc.grade_consistency 를 그대로 노출 (C 그룹에서도 참조하도록)."""
    from .qc import grade_consistency
    return grade_consistency(df)
