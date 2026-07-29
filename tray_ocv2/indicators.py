"""자기방전 3지표 삼각검증 — docs/01_analysis_plan.md C 그룹 (진단의 최우선 관문).

⚠️ docs/04_logic_audit.md 오류 1: S_A·S_B 는 충전 "직후" 값의 차이라 이완(relaxation)이
섞여 있을 수 있다. compute() 가 반환하는 rest_before_from 컬럼으로 반드시 함께 확인한다.

⚠️ 오류8: C1(상관행렬)의 무상관은 "신호 없음"과 "신뢰도가 낮아 상관이 감쇠됨"을
구분하지 못한다. reliability_report()·disattenuated_matrix() 를 원 상관행렬과
반드시 병기한다.

⚠️ 오류9: S_A(HT1)·S_B(HT2)는 고온 노출시간이 3배 넘게 다를 수 있어(실측 Run #1:
6.0h vs 18.0h) 정규화 없이 나란히 비교하면 안 된다. compute() 가 아레니우스 등가율
(`*_rate_arrhenius`) 을 같이 계산한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema
from .fields import pairwise_corr, percentile_rank, tray_delta
from .timing import (arrhenius_normalize, indicator_high_temp_hours, indicator_hours,
                     rate_mv_per_day, rest_before_ocv)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """S_A/S_B/S_C(mV) + 구간시간(h) + 율(mV/day) + 아레니우스 등가율 + 직전 휴지시간(h)."""
    out = df.copy()
    for ind in schema.INDICATORS:
        vf, vt = f"v_{ind.from_stage}", f"v_{ind.to_stage}"
        if vf in out.columns and vt in out.columns:
            out[ind.key] = out[vf] - out[vt]
        hours = indicator_hours(out, ind.key)
        out[f"{ind.key}_hours"] = hours
        if ind.key in out.columns:
            out[f"{ind.key}_rate"] = rate_mv_per_day(out[ind.key], hours)

            hot_h = indicator_high_temp_hours(out, ind.key)
            room_h = (hours - hot_h).clip(lower=0)
            out[f"{ind.key}_hot_hours"] = hot_h
            out[f"{ind.key}_room_hours"] = room_h
            out[f"{ind.key}_rate_arrhenius"] = arrhenius_normalize(
                out[ind.key], hot_h, room_h)
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


# ---------------------------------------------------------------------------
# [오류8 수정] 신뢰도 · 감쇠보정 — C1 해석의 전제
# ---------------------------------------------------------------------------
# 상관계수는 측정 신뢰도가 낮으면 자동으로 0 쪽으로 끌려간다(감쇠, attenuation):
#   관측 상관 최댓값 ≈ sqrt(신뢰도_A × 신뢰도_B)
# 즉 S_A 신뢰도가 낮으면 진짜 성분이 완벽히 같아도 관측 상관은 0에 가깝게 나온다.
# C1 ≈ 0 을 "세 지표가 다른 걸 잰다"로 곧장 해석하면 안 되는 이유다.

def split_half_reliability_S_C(df: pd.DataFrame) -> dict:
    """[오류8] PRVT 3점으로 S_C 신뢰도 추정 (Spearman-Brown 반분신뢰도).

    (P1−P2)와 (P2−P3)를 반분(half-test)으로 보고, 두 반분 간 within-tray 상관에서
    전체(P1−P3=S_C)의 신뢰도를 역산한다. 이완 오염이 있으면 보수적(과소)으로 나온다.
    """
    if not all(c in df.columns for c in ("v_prvt1", "v_prvt2", "v_prvt3")):
        return {"ok": False, "reason": "PRVT2 컬럼 없음"}
    d = df.copy()
    d["_half1"] = d["v_prvt1"] - d["v_prvt2"]
    d["_half2"] = d["v_prvt2"] - d["v_prvt3"]
    h1 = tray_delta(d, "_half1", stat="median")
    h2 = tray_delta(d, "_half2", stat="median")
    ok = h1.notna() & h2.notna()
    if ok.sum() < 30:
        return {"ok": False, "reason": "표본 부족(<30)"}
    if h1[ok].std() < 1e-12 or h2[ok].std() < 1e-12:
        return {"ok": False, "reason": "분산 0"}
    r12 = float(np.corrcoef(h1[ok], h2[ok])[0, 1])
    sb = 2 * r12 / (1 + r12) if r12 > -1 else np.nan
    return {
        "ok": True, "half_corr": r12, "spearman_brown_reliability": float(sb),
        "n": int(ok.sum()),
        "note": ("⚠️ 오류11: 이 지표는 '모두가 특성을 갖고 측정오차가 섞인다'는 고전검사"
                "이론 가정이라, 불량률 0.065% 같은 희귀사건 데이터엔 부적합하다. 낮게 "
                "나와도 '꼬리(진짜 불량)가 불안정하다'는 뜻이 아니다 — "
                "tail_consistency_check() 를 함께 볼 것"),
    }


def tail_consistency_check(df: pd.DataFrame, top_frac: float = 0.01) -> dict:
    """[오류11 수정] ★희귀사건용 — '판정 대상 꼬리'에서만 두 반분이 일관되는지 본다.

    split_half_reliability_S_C() 는 전체 모집단 상관이라, 99.9%가 정상(=잴 특성 없음)인
    데이터에선 낮게 나오는 게 당연하다. 실제로 Run #2 에서 half_corr=0.00626 이 나왔는데,
    이는 96개 불량셀이 양쪽 반분에서 함께 튀는 것만으로도 (96/148858)×3² ≈ 0.006 으로
    거의 전부 설명된다 — '신호 없음'의 증거가 아니다.

    올바른 질문: **S_C 상위 꼬리로 뽑힌 셀들이 두 하위구간(P1→P2, P2→P3) 모두에서
    높은가?** 진짜 누설이면 두 구간 다 높아야 하고, 단발 잡음이면 한쪽만 높다.

    top_frac: 꼬리로 볼 상위 비율 (기본 1%).
    """
    if not all(c in df.columns for c in ("v_prvt1", "v_prvt2", "v_prvt3")):
        return {"ok": False, "reason": "PRVT2 컬럼 없음"}
    d = df.copy()
    d["_half1"] = d["v_prvt1"] - d["v_prvt2"]
    d["_half2"] = d["v_prvt2"] - d["v_prvt3"]
    d["_h1"] = tray_delta(d, "_half1", stat="median")
    d["_h2"] = tray_delta(d, "_half2", stat="median")
    d["_full"] = d["_h1"] + d["_h2"]           # = S_C 의 트레이내 편차

    ok = d["_h1"].notna() & d["_h2"].notna()
    sub = d[ok]
    if len(sub) < 200:
        return {"ok": False, "reason": "표본 부족"}

    n_tail = max(10, int(len(sub) * top_frac))
    tail = sub.nlargest(n_tail, "_full")
    rest = sub.drop(tail.index)

    # 꼬리 셀이 각 반분에서 전체 대비 어느 백분위에 있는가 (0.5=무작위)
    p1 = percentile_rank(sub["_h1"])
    p2 = percentile_rank(sub["_h2"])
    both_high = ((p1.loc[tail.index] > 0.9) & (p2.loc[tail.index] > 0.9)).mean()

    return {
        "ok": True, "n_tail": int(n_tail), "top_frac": top_frac,
        "tail_median_pct_half1": float(p1.loc[tail.index].median()),
        "tail_median_pct_half2": float(p2.loc[tail.index].median()),
        "frac_tail_high_in_both_halves": float(both_high),
        "expected_if_random": round(0.1 * 0.1, 4),
        "tail_mean_half1": float(tail["_h1"].mean()),
        "tail_mean_half2": float(tail["_h2"].mean()),
        "rest_mean_half1": float(rest["_h1"].mean()),
        "rest_mean_half2": float(rest["_h2"].mean()),
        "note": ("frac_tail_high_in_both_halves 가 무작위 기대치(0.01)보다 훨씬 크고 "
                "tail_median_pct 가 양쪽 반분 모두 0.5보다 크게 높으면 → 꼬리는 "
                "단발 잡음이 아니라 두 구간에 걸쳐 지속되는 실제 신호"),
    }


def quantization_reliability_bound(df: pd.DataFrame, indicator_key: str,
                                   lsb_mv: float) -> dict:
    """[오류8] LSB(분해능)만으로 추정하는 신뢰도 '상한'. 실제 신뢰도는 이보다 낮을 수 있다.

    차분(S_A=OCV2−OCV3 등)은 두 개의 독립 양자화 잡음을 물려받는다고 가정:
    noise_var = 2 × (LSB²/12) (균일분포 양자화 잡음 모델).
    """
    if indicator_key not in df.columns or not np.isfinite(lsb_mv):
        return {"ok": False, "reason": "지표 컬럼 또는 LSB 없음"}
    var_total = float(pd.to_numeric(df[indicator_key], errors="coerce").var())
    if not np.isfinite(var_total) or var_total <= 0:
        return {"ok": False, "reason": "분산 계산 불가"}
    noise_var = 2 * (lsb_mv ** 2) / 12.0
    reliability = max(0.0, 1 - noise_var / var_total)
    return {
        "ok": True, "lsb_mv": lsb_mv, "noise_var": noise_var,
        "signal_var_total": var_total, "reliability_upper_bound": reliability,
        "note": "양자화만 고려한 상한 — 실제 신뢰도는 이보다 낮을 수 있음(과장 금지)",
    }


def reliability_report(df: pd.DataFrame) -> dict:
    """[오류8] 세 지표의 신뢰도 추정 — C1 상관행렬과 반드시 병기한다."""
    from .qc import lsb_estimate

    out = {}
    for ind in schema.INDICATORS:
        if ind.key == "S_C":
            out[ind.key] = split_half_reliability_S_C(df)
            continue
        vf, vt = f"v_{ind.from_stage}", f"v_{ind.to_stage}"
        lsbs = [lsb_estimate(df[c]) for c in (vf, vt) if c in df.columns]
        lsbs = [x for x in lsbs if np.isfinite(x)]
        if not lsbs:
            out[ind.key] = {"ok": False, "reason": "LSB 추정 불가"}
            continue
        out[ind.key] = quantization_reliability_bound(df, ind.key, max(lsbs))
    return out


def _reliability_value(rel: dict) -> float:
    if not rel.get("ok"):
        return np.nan
    return rel.get("reliability_upper_bound", rel.get("spearman_brown_reliability", np.nan))


def disattenuated_corr(r_observed: float, reliability_a: float, reliability_b: float) -> float:
    """[오류8] 신뢰도로 보정한 상관 — 진짜 성분끼리의 상관 추정치(과대추정 가능, 참고용)."""
    if not (np.isfinite(reliability_a) and np.isfinite(reliability_b)):
        return np.nan
    denom = np.sqrt(reliability_a * reliability_b)
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    return float(np.clip(r_observed / denom, -1, 1))


def disattenuated_matrix(corr_df: pd.DataFrame, reliability: dict) -> pd.DataFrame:
    """[오류8] C1 원 상관행렬을 신뢰도로 감쇠보정한 버전. 원본과 나란히 제시할 것."""
    keys = list(corr_df.index)
    mat = corr_df.copy().astype(float)
    rel_vals = {k.replace("_dev", ""): _reliability_value(reliability.get(k.replace("_dev", ""), {}))
               for k in keys}
    for i in keys:
        ri = rel_vals.get(i.replace("_dev", ""), np.nan)
        for j in keys:
            if i == j:
                continue
            rj = rel_vals.get(j.replace("_dev", ""), np.nan)
            mat.loc[i, j] = disattenuated_corr(corr_df.loc[i, j], ri, rj)
    return mat
