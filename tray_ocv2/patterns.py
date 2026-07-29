"""위치 패턴 유형·비율 — docs/01_analysis_plan.md L 그룹.

원인(팬 등)이 확정되지 않아도 성립하는 문제점 — "패턴이 한 종류가 아니고,
고정 보정으로는 일부만 잡힌다"는 적응형/ML 보정 필요성의 논리적 다리.
spatial.py 의 PCA(EOF) 점수로 유형을 분류한다 — 눈으로 8유형을 나누는 대신
연속 좌표(모드 점수)의 부호·크기로 분류해 자의성을 줄인다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema
from .fields import position_field, tray_delta
from .spatial import build_tray_matrix, pca_modes


def classify_patterns(df: pd.DataFrame, value_col: str, n_modes: int = 2,
                      weak_threshold_frac: float = 0.15) -> pd.DataFrame:
    """[L1] 트레이별 패턴 유형 분류 — PCA mode0(주로 링/역링) 부호 + 크기 기준.

    유형: 링(mode0 점수 > 임계) / 역링(< −임계) / 약함·무구조(|점수| < 임계).
    임계값은 |score| 분포의 weak_threshold_frac 분위수로 자동 결정한다
    (손으로 정한 상수 대신 — 자의성 축소).
    """
    tray_ids, mat = build_tray_matrix(df, value_col)
    pca = pca_modes(mat, n_modes=n_modes)
    if not pca.get("ok"):
        return pd.DataFrame()

    used_idx = pca["tray_index_valid"]
    used_trays = tray_ids[used_idx]
    scores = pca["scores"]
    mode0 = scores[:, 0]
    thresh = float(np.quantile(np.abs(mode0), weak_threshold_frac))

    labels = np.where(mode0 > thresh, "링(ring)",
             np.where(mode0 < -thresh, "역링(anti-ring)", "약함/무구조"))

    out = pd.DataFrame({"tray_id": used_trays, "mode0_score": mode0, "pattern": labels})
    if scores.shape[1] > 1:
        out["mode1_score"] = scores[:, 1]
    return out


def pattern_ratio(pattern_df: pd.DataFrame) -> dict:
    """[L2] 유형별 비율."""
    if pattern_df is None or pattern_df.empty:
        return {"ok": False, "reason": "분류 결과 없음"}
    counts = pattern_df["pattern"].value_counts(normalize=True).to_dict()
    return {"ok": True, "ratio": {k: round(v, 4) for k, v in counts.items()},
           "n_trays": int(len(pattern_df))}


def pattern_gradient_size(df: pd.DataFrame, value_col: str,
                          pattern_df: pd.DataFrame) -> dict:
    """[L3] 유형별 트레이 내 구배 크기(p2p, 판정기준 대비 %)."""
    if pattern_df is None or pattern_df.empty:
        return {"ok": False, "reason": "분류 결과 없음"}
    d = df.copy()
    d["_dev"] = tray_delta(d, value_col, stat="median")
    p2p = d.groupby("tray_id")["_dev"].apply(lambda s: float(s.max() - s.min()))
    merged = pattern_df.merge(p2p.rename("p2p"), left_on="tray_id",
                             right_index=True, how="left")
    out = {}
    for pat, g in merged.groupby("pattern"):
        out[pat] = {
            "median_p2p_mv": float(g["p2p"].median()),
            "median_p2p_pct_of_offset": float(g["p2p"].median()
                                              / schema.JUDGE_OFFSET_MV * 100),
            "n": int(len(g)),
        }
    return {"ok": True, "by_pattern": out}


def pattern_failure_rate(df: pd.DataFrame, pattern_df: pd.DataFrame,
                         grade_col: str = "grade") -> dict:
    """[L4] 유형별 트레이당 평균 불량(E) 판정률."""
    if pattern_df is None or pattern_df.empty or grade_col not in df.columns:
        return {"ok": False, "reason": "분류 결과 또는 등급 컬럼 없음"}
    fail_rate = df.groupby("tray_id")[grade_col].apply(
        lambda s: float((s.astype(str) == schema.GRADE_FAIL).mean()))
    merged = pattern_df.merge(fail_rate.rename("fail_rate"), left_on="tray_id",
                             right_index=True, how="left")
    grp = merged.groupby("pattern")["fail_rate"].agg(["mean", "count"])
    return {
        "ok": True,
        "by_pattern": {idx: {"mean_fail_rate": float(row["mean"]), "n": int(row["count"])}
                      for idx, row in grp.iterrows()},
        "note": "유형에 따라 평균 불량률이 크게 다르면 패턴이 판정에 실제 영향을 준다는 뜻",
    }


def fixed_correction_limit(variance_decomp: dict) -> dict:
    """[L5, 오류5 수정] 고정 보정의 한계 — spatial.variance_decomposition() 결과 재노출.

    구 계산("고정성분은 분산의 약 10%")은 p2p 비율 제곱 어림이라 부정확했다(오류5).
    정식 분산분해 결과(frac_position_fixed)를 그대로 인용한다.
    """
    if not variance_decomp.get("ok"):
        return {"ok": False, "reason": "분산분해 실패"}
    frac = variance_decomp["frac_position_fixed"]
    return {
        "ok": True, "frac_position_fixed": frac,
        "frac_position_fixed_pct": round(frac * 100, 1),
        "note": (f"고정 위치맵(모든 트레이 공통)으로 잡을 수 있는 분산은 전체의 약 "
                f"{frac * 100:.1f}%뿐. 나머지는 트레이마다 달라 고정 보정으론 못 잡는다 "
                "(정식 분산분해 기반 — 오류5의 p2p 비율 제곱 어림을 대체)"),
    }


def split_half_reproducibility(df: pd.DataFrame, value_col: str, seed: int = 0) -> dict:
    """[L6] 랏 1개일 때의 재현성 확인 — 트레이를 무작위 절반으로 나눠 공통 지문 비교.

    양쪽에서 같은 무늬가 재현되면 우연이 아니라는 근거(랏 간 비교의 대체).
    """
    d = df.copy()
    d["_dev"] = tray_delta(d, value_col, stat="median")
    tray_ids = d["tray_id"].dropna().unique()
    if len(tray_ids) < 20:
        return {"ok": False, "reason": "트레이 수 부족(<20)"}
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(tray_ids)
    half = len(shuffled) // 2
    g1, g2 = set(shuffled[:half]), set(shuffled[half:])

    f1 = position_field(d[d["tray_id"].isin(g1)], "_dev", agg="mean").to_numpy().ravel()
    f2 = position_field(d[d["tray_id"].isin(g2)], "_dev", agg="mean").to_numpy().ravel()
    ok = np.isfinite(f1) & np.isfinite(f2)
    if ok.sum() < 20:
        return {"ok": False, "reason": "유효 위치 부족"}
    r = float(np.corrcoef(f1[ok], f2[ok])[0, 1])
    return {
        "ok": True, "split_half_corr": r,
        "n_tray_group1": len(g1), "n_tray_group2": len(g2),
        "note": "무작위 절반끼리 공통 지문(평균 필드) 상관. 높으면(예: >0.5) 우연이 아님",
    }
