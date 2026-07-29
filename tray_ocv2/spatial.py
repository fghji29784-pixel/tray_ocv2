"""공간 지문 · 분산 분해 — docs/01_analysis_plan.md E 그룹 (발표 파트 I 핵심).

scipy/sklearn 없이 numpy SVD 로 PCA(EOF)를 직접 구현한다(docs/00_facts.md §6 환경 제약).
E1(평균+RMS 필드)은 fields.position_field 를 그대로 쓰면 되므로 여기 없음.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .fields import N_COLS, N_ROWS, robust_mad, to_grid, tray_delta


def quantile_field(df: pd.DataFrame, value_col: str, q: float,
                   row_col: str = "row", col_col: str = "col") -> pd.DataFrame:
    """[E2] 위치별 분위수(q) 필드 (트레이 내 편차 기준)."""
    d = df.copy()
    d["_dev"] = tray_delta(d, value_col, stat="median")
    d[row_col] = pd.to_numeric(d[row_col], errors="coerce")
    d[col_col] = pd.to_numeric(d[col_col], errors="coerce")
    d = d.dropna(subset=[row_col, col_col, "_dev"])
    out = d.groupby([row_col, col_col])["_dev"].quantile(q)
    grid = np.full((N_ROWS, N_COLS), np.nan)
    for (ri, ci), vi in out.items():
        ri, ci = int(ri), int(ci)
        if 1 <= ri <= N_ROWS and 1 <= ci <= N_COLS:
            grid[ri - 1, ci - 1] = vi
    return pd.DataFrame(grid, index=range(1, N_ROWS + 1), columns=list("ABCDEFGHIJKL"))


def mad_field(df: pd.DataFrame, value_col: str,
             row_col: str = "row", col_col: str = "col") -> pd.DataFrame:
    """[E2] 위치별 MAD(로버스트 산포) 필드."""
    d = df.copy()
    d["_dev"] = tray_delta(d, value_col, stat="median")
    d[row_col] = pd.to_numeric(d[row_col], errors="coerce")
    d[col_col] = pd.to_numeric(d[col_col], errors="coerce")
    d = d.dropna(subset=[row_col, col_col, "_dev"])
    out = d.groupby([row_col, col_col])["_dev"].apply(robust_mad)
    grid = np.full((N_ROWS, N_COLS), np.nan)
    for (ri, ci), vi in out.items():
        ri, ci = int(ri), int(ci)
        if 1 <= ri <= N_ROWS and 1 <= ci <= N_COLS:
            grid[ri - 1, ci - 1] = vi
    return pd.DataFrame(grid, index=range(1, N_ROWS + 1), columns=list("ABCDEFGHIJKL"))


def quantile_spread_report(df: pd.DataFrame, value_col: str) -> dict:
    """[E2 판정] 위치별 Q10/Q50/Q90 이 평행이동인지 산포까지 달라지는지 판별.

    docs/04_logic_audit.md 오류3 수정판: 평행이동/꼬리는 '아티팩트냐 실체냐'가 아니라
    '전체 영향이냐 산포까지 위치마다 다르냐'를 가른다. 여기선 정량 지표만 내고,
    측정계 판단(어디서 왔는지)은 별도로 한다.
    """
    q10 = quantile_field(df, value_col, 0.10).to_numpy()
    q50 = quantile_field(df, value_col, 0.50).to_numpy()
    q90 = quantile_field(df, value_col, 0.90).to_numpy()
    spread = q90 - q10
    ok = np.isfinite(q50) & np.isfinite(spread)
    if ok.sum() < 5:
        return {"ok": False, "reason": "유효 위치 부족"}
    return {
        "ok": True,
        "q50_range_mv": float(np.nanmax(q50) - np.nanmin(q50)),
        "iqr80_range_mv": float(np.nanmax(spread) - np.nanmin(spread)),
        "corr_q50_vs_spread": float(np.corrcoef(q50[ok], spread[ok])[0, 1]),
        "note": ("corr_q50_vs_spread 가 낮으면 위치 차이는 주로 '중심값 평행이동' "
                "— 모든 셀에 고르게 영향. 높으면 산포 자체도 위치마다 달라 "
                "'특정 위치에 불량이 집중'되어 있을 수 있음 — 위치 보정 시 "
                "그 집단을 통째로 놓칠 위험(오류3)"),
    }


def build_tray_matrix(df: pd.DataFrame, value_col: str, tray_col: str = "tray_id",
                      row_col: str = "row", col_col: str = "col",
                      already_relative: bool = False):
    """[E7/E8 공용] 트레이×144위치 행렬 구성. 결측은 NaN 유지(호출부에서 처리).

    already_relative=True 면 value_col 이 이미 트레이 내 편차라 tray_delta 를 다시
    걸지 않는다(E8 에서 위치고정 성분 제거 후 잔차를 넘길 때 사용).
    """
    d = df.copy()
    if already_relative:
        dev_col = value_col
    else:
        d["_dev"] = tray_delta(d, value_col, stat="median")
        dev_col = "_dev"
    tray_ids = d[tray_col].dropna().unique()
    mat = np.full((len(tray_ids), N_ROWS * N_COLS), np.nan)
    for i, tid in enumerate(tray_ids):
        sub = d[d[tray_col] == tid]
        grid = to_grid(sub, dev_col, row_col=row_col, col_col=col_col)
        mat[i] = grid.ravel()
    return tray_ids, mat


def pca_modes(matrix: np.ndarray, n_modes: int = 3) -> dict:
    """[E7] numpy SVD 기반 PCA(EOF). scipy/sklearn 없이 구현.

    matrix: (n_tray, 144). 결측(NaN)은 0으로 채운 뒤 진행(표준 EOF 관행) — 결측이
    많은 위치는 그만큼 신호가 죽는다는 한계가 있다.
    """
    valid_trays = ~np.all(np.isnan(matrix), axis=1)
    m = matrix[valid_trays]
    m = np.where(np.isfinite(m), m, 0.0)
    if m.shape[0] < max(5, n_modes + 2):
        return {"ok": False, "reason": "유효 트레이 수 부족"}

    col_mean = m.mean(axis=0)
    m_centered = m - col_mean

    U, S, Vt = np.linalg.svd(m_centered, full_matrices=False)
    k = min(n_modes, len(S))
    total_var = float(np.sum(S ** 2))
    if total_var < 1e-12:
        return {"ok": False, "reason": "분산 0"}

    loadings = Vt[:k].reshape(k, N_ROWS, N_COLS)
    scores = U[:, :k] * S[:k]
    explained = (S[:k] ** 2) / total_var

    return {
        "ok": True, "loadings": loadings, "scores": scores,
        "explained_variance_ratio": explained.tolist(),
        "tray_index_valid": np.where(valid_trays)[0],
        "n_tray_used": int(m.shape[0]),
        "note": ("loadings[k]=k번째 공간모드(12x12). scores[:,k]=트레이별 부호있는 점수 "
                "— 링/역링은 mode0 점수의 부호 차이일 뿐(패턴유형 분류에 그대로 사용)"),
    }


def variance_decomposition(df: pd.DataFrame, value_col: str, n_modes: int = 3) -> dict:
    """[E8, 오류5 수정] ★분산을 위치고정 / 트레이×위치(EOF) / 셀잔차로 정식 분해.

    ⚠️ 오류5: 기존 "고정성분은 분산의 약 10%"는 p2p 비율의 제곱으로 어림한 것이라
    부정확했다(분자=1035개 평균이라 잡음 거의 없음, 분모=개별 트레이라 측정잡음 포함
    → 비교 자체가 안 맞음). 이 함수가 SS(제곱합) 기반 정식 분해로 그 수치를 대체한다.

    분해:
      SS_total          = Σ(dev²)                (트레이내 편차 기준, 그랜드평균=0)
      SS_position       = Σ(pos_mean²)            위치별 전 트레이 평균(공통지문)이 설명
      SS_tray_position  = 잔차의 top-n_modes PCA 가 설명 (트레이별 저차 기저)
      SS_cell_residual  = 나머지 (셀 고유 신호 — 보정하면 안 되는 부분)
    """
    d = df.copy()
    d["_dev"] = tray_delta(d, value_col, stat="median")
    valid = d["_dev"].notna() & d["row"].notna() & d["col"].notna()
    sub = d[valid].copy()
    if len(sub) < 200:
        return {"ok": False, "reason": "표본 부족"}

    ss_total = float((sub["_dev"] ** 2).sum())
    if ss_total < 1e-12:
        return {"ok": False, "reason": "분산 0"}

    pos_mean = sub.groupby(["row", "col"])["_dev"].transform("mean")
    ss_position = float((pos_mean ** 2).sum())
    sub["_resid_after_pos"] = sub["_dev"] - pos_mean

    _, mat_resid = build_tray_matrix(sub, value_col="_resid_after_pos",
                                     already_relative=True)
    pca = pca_modes(mat_resid, n_modes=n_modes)

    if pca.get("ok"):
        m = mat_resid[~np.all(np.isnan(mat_resid), axis=1)]
        m = np.where(np.isfinite(m), m, 0.0)
        ss_resid_total = float(np.sum((m - m.mean(axis=0)) ** 2))
        ss_tray_position = ss_resid_total * float(np.sum(pca["explained_variance_ratio"]))
    else:
        ss_tray_position = 0.0

    ss_cell_residual = max(0.0, ss_total - ss_position - ss_tray_position)

    return {
        "ok": True, "ss_total": ss_total,
        "frac_position_fixed": ss_position / ss_total,
        "frac_tray_position_eof": ss_tray_position / ss_total,
        "frac_cell_residual": ss_cell_residual / ss_total,
        "n_modes_used": n_modes,
        "pca_explained_within_residual": pca.get("explained_variance_ratio"),
        "note": ("frac_position_fixed = 고정 위치맵으로 잡을 수 있는 한계(오류5가 지적한 "
                "'10%'의 정식 대체값). frac_tray_position_eof = 트레이별 저차 기저(PCA "
                "top-n_modes)로 추가로 잡을 수 있는 몫. frac_cell_residual = 셀 고유 "
                "신호(보정하면 안 됨). 셋을 합치면 1.0"),
    }
