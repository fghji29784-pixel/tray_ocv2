"""격자 변환 · 로버스트 통계 · 트레이 내 편차 — 여러 분석 그룹이 공유하는 기반.

scipy 없이 numpy 만으로 구현한다 (docs/00_facts.md §6 환경 제약).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_ROWS, N_COLS = 12, 12


def robust_mad(x) -> float:
    """중앙값 절대편차 (정규분포 스케일 보정, ×1.4826)."""
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy()
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return float(np.median(np.abs(x - med)) * 1.4826)


def mode_estimate(x, bin_mv: float) -> float:
    """히스토그램 기반 최빈값. 판정 로직(mode+offset) 재현용."""
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy()
    if x.size == 0 or bin_mv <= 0:
        return np.nan
    binned = np.round(x / bin_mv) * bin_mv
    vals, counts = np.unique(binned, return_counts=True)
    return float(vals[np.argmax(counts)])


def freedman_diaconis_bin(x) -> float:
    """Freedman-Diaconis 규칙으로 히스토그램 bin 폭을 자동 추정한다.

    mode 계산은 bin 폭에 민감하다(양자화) — 변수마다 손으로 bin을 정하지 않고
    분포(IQR·표본수)에서 자동으로 정하기 위함. IQR=0 이거나 표본이 너무 적으면 NaN.
    """
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy()
    if x.size < 2:
        return np.nan
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr <= 0:
        return np.nan
    return float(2.0 * iqr / np.cbrt(x.size))


def tray_delta(df: pd.DataFrame, value_col: str, by: str = "tray_id",
               stat: str = "median", mode_bin_mv: float | None = None) -> pd.Series:
    """셀 값 − 같은 트레이 내 중심통계. 위치 편차(구배) 추출의 기본 연산.

    stat="mode" 일 때 bin 폭은 기본적으로 `freedman_diaconis_bin` 으로 전체 컬럼에서
    한 번 추정해 모든 트레이에 동일하게 적용한다(트레이마다 다른 bin 폭을 쓰면 트레이간
    비교가 깨진다). 자동추정이 안 되면(IQR=0 등) 0.01 로 폴백한다.
    docv7처럼 실제 판정에 쓰이는 값은 `mode_bin_mv` 로 판정 로직의 고정 bin(예:
    `schema.JUDGE_MODE_BIN_MV`)을 직접 넘겨 자동추정을 우회한다.
    """
    g = df.groupby(by)[value_col]
    if stat != "mode":
        return df[value_col] - g.transform(stat)

    bin_mv = mode_bin_mv
    if bin_mv is None:
        bin_mv = freedman_diaconis_bin(df[value_col])
        if not np.isfinite(bin_mv) or bin_mv <= 0:
            bin_mv = 0.01
    center = g.transform(lambda s: mode_estimate(s, bin_mv=bin_mv))
    return df[value_col] - center


def sensitivity_median_vs_mode(df: pd.DataFrame, value_col: str, by: str = "tray_id",
                               mode_bin_mv: float | None = None) -> pd.DataFrame:
    """[mode 전환 위험 완화] 트레이별 median 중심값과 mode 중심값을 나란히 비교.

    두 통계가 크게 갈리는 트레이가 있으면 그 트레이에서 mode 가 양자화로 불안정하다는
    신호다(과거 랏에서 필터만 바꿔도 16셀 판정이 뒤집힌 사례 참고). 반환은 abs_diff
    내림차순 — 위쪽 행이 가장 위험한 트레이.
    """
    bin_mv = mode_bin_mv
    if bin_mv is None:
        bin_mv = freedman_diaconis_bin(df[value_col])
        if not np.isfinite(bin_mv) or bin_mv <= 0:
            bin_mv = 0.01

    g = df.groupby(by)[value_col]
    tmp = df[[by]].copy()
    tmp["median_center"] = g.transform("median")
    tmp["mode_center"] = g.transform(lambda s: mode_estimate(s, bin_mv=bin_mv))
    out = tmp.drop_duplicates(subset=[by]).reset_index(drop=True)
    out["diff"] = out["mode_center"] - out["median_center"]
    out["abs_diff"] = out["diff"].abs()
    return out.sort_values("abs_diff", ascending=False).reset_index(drop=True)


def to_grid(df: pd.DataFrame, value_col: str, row_col: str = "row",
            col_col: str = "col") -> np.ndarray:
    """단일 트레이 부분집합 → (12,12) 격자. 중복 위치는 마지막 값이 남는다."""
    grid = np.full((N_ROWS, N_COLS), np.nan)
    r = pd.to_numeric(df[row_col], errors="coerce")
    c = pd.to_numeric(df[col_col], errors="coerce")
    v = pd.to_numeric(df[value_col], errors="coerce")
    ok = r.notna() & c.notna() & v.notna()
    for ri, ci, vi in zip(r[ok].astype(int), c[ok].astype(int), v[ok]):
        if 1 <= ri <= N_ROWS and 1 <= ci <= N_COLS:
            grid[ri - 1, ci - 1] = vi
    return grid


def position_field(df: pd.DataFrame, value_col: str, agg: str = "mean",
                   row_col: str = "row", col_col: str = "col") -> pd.DataFrame:
    """전 트레이에 걸쳐 위치별로 집계한 (12,12) 필드. agg: mean/median/rms/std/count.

    RMS 는 부호 무관 진폭 — 링/역링이 상쇄되는 mean 필드와 반드시 병기한다
    (docs/01_analysis_plan.md E1).
    """
    d = df[[row_col, col_col, value_col]].copy()
    d[row_col] = pd.to_numeric(d[row_col], errors="coerce")
    d[col_col] = pd.to_numeric(d[col_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna()

    if agg == "rms":
        out = d.groupby([row_col, col_col])[value_col].apply(
            lambda s: float(np.sqrt(np.mean(np.square(s)))))
    elif agg == "count":
        out = d.groupby([row_col, col_col])[value_col].count()
    else:
        out = d.groupby([row_col, col_col])[value_col].agg(agg)

    grid = np.full((N_ROWS, N_COLS), np.nan)
    for (ri, ci), vi in out.items():
        ri, ci = int(ri), int(ci)
        if 1 <= ri <= N_ROWS and 1 <= ci <= N_COLS:
            grid[ri - 1, ci - 1] = vi
    return pd.DataFrame(grid, index=range(1, N_ROWS + 1), columns=list("ABCDEFGHIJKL"))


def edge_distance() -> np.ndarray:
    """(12,12) 각 셀의 테두리까지 최단거리(0=테두리)."""
    rr, cc = np.mgrid[0:N_ROWS, 0:N_COLS]
    return np.minimum.reduce([rr, N_ROWS - 1 - rr, cc, N_COLS - 1 - cc])


def cell_edge_distance(df: pd.DataFrame, row_col: str = "row",
                       col_col: str = "col") -> pd.Series:
    """[오류6 수정용] 각 셀의 (row,col) 에 대응하는 테두리 거리(0~5).

    등방적 중앙-가장자리 성분을 제거(detrend)하는 데 쓴다 — F2 밴드 비교가 이 성분과
    팬 성분을 못 가르는 문제(docs/04_logic_audit.md 오류6)의 해결책.
    """
    grid = edge_distance()
    row = pd.to_numeric(df[row_col], errors="coerce")
    col = pd.to_numeric(df[col_col], errors="coerce")
    out = pd.Series(np.nan, index=df.index)
    ok = row.notna() & col.notna()
    if not ok.any():
        return out
    r = row[ok].astype(int).to_numpy() - 1
    c = col[ok].astype(int).to_numpy() - 1
    valid = (r >= 0) & (r < N_ROWS) & (c >= 0) & (c < N_COLS)
    idx = df.index[ok][valid]
    out.loc[idx] = grid[r[valid], c[valid]]
    return out


def ring_score(grid: np.ndarray) -> float:
    """테두리 평균 − 중앙 평균. 양수면 링(테두리 高), 음수면 역링."""
    d = edge_distance()
    border = np.isfinite(grid) & (d == 0)
    center = np.isfinite(grid) & (d >= 2)
    if not border.any() or not center.any():
        return np.nan
    return float(np.nanmean(grid[border]) - np.nanmean(grid[center]))


def pairwise_corr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """NaN 쌍삭제(pairwise-complete) Pearson 상관행렬. scipy 없이 numpy 로."""
    n = len(cols)
    mat = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            a = pd.to_numeric(df[cols[i]], errors="coerce")
            b = pd.to_numeric(df[cols[j]], errors="coerce")
            ok = a.notna() & b.notna()
            if ok.sum() < 3:
                continue
            av, bv = a[ok].to_numpy(), b[ok].to_numpy()
            if av.std() < 1e-12 or bv.std() < 1e-12:
                r = 1.0 if i == j else np.nan
            else:
                r = float(np.corrcoef(av, bv)[0, 1])
            mat[i, j] = mat[j, i] = r
    return pd.DataFrame(mat, index=cols, columns=cols)


def percentile_rank(x: pd.Series) -> pd.Series:
    """0~1 백분위 순위 (NaN 은 NaN 유지)."""
    return x.rank(pct=True, na_option="keep")
