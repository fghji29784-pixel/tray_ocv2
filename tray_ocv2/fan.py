"""충방전기 냉각팬 서명 검증 — docs/01_analysis_plan.md F 그룹.

온도로 먼저 검증한다(F0~F3). 전압은 SOC·자기방전 등이 섞이지만 온도는 순수한 열장이다.
예측값은 schema.py 에 데이터 라벨(세로 1~12, 가로 A~L)로 못박아 두었다.

★ docs/04_logic_audit.md 오류6·7·10 을 반영해 재작성했다:
  - 오류6: F2 밴드1·3(상·하)은 둘 다 '가장자리' → 팬과 무관한 중앙-가장자리 효과만으로도
    '상단≈하단≠중앙'이 나온다. edge_detrend() 로 그 성분을 먼저 제거한 뒤 비교한다.
    그리고 schema.FAN_COL_MIN_OUTER/MIDDLE(팬 고유 예측)을 predicted_minima_check() 로
    직접 검정한다 — band_similarity 만으로는 팬을 고유하게 식별하지 못한다.
  - 오류7: F0 옛 기준("모든 스텝이 같아야")이 F3과 모순됐다. cluster_reproducibility() 로
    "열이력이 같은 스텝끼리만 재현되어야 한다"로 재정의했다.
  - 오류10: 발열/무발열 이분법(구 amplitude_vs_heat)은 LCI 등 교란에 취약하다.
    heat_vs_amplitude_regression() 으로 실제 발열량(연속량) 대 진폭을 회귀한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema
from .fields import cell_edge_distance, position_field, tray_delta
from .spatial import build_tray_matrix, pca_modes


def row_profile(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """[F1] 세로(1~12) 프로파일. 트레이 내 편차 기준, 평균±표준오차.

    ⚠️ 여기엔 등방 중앙-가장자리 성분이 그대로 섞여 있다(그 자체가 관측 대상이기도
    하다). 팬 고유 성분만 보려면 F2(edge_detrend 적용)를 봐야 한다.
    """
    d = df.copy()
    d["_dev"] = tray_delta(d, value_col, stat="median")
    g = d.groupby("row")["_dev"]
    out = pd.DataFrame({"row": g.mean().index, "mean": g.mean().to_numpy(),
                        "se": (g.std() / np.sqrt(g.count())).to_numpy(),
                        "n": g.count().to_numpy()})
    out["is_fan_axis_max"] = out["row"].isin(schema.FAN_ROW_MAX)
    out["is_fan_axis_min"] = out["row"].isin(schema.FAN_ROW_MIN)
    return out.sort_values("row", ignore_index=True)


def edge_detrend(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """[오류6 수정] 테두리거리별 평균(등방 중앙-가장자리 성분)을 뺀 잔차(_resid) 추가.

    F2 밴드1(세로1~4)·밴드3(세로9~12)은 둘 다 '가장자리'다. 팬과 무관한 어떤
    중앙-가장자리 열전달 효과도 '밴드1≈밴드3≠밴드2(중앙)'를 만든다. 이 성분을
    먼저 제거해야 남는 신호가 팬 고유(주기적) 성분에 가까워진다.
    """
    out = df.copy()
    out["_dev"] = tray_delta(out, value_col, stat="median")
    out["_edge_d"] = cell_edge_distance(out)
    edge_profile = out.groupby("_edge_d")["_dev"].transform("mean")
    out["_resid"] = out["_dev"] - edge_profile
    return out


def band_column_profile(df: pd.DataFrame, value_col: str, detrend: bool = True) -> dict:
    """[F2] 밴드별(상/중/하) 가로(A~L) 프로파일.

    detrend=True(기본): edge_detrend 잔차 사용 — 중앙-가장자리 성분 제거 후 비교.
    detrend=False: 옛 방식(등방 성분 포함) — 비교용으로만 남겨둔다.
    """
    if detrend:
        d = edge_detrend(df, value_col)
        col_for_profile = "_resid"
    else:
        d = df.copy()
        d["_dev"] = tray_delta(d, value_col, stat="median")
        col_for_profile = "_dev"

    profiles = {}
    for band in schema.FAN_BANDS:
        r0, r1 = band["rows"]
        sub = d[(d["row"] >= r0) & (d["row"] <= r1)]
        g = sub.groupby("col")[col_for_profile]
        prof = pd.DataFrame({"col": g.mean().index, "mean": g.mean().to_numpy(),
                            "n": g.count().to_numpy()}).sort_values("col")
        prof["col_label"] = prof["col"].map(
            lambda c: chr(ord("A") + int(c) - 1) if pd.notna(c) else None)
        profiles[band["name"]] = prof
    return profiles


def band_similarity(profiles: dict) -> dict:
    """[F2] 상단 vs 중앙, 하단 vs 중앙, 상단 vs 하단 프로파일 상관.

    ⚠️ detrend 적용 후에도, 이 지표 하나만으로는 팬을 '고유하게' 식별하지 못한다
    (다른 비등방 원인도 같은 패턴을 만들 수 있음). predicted_minima_check() 와
    함께 봐야 한다 — 오류6 참조.
    """
    names = list(profiles)
    if len(names) < 3:
        return {}
    merged = {}
    for name in names:
        p = profiles[name].set_index("col")["mean"]
        merged[name] = p
    aligned = pd.DataFrame(merged).dropna()
    if len(aligned) < 3:
        return {"ok": False, "reason": "공통 열 표본 부족"}

    def corr(a, b):
        if aligned[a].std() < 1e-12 or aligned[b].std() < 1e-12:
            return np.nan
        return float(np.corrcoef(aligned[a], aligned[b])[0, 1])

    top, mid, bot = names[0], names[1], names[2]
    return {
        "ok": True,
        f"{top}_vs_{bot}": corr(top, bot),
        f"{top}_vs_{mid}": corr(top, mid),
        f"{bot}_vs_{mid}": corr(bot, mid),
        "note": ("detrend 적용됨(기본). 이 지표는 참고용 — 팬 고유 식별은 "
                "predicted_minima_check() 로 별도 검정할 것"),
    }


def predicted_minima_check(profiles: dict) -> dict:
    """[오류6 수정] ★판결문 대체 — schema.FAN_COL_COLD_OUTER/MIDDLE 예측을 직접 검정.

    band_similarity(상관)는 팬과 다른 비등방 원인을 못 가른다. 여기서는 도면에서
    미리 못박은 '냉각 최강(차가움) 예측 열'(팬 중심에 가장 가까운 열)의 평균이,
    detrend 잔차 기준으로 나머지 열보다 실제로 낮은지(더 차가운지)를 직접 확인한다 —
    팬 기하 자체의 사전등록 예측이라 band_similarity보다 식별력이 높다.
    """
    band_pred = {"상단": schema.FAN_COL_COLD_OUTER, "하단": schema.FAN_COL_COLD_OUTER,
                "중앙": schema.FAN_COL_COLD_MIDDLE}
    results = {}
    for name, prof in profiles.items():
        pred_cols = band_pred.get(name, [])
        if not pred_cols or prof.empty:
            continue
        is_pred = prof["col_label"].isin(pred_cols)
        if is_pred.sum() == 0 or (~is_pred).sum() == 0:
            continue
        pred_mean = float(prof.loc[is_pred, "mean"].mean())
        other_mean = float(prof.loc[~is_pred, "mean"].mean())
        results[name] = {
            "predicted_cold_cols": pred_cols,
            "predicted_mean": pred_mean, "other_cols_mean": other_mean,
            "colder_as_predicted": bool(pred_mean < other_mean),
        }
    results["note"] = ("등방 성분(테두리거리) 제거된 잔차 기준. 예측 열의 평균이 "
                       "나머지보다 낮아야(더 차가워야) 팬 도면 예측에 부합한다. "
                       "band_similarity 와 반드시 함께 본다")
    return results


def fan_radial_profile(df: pd.DataFrame, value_col: str, n_bins: int = 8,
                       max_r: float = 4.0) -> dict:
    """[F4] ★부호 논쟁을 가르는 결정적 검정 — 가장 가까운 팬 중심까지의 거리별 프로파일.

    배경(docs/04_logic_audit.md 오류12): 등록했던 예측 '팬 중심 = 냉각 최강(차가움)'이
    Run #2 에서 3밴드 전부 반대로 나왔다. 두 해석이 가능하다 —
      (a) 팬 가설이 틀렸다
      (b) 부호 규약이 틀렸다: 충돌제트는 팬 **허브 바로 아래가 정체점(유속 최소)** 이라
          중심이 오히려 덜 식는다. 최대 냉각은 반경 0.5~1D 의 고리에서 일어난다.

    ⚠️ 결과를 보고 부호만 뒤집는 것은 순환 논리다. (b)가 맞다면 단순 부호 반전으로는
    설명 못 하는 **고유한 형상**이 나와야 한다: 중심에서 극값 → 중간 반경에서 반대 극값
    → 바깥에서 감쇠(비단조, 고리 구조). 단조 프로파일이면 (b)는 지지되지 않는다.

    반환의 `is_monotonic` 이 False 이고 `extremum_bin_idx` 가 중간 구간이면 고리 구조.
    """
    d = edge_detrend(df, value_col)          # 등방 중앙-가장자리 성분 제거 후
    rows = pd.to_numeric(d["row"], errors="coerce").to_numpy(dtype=float)
    cols = pd.to_numeric(d["col"], errors="coerce").to_numpy(dtype=float)

    centers = [(b["row_center"], c) for b in schema.FAN_BANDS for c in b["col_centers"]]
    dist = np.full(len(d), np.inf)
    for rc, cc in centers:
        dist = np.minimum(dist, np.sqrt((rows - rc) ** 2 + (cols - cc) ** 2))

    d = d.assign(_r=dist)
    valid = np.isfinite(d["_r"]) & d["_resid"].notna() & (d["_r"] <= max_r)
    sub = d[valid]
    if len(sub) < 200:
        return {"ok": False, "reason": "유효 표본 부족"}

    edges = np.linspace(0, max_r, n_bins + 1)
    labels = pd.cut(sub["_r"], bins=edges, include_lowest=True)
    g = sub.groupby(labels, observed=True)["_resid"]
    table = pd.DataFrame({
        "r_mid": [iv.mid for iv in g.mean().index],
        "mean": g.mean().to_numpy(),
        "se": (g.std() / np.sqrt(g.count())).to_numpy(),
        "n": g.count().to_numpy(),
    }).sort_values("r_mid", ignore_index=True)

    info = _monotonic_info(table)
    if not info.get("ok"):
        return {"ok": False, "reason": "구간 수 부족", "table": table}

    return {
        "ok": True, "table": table,
        "center_mean": info["center_mean"],
        "extremum_bin_idx": int(np.argmax(np.abs(table["mean"].to_numpy()
                                                 - info["center_mean"]))),
        "extremum_r_mid": info["extremum_d_mid"], "extremum_mean": info["extremum_mean"],
        "is_monotonic": info["is_monotonic"], "trend_corr": info["trend_corr"],
        "note": ("is_monotonic 하나만 보지 말 것 — 끝단 표본이 적은 구간의 잡음만으로도 "
                "뒤집힌다. trend_corr(거리-값 선형상관)이 크면(|r|>0.8 등) 잡음이 있어도 "
                "전반적으로는 단조에 가깝다는 뜻. 비단조 + trend_corr 도 약하고 극값이 "
                "중간 반경이면 충돌제트 고리 구조에 부합 → 해석(b) 지지. 결과 보고 "
                "부호만 뒤집는 사후 구제를 막기 위한 독립 검정"),
    }


def _monotonic_info(t: pd.DataFrame) -> dict:
    """반경/거리 프로파일의 단조성 판정.

    ⚠️ `is_monotonic`(구간별 부호 전부 일치)은 이진 판정이라 끝단의 표본이 적은
    구간 하나의 잡음만으로도 뒤집힌다(합성 데이터로 실제 발견 — 단조로 설계했는데
    맨 끝 구간에서 근소하게 흔들려 False 로 나옴). 오류12에서 지적한 "사전등록
    검정의 판별력 부족"과 같은 종류의 문제라 여기서도 보강한다: 구간중점과 평균값의
    **선형 상관(trend_corr)** 을 같이 낸다 — 전반적 추세는 단일 잡음 구간에 흔들리지
    않는다. 해석 시 `is_monotonic` 하나만 보지 말고 `trend_corr` 크기와 함께 볼 것
    (|trend_corr| 가 크면(예 >0.8) 잡음이 있어도 전반적으로는 단조에 가깝다는 뜻).
    """
    y = t["mean"].to_numpy()
    x = t.iloc[:, 0].to_numpy(dtype=float)
    if len(y) < 4:
        return {"ok": False}
    diffs = np.diff(y)
    is_mono = bool(np.all(diffs >= 0) or np.all(diffs <= 0))
    trend_corr = (float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 1e-12 else np.nan)
    extremum_idx = int(np.argmax(np.abs(y - y[0])))
    return {
        "ok": True, "is_monotonic": is_mono, "trend_corr": trend_corr,
        "center_mean": float(y[0]),
        "extremum_d_mid": float(t.iloc[extremum_idx, 0]),
        "extremum_mean": float(y[extremum_idx]),
    }


def fan_radial_profile_anisotropic(df: pd.DataFrame, value_col: str,
                                   n_bins: int = 6) -> dict:
    """[F4 이방성 분리] ★행/열 방향을 따로 — 밴드경계 효과와 팬간격 효과를 가른다.

    등방 반경(fan_radial_profile, F4)은 여러 밴드에 걸친 대각선 거리를 하나로 뭉쳐서,
    '밴드 경계를 넘어가며 생기는 효과'와 '한 밴드 안에서 팬 사이 간격 때문에 생기는
    효과'를 구분하지 못한다(Run #3에서 r=1.75/2.25 의 비단조 패턴이 어느 쪽인지
    불명확했던 이유).

    각 셀을 먼저 자신이 속한 밴드로 배정한 뒤, **같은 밴드 안에서만**
      ① 세로 거리 = |row − 그 밴드의 row_center|  (밴드 폭 내부로 자동 한정)
      ② 가로 거리 = |col − 그 밴드 내 가장 가까운 col_center|  (같은 밴드 내부로 한정,
         옆 밴드로 안 넘어감)
    를 따로 프로파일링한다. 세로축은 '허브 정체점' 가설을, 가로축은 '팬 사이 간격'
    가설을 밴드경계 오염 없이 각각 순수하게 검정한다.
    """
    d = edge_detrend(df, value_col)
    rows = pd.to_numeric(d["row"], errors="coerce").to_numpy(dtype=float)
    cols = pd.to_numeric(d["col"], errors="coerce").to_numpy(dtype=float)

    d_row = np.full(len(d), np.nan)
    d_col = np.full(len(d), np.nan)
    for band in schema.FAN_BANDS:
        r0, r1 = band["rows"]
        mask = (rows >= r0) & (rows <= r1)
        if not mask.any():
            continue
        d_row[mask] = np.abs(rows[mask] - band["row_center"])
        centers = np.asarray(band["col_centers"])
        d_col[mask] = np.min(np.abs(cols[mask][:, None] - centers[None, :]), axis=1)

    d = d.assign(_d_row=d_row, _d_col=d_col)
    valid = np.isfinite(d["_d_row"]) & np.isfinite(d["_d_col"]) & d["_resid"].notna()
    sub = d[valid]
    if len(sub) < 200:
        return {"ok": False, "reason": "유효 표본 부족"}

    def _profile(distcol, maxd):
        edges = np.linspace(0, maxd, n_bins + 1)
        labels = pd.cut(sub[distcol], bins=edges, include_lowest=True)
        g = sub.groupby(labels, observed=True)["_resid"]
        return pd.DataFrame({
            "d_mid": [iv.mid for iv in g.mean().index],
            "mean": g.mean().to_numpy(),
            "se": (g.std() / np.sqrt(g.count())).to_numpy(),
            "n": g.count().to_numpy(),
        }).sort_values("d_mid", ignore_index=True)

    row_prof = _profile("_d_row", float(np.nanmax(sub["_d_row"])) or 2.0)
    col_prof = _profile("_d_col", float(np.nanmax(sub["_d_col"])) or 2.0)

    return {
        "ok": True,
        "row_axis_profile": row_prof, "col_axis_profile": col_prof,
        "row_axis_summary": _monotonic_info(row_prof),
        "col_axis_summary": _monotonic_info(col_prof),
        "note": ("row_axis = 밴드 폭 내부 세로거리(허브 정체점 가설 순수 검정). "
                "col_axis = 같은 밴드 내부 가로거리(팬간격 가설 순수 검정, 밴드경계 "
                "안 넘어감). 등방 F4의 비단조성이 어느 축에서 오는지 여기서 가른다"),
    }


def classify_temp_cols(temp_cols: list[str]) -> dict:
    """[오류7] 컬럼명(t_<key> 또는 temp_<key>_<stat>) → schema.classify_thermal_state 매핑.

    분류 결과를 카테고리별 컬럼 리스트로 묶어 반환한다.
    """
    def _key(col: str) -> str | None:
        if col.startswith("t_"):
            return col[2:]
        if col.startswith("temp_"):
            rest = col[len("temp_"):]
            for suf in ("_min", "_mean", "_max", "_single"):
                if rest.endswith(suf):
                    return rest[: -len(suf)]
            return rest
        return None

    groups: dict = {}
    for c in temp_cols:
        key = _key(c)
        cat = schema.classify_thermal_state(key) if key else "unknown"
        groups.setdefault(cat, []).append(c)
    return groups


def cluster_reproducibility(corr: pd.DataFrame) -> dict:
    """[오류7 수정] F0 판정 기준 재정의.

    옛 기준("모든 스텝에서 같은 무늬")은 F3(발열이 있어야 구배가 생김)과 모순된다.
    실측(Run #1)도 'equilibrium'(에이징 직후 열평형)과 'recent_heat'(충방전 직후
    잔열) 두 군집으로 뚜렷이 갈렸다(군집 내 |r|>0.7, 군집 간 |r|<0.3).
    올바른 기준: recent_heat 끼리는 강하게 재현되고, 두 부류 사이는 약해야 한다.
    """
    groups = classify_temp_cols(list(corr.index))

    def _avg(pairs):
        vals = [corr.loc[a, b] for a, b in pairs if a in corr.index and b in corr.columns]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else np.nan

    within = {}
    for cat, cols in groups.items():
        pairs = [(a, b) for i, a in enumerate(cols) for b in cols[i + 1:]]
        within[cat] = _avg(pairs)

    cats = [c for c in groups if len(groups[c]) > 0]
    cross_pairs = []
    for i, c1 in enumerate(cats):
        for c2 in cats[i + 1:]:
            cross_pairs += [(a, b) for a in groups[c1] for b in groups[c2]]
    cross = _avg(cross_pairs)

    return {
        "within_by_category": within, "cross_category_mean": cross,
        "n_by_category": {k: len(v) for k, v in groups.items()},
        "note": ("within_by_category['recent_heat'] 가 cross_category_mean 보다 "
                "뚜렷이 크면(예: >0.5 vs <0.3) 재정의된 F0 기준 충족. "
                "'모든 스텝이 같아야' 라는 옛 기준은 폐기"),
    }


def stage_field_reproducibility(df: pd.DataFrame, temp_cols: list[str]) -> pd.DataFrame:
    """[F0] 여러 단계의 온도 필드(트레이 편차)가 서로 재현되는가 — 원시 상관행렬.

    판정은 이 행렬 자체가 아니라 cluster_reproducibility() 의 군집별 요약으로 한다
    (오류7 — '모든 스텝이 같아야'는 틀린 기준).
    """
    d = df.copy()
    dev_cols = []
    for c in temp_cols:
        if c not in d.columns:
            continue
        dc = f"_dev_{c}"
        d[dc] = tray_delta(d, c, stat="median")
        dev_cols.append(dc)
    if len(dev_cols) < 2:
        return pd.DataFrame()

    fields = {}
    for dc in dev_cols:
        f = position_field(d, dc, agg="mean")
        fields[dc.replace("_dev_", "")] = f.to_numpy().ravel()
    keys = list(fields)
    n = len(keys)
    mat = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            a, b = fields[keys[i]], fields[keys[j]]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 20:
                continue
            av, bv = a[ok], b[ok]
            if av.std() < 1e-12 or bv.std() < 1e-12:
                r = 1.0 if i == j else np.nan
            else:
                r = float(np.corrcoef(av, bv)[0, 1])
            mat[i, j] = mat[j, i] = r
    return pd.DataFrame(mat, index=keys, columns=keys)


def _rms_amp(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return np.nan
    d = df.copy()
    d["_dev"] = tray_delta(d, col, stat="median")
    f = position_field(d, "_dev", agg="rms").to_numpy()
    return float(np.nanmean(f))


def amplitude_vs_heat(df: pd.DataFrame, no_heat_cols: list[str],
                      heat_cols: list[str]) -> dict:
    """[F3-구버전] 무발열 vs 발열 이분법 진폭(RMS) 비교. 참고용으로 유지.

    ⚠️ 오류10: 이분법이 실제 발열량 차이를 반영 못 하는 경우가 있다(예: 1차 충전은
    전류가 작아 발열이 적을 수 있음 — Run #1 에서 temp_charge1_mean 진폭이 '무발열'
    기준 t_ocv1 보다도 작게 나온 실례). 주 판정은 heat_vs_amplitude_regression() 로 한다.
    호출 시 no_heat_cols 에서 'separate_instrument'(LCI 등)로 분류되는 스텝은 미리
    제외할 것 — 안 그러면 팬과 무관한 이상치가 '무발열' 평균을 왜곡한다(오류10 실례).
    """
    no_heat = {c: _rms_amp(df, c) for c in no_heat_cols}
    heat = {c: _rms_amp(df, c) for c in heat_cols}
    no_heat_mean = np.nanmean(list(no_heat.values())) if no_heat else np.nan
    heat_mean = np.nanmean(list(heat.values())) if heat else np.nan
    return {
        "no_heat_stage_amplitudes": no_heat, "heat_stage_amplitudes": heat,
        "no_heat_mean_rms": no_heat_mean, "heat_mean_rms": heat_mean,
        "ratio_heat_over_no_heat": (heat_mean / no_heat_mean
                                    if no_heat_mean and no_heat_mean > 0 else np.nan),
        "note": "참고용 이분법 비교. 주 판정은 heat_vs_amplitude_regression() 를 볼 것",
    }


def heat_vs_amplitude_regression(df: pd.DataFrame,
                                 rise_stat_cols: list[tuple]) -> dict:
    """[오류10 수정] ★F3 주 판정 — 발열량(연속량) 대 위치필드 진폭(RMS) 회귀.

    발열/무발열 이분법 대신, 각 스텝의 실제 자기발열 크기(최고−최저 온도 중앙값)를
    가로축으로, 그 스텝 온도필드의 RMS 진폭을 세로축으로 놓고 상관·기울기를 본다.
    양의 상관·양의 기울기가 뚜렷하면 F3(진폭∝발열량) 이 이분법보다 강한 증거로 지지된다.

    rise_stat_cols: [(mean_col, min_col, max_col, label), ...] — Charge/DisCharge 처럼
    최저/평균/최고 3통계가 있는 스텝만 대상이 된다.
    """
    rows = []
    for mean_col, min_col, max_col, label in rise_stat_cols:
        if not all(c in df.columns for c in (mean_col, min_col, max_col)):
            continue
        rise = (pd.to_numeric(df[max_col], errors="coerce")
               - pd.to_numeric(df[min_col], errors="coerce"))
        amp = _rms_amp(df, mean_col)
        rows.append({"stage": label, "median_rise_c": float(rise.median()),
                    "field_rms": amp})
    table = pd.DataFrame(rows)
    if len(table) < 4:
        return {"table": table, "ok": False, "reason": "회귀에 필요한 스텝 수 부족(<4)"}

    x = table["median_rise_c"].to_numpy()
    y = table["field_rms"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or np.std(x[ok]) < 1e-9:
        return {"table": table, "ok": False, "reason": "유효 표본·분산 부족"}

    r = float(np.corrcoef(x[ok], y[ok])[0, 1])
    slope, intercept = (float(v) for v in np.polyfit(x[ok], y[ok], 1))
    return {
        "table": table, "ok": True, "corr": r, "slope": slope, "intercept": intercept,
        "note": ("발열량(최고-최저 중앙값, °C) vs 필드 진폭(RMS) 의 스텝간 상관·기울기. "
                "이분법(F3-구버전) 대체. corr·slope 모두 뚜렷한 양수여야 F3 지지"),
    }


# ---------------------------------------------------------------------------
# 호기별(충방전기 2~5호) 기류 분석 — docs/00_facts.md §3.3 (2026-07-31 확인)
#
# 2·3호는 트레이 아래 6×6 바닥팬 + 앞뒤 외기, 4·5호는 윗팬(4-5-4, 하향) + 바닥
# 4×4(상향) + 외기(4호=행1쪽 모서리, 5호=행12쪽 모서리). ★모두 세로(행 1~12)축
# — 처음엔 "측면"이라는 표현 때문에 가로(A~L)축으로 예측했으나, 실측(아래
# charger_directional_profile 결과)과 사용자 확인으로 세로축이 맞다고 정정됨
# (2026-07-31). schema.charger_from_box()/schema.parse_box() 로 뽑은
# charger_<stage_key> 컬럼(값 2~5)으로 층화해 검증한다.
# ---------------------------------------------------------------------------


def _tray_uid_col(df: pd.DataFrame) -> pd.Series:
    """여러 랏을 합쳐도 트레이ID 충돌이 없는 고유 키. `product_lot` 있으면 그것과 결합.

    docs/00_facts.md §2.1 Q6(트레이 ID의 랏 간 재사용 여부 미확인) 때문에, 랏을 여러 개
    합칠 때 `tray_id` 단독으로 묶으면 서로 다른 물리 트레이가 섞일 위험이 있다.
    """
    if "product_lot" in df.columns:
        return df["product_lot"].astype(str) + "_" + df["tray_id"].astype(str)
    return df["tray_id"].astype(str)


def _linreg(x, y) -> tuple[float, float]:
    """단순 선형회귀 기울기·상관계수. scipy 없이 numpy만(§00_facts 실행환경 제약)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) < 1e-9:
        return np.nan, np.nan
    slope, _intercept = np.polyfit(x[ok], y[ok], 1)
    r = float(np.corrcoef(x[ok], y[ok])[0, 1])
    return float(slope), r


def charger_directional_profile(df: pd.DataFrame, value_col: str,
                                charger_col: str) -> pd.DataFrame:
    """[②방향성 구배] 호기별 가로(A~L)·세로(1~12) 위치편차 기울기.

    §3.3 예측(2026-07-31 실측 확정)의 직접 검정: **세로(행) 기울기**가 4호는 뚜렷한
    양수(행1쪽 외기 → 행1이 차가움), 5호는 뚜렷한 음수(행12쪽 외기, 4호 반전), 2·3호는
    같은 축이지만 더 약해야 한다. (가로/열 축은 처음엔 "측면"이라는 표현 때문에
    예측했었으나 실측에서 기각됨 — 외기는 세로축으로 들어온다)
    """
    d = df.copy()
    d["_uid"] = _tray_uid_col(d)
    d["_dev"] = tray_delta(d, value_col, by="_uid", stat="median")
    d = d.dropna(subset=["_dev", "row", "col", charger_col])

    rows = []
    for charger, sub in d.groupby(charger_col):
        col_mean = sub.groupby("col")["_dev"].mean()
        row_mean = sub.groupby("row")["_dev"].mean()
        col_slope, col_corr = _linreg(col_mean.index.to_numpy(), col_mean.to_numpy())
        row_slope, row_corr = _linreg(row_mean.index.to_numpy(), row_mean.to_numpy())
        rows.append({
            "charger": int(charger),
            "n_trays": int(sub["_uid"].nunique()), "n_cells": int(len(sub)),
            "col_slope_per_col": col_slope, "col_corr": col_corr,
            "row_slope_per_row": row_slope, "row_corr": row_corr,
        })
    out = pd.DataFrame(rows).sort_values("charger", ignore_index=True)
    return out.assign(note=(
        "col_slope: 열(A=1~L=12) 방향 기울기(값/열). row_slope: 행(1~12) 방향 기울기. "
        "★핵심 지표는 row_slope — 4호=양수·5호=음수(거울상) 예측, 크기는 2·3호보다 커야 함. "
        "col_slope 는 참고용(2026-07-31 실측: 4개 호기 전부 약함, 외기와 무관 확인됨)"))


def charger_eof_mode1(df: pd.DataFrame, value_col: str, charger_col: str,
                      n_modes: int = 1) -> dict:
    """[③EOF 모드1] 호기별 지배 공간 패턴을 링/역링 가정 없이 데이터에서 직접 뽑는다.

    링·역링을 미리 정해두고 맞는지 보는 대신, 각 호기의 (트레이×144위치) 행렬을
    PCA(EOF, numpy SVD)로 분해해 실제로 가장 많이 반복되는 무늬 자체를 얻는다.
    """
    d = df.copy()
    d["_uid"] = _tray_uid_col(d)
    d["_dev"] = tray_delta(d, value_col, by="_uid", stat="median")
    d = d.dropna(subset=[charger_col])

    out = {}
    for charger, sub in d.groupby(charger_col):
        _, mat = build_tray_matrix(sub, value_col="_dev", tray_col="_uid",
                                   already_relative=True)
        out[int(charger)] = pca_modes(mat, n_modes=n_modes)
    return out


def charger_diff_map(df: pd.DataFrame, value_col: str, charger_col: str,
                     charger_a: int = 4, charger_b: int = 5) -> dict:
    """[⑥4-5호 차분] 두 호기의 위치 지도를 직접 뺀다.

    4·5호는 바닥 4×4 팬 등 공통 요소가 많다 — 그대로 빼면 공통분은 상쇄되고
    **외기 방향 차이(4호=행1쪽, 5호=행12쪽)만 남아야 한다**(세로 방향 비대칭 예측).
    2026-07-31 실측 확정: 행1쪽 파랑(4호가 낮음)·행12쪽 빨강(4호가 높음=5호가 낮음)으로
    깨끗한 세로 경사 확인, 가로 방향 구조는 거의 없었다.
    """
    d = df.copy()
    d["_uid"] = _tray_uid_col(d)
    d["_dev"] = tray_delta(d, value_col, by="_uid", stat="median")

    sub_a = d[d[charger_col] == charger_a]
    sub_b = d[d[charger_col] == charger_b]
    if sub_a.empty or sub_b.empty:
        return {"ok": False, "reason": f"호기 {charger_a} 또는 {charger_b} 데이터 없음"}

    field_a = position_field(sub_a, "_dev", agg="mean")
    field_b = position_field(sub_b, "_dev", agg="mean")
    diff = field_a - field_b
    return {
        "ok": True, "charger_a": charger_a, "charger_b": charger_b,
        "field_a": field_a, "field_b": field_b, "diff": diff,
        "n_trays_a": int(sub_a["_uid"].nunique()), "n_trays_b": int(sub_b["_uid"].nunique()),
        "note": "diff = field_a - field_b. 세로(행) 방향 경사가 남아야 외기 가설 지지(실측 확인됨, "
               "2026-07-31 — 처음엔 가로 방향으로 예측했으나 세로축이 맞았다)",
    }


#: [⑧] 안쪽 4×4 박스 꼭지점(팬격자 접점 후보) / 바깥 트레이 꼭지점. (열문자, 행) 좌표.
SPECIAL_CELL_GROUPS = {
    "inner_corners": [("E", 5), ("E", 8), ("H", 5), ("H", 8)],
    "outer_corners": [("A", 1), ("A", 12), ("L", 1), ("L", 12)],
}


def charger_special_cell_contrast(df: pd.DataFrame, value_col: str,
                                  charger_col: str) -> pd.DataFrame:
    """[⑧8셀 대조] 안쪽 4×4 꼭지점·바깥 꼭지점이 호기별로도 이상값인지 대조.

    가설(2026-07-31 실측으로 기각됨): 이 8셀이 순수 바닥 팬격자 아티팩트라면 4×4 격자인
    4·5호에서 강하고 6×6 격자인 2·3호에서 약해야 하는데, 실제로는 **6×6인 3호가 가장
    강하게** 나왔다 — "바닥 팬격자" 원인은 기각, 정체는 미상. 대신 **바깥 꼭지점**이
    안쪽보다 훨씬 크게(배율 7~11배) 나오고 **2호만 부호가 반대**였다 — 이쪽이 더 큰
    미해결 질문으로 남는다. `vs_typical_abs_ratio` 가 1보다 훨씬 크면 "그 호기에서 이
    자리가 유독 튄다"는 뜻.
    """
    d = df.copy()
    d["_uid"] = _tray_uid_col(d)
    d["_dev"] = tray_delta(d, value_col, by="_uid", stat="median")
    d = d.dropna(subset=[charger_col])

    rows = []
    for charger, sub in d.groupby(charger_col):
        field = position_field(sub, "_dev", agg="mean")
        arr = field.to_numpy()
        typical_abs = float(np.nanmedian(np.abs(arr)))
        for group_name, cells in SPECIAL_CELL_GROUPS.items():
            vals = [field.loc[row, col] for col, row in cells
                   if row in field.index and col in field.columns]
            if not vals or typical_abs < 1e-12:
                continue
            mean_val = float(np.nanmean(vals))
            rows.append({
                "charger": int(charger), "cell_group": group_name,
                "mean_dev": mean_val,
                "vs_typical_abs_ratio": abs(mean_val) / typical_abs,
                "n_trays": int(sub["_uid"].nunique()),
            })
    return pd.DataFrame(rows).sort_values(["cell_group", "charger"], ignore_index=True)


def charger_stage_report(df: pd.DataFrame, value_col: str, charger_col: str) -> dict:
    """[편의 함수] 한 스텝(예: t_ocv2 + charger_ocv2)에 대해 ②③⑧을 한 번에 실행."""
    return {
        "directional_profile": charger_directional_profile(df, value_col, charger_col),
        "eof_mode1": charger_eof_mode1(df, value_col, charger_col),
        "special_cell_contrast": charger_special_cell_contrast(df, value_col, charger_col),
    }


# ---------------------------------------------------------------------------
# 연(bay)·단(tier) 교란 점검 — 호기 효과가 진짜 팬(호기) 고유 효과인지, 그 호기
# 트레이들이 우연히 특정 랙 위치(연·단)에 몰려서 생긴 착시인지 가른다.
# io_load 가 만드는 bay_<stage_key>/tier_<stage_key> 컬럼과 함께 쓴다.
# ---------------------------------------------------------------------------


def charger_bay_tier_coverage(df: pd.DataFrame, charger_col: str, bay_col: str,
                              tier_col: str) -> pd.DataFrame:
    """[연·단 교란점검 1] 호기별로 연(bay)·단(tier)이 얼마나 다양하게 퍼져 있는지.

    한 호기가 특정 연·단 몇 개에만 몰려 있으면 "호기 효과"와 "그 연·단 위치 효과"를
    데이터로 분리할 수 없다(완전교락). 넓게 퍼져 있어야 아래 층화분석이 의미 있다.
    """
    d = df.copy()
    d["_uid"] = _tray_uid_col(d)
    d = d.dropna(subset=[charger_col])

    rows = []
    for charger, sub in d.groupby(charger_col):
        uniq = sub[["_uid", bay_col, tier_col]].drop_duplicates(subset="_uid")
        bay_ok = uniq[bay_col].notna()
        tier_ok = uniq[tier_col].notna()
        rows.append({
            "charger": int(charger), "n_trays": int(uniq["_uid"].nunique()),
            "n_distinct_bay": int(uniq.loc[bay_ok, bay_col].nunique()),
            "n_distinct_tier": int(uniq.loc[tier_ok, tier_col].nunique()),
            "bay_min": (int(uniq.loc[bay_ok, bay_col].min()) if bay_ok.any() else None),
            "bay_max": (int(uniq.loc[bay_ok, bay_col].max()) if bay_ok.any() else None),
            "tier_min": (int(uniq.loc[tier_ok, tier_col].min()) if tier_ok.any() else None),
            "tier_max": (int(uniq.loc[tier_ok, tier_col].max()) if tier_ok.any() else None),
        })
    out = pd.DataFrame(rows).sort_values("charger", ignore_index=True)
    return out.assign(note=(
        "n_distinct_bay/tier 가 작으면(예 <5) 그 호기는 랙의 좁은 구역에만 몰려 있다는 "
        "뜻 — 호기 효과와 연·단 효과가 교락돼 아래 층화분석 신뢰도가 떨어진다"))


def charger_directional_profile_by_tier(df: pd.DataFrame, value_col: str, charger_col: str,
                                        tier_col: str) -> pd.DataFrame:
    """[연·단 교란점검 2 — 핵심] 단(tier, 높이)별로 층화해 호기별 세로 기울기가
    일관되게 재현되는지 확인.

    모든 단에서 4호=양수·5호=음수(반전)가 반복되면 "호기(팬) 고유 효과"로 확정할 수
    있다(특정 단 하나의 우연이 아님). 단마다 부호가 뒤집히거나 사라지면, 실은 특정
    단(높이) 효과가 호기와 교락된 것일 수 있다 — `charger_bay_tier_coverage` 로 먼저
    교락 가능성 자체를 확인할 것.
    """
    d = df.copy()
    d["_uid"] = _tray_uid_col(d)
    d["_dev"] = tray_delta(d, value_col, by="_uid", stat="median")
    d = d.dropna(subset=["_dev", "row", charger_col, tier_col])

    rows = []
    for (tier, charger), sub in d.groupby([tier_col, charger_col]):
        row_mean = sub.groupby("row")["_dev"].mean()
        row_slope, row_corr = _linreg(row_mean.index.to_numpy(), row_mean.to_numpy())
        rows.append({
            "tier": int(tier), "charger": int(charger),
            "n_trays": int(sub["_uid"].nunique()),
            "row_slope_per_row": row_slope, "row_corr": row_corr,
        })
    return pd.DataFrame(rows).sort_values(["tier", "charger"], ignore_index=True)


def bay_tier_gradient(df: pd.DataFrame, value_col: str, bay_col: str,
                      tier_col: str) -> dict:
    """[연·단 자체 효과] 호기와 무관하게, 연(가로 랙위치)·단(높이) 자체가 트레이
    전체 수준에서 체계적 영향을 주는지.

    ★ 트레이 내부 위치편차(tray_delta)가 아니라 **트레이별 원본값 평균**을 연·단에
    회귀한다 — 단(tier) 효과는 "트레이 전체가 통째로 더 덥거나 차가운가"라서, 트레이
    내부 편차만 보면(트레이 median을 빼버리면) 이 효과 자체가 상쇄돼 사라진다.
    """
    d = df.copy()
    d["_uid"] = _tray_uid_col(d)
    tray_level = d.groupby("_uid").agg(
        value=(value_col, "mean"), bay=(bay_col, "first"), tier=(tier_col, "first"))
    tray_level = tray_level.dropna()
    if len(tray_level) < 10:
        return {"ok": False, "reason": "표본 부족"}

    bay_slope, bay_corr = _linreg(tray_level["bay"].to_numpy(), tray_level["value"].to_numpy())
    tier_slope, tier_corr = _linreg(tray_level["tier"].to_numpy(), tray_level["value"].to_numpy())
    return {
        "ok": True, "n_trays": int(len(tray_level)),
        "n_bay": int(tray_level["bay"].nunique()), "n_tier": int(tray_level["tier"].nunique()),
        "bay_slope": bay_slope, "bay_corr": bay_corr,
        "tier_slope": tier_slope, "tier_corr": tier_corr,
        "note": ("tier_slope 가 뚜렷하면 높이에 따른 열성층(위/아래 온도차)이 실재한다는 "
                "뜻. bay_slope 가 뚜렷하면 연(가로 랙위치)에 따른 체계 효과(외기 입구 "
                "근접도 등)가 있다는 뜻. 둘 다 트레이 전체 평균 기준(위치편차 아님)"),
    }


def charger_bay_tier_report(df: pd.DataFrame, value_col: str, charger_col: str,
                            bay_col: str, tier_col: str) -> dict:
    """[편의 함수] 연·단 교란점검 3종을 한 번에 실행."""
    return {
        "coverage": charger_bay_tier_coverage(df, charger_col, bay_col, tier_col),
        "row_slope_by_tier": charger_directional_profile_by_tier(
            df, value_col, charger_col, tier_col),
        "bay_tier_gradient": bay_tier_gradient(df, value_col, bay_col, tier_col),
    }
