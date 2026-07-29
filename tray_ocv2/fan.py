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

    y = table["mean"].to_numpy()
    if len(y) < 4:
        return {"ok": False, "reason": "구간 수 부족", "table": table}

    diffs = np.diff(y)
    is_monotonic = bool(np.all(diffs >= 0) or np.all(diffs <= 0))
    # 중심(첫 구간) 대비 가장 크게 벗어나는 구간
    extremum_idx = int(np.argmax(np.abs(y - y[0])))

    return {
        "ok": True, "table": table,
        "center_mean": float(y[0]), "extremum_bin_idx": extremum_idx,
        "extremum_r_mid": float(table["r_mid"].iloc[extremum_idx]),
        "extremum_mean": float(y[extremum_idx]),
        "is_monotonic": is_monotonic,
        "note": ("비단조(is_monotonic=False)이고 극값이 중간 반경(첫·마지막 구간이 아님)"
                "이면 충돌제트 고리 구조에 부합 → 해석(b) 지지. 단조면 (b) 미지지 — "
                "결과 보고 부호만 뒤집는 사후 구제를 막기 위한 독립 검정"),
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
