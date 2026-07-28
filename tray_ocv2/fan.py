"""충방전기 냉각팬 서명 검증 — docs/01_analysis_plan.md F 그룹.

온도로 먼저 검증한다(F0~F3). 전압은 SOC·자기방전 등이 섞이지만 온도는 순수한 열장이다.
예측값은 schema.py 에 데이터 라벨(세로 1~12, 가로 A~L)로 못박아 두었다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema
from .fields import position_field, tray_delta


def row_profile(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """[F1] 세로(1~12) 프로파일. 트레이 내 편차 기준, 평균±표준오차."""
    d = df.copy()
    d["_dev"] = tray_delta(d, value_col, stat="median")
    g = d.groupby("row")["_dev"]
    out = pd.DataFrame({"row": g.mean().index, "mean": g.mean().to_numpy(),
                        "se": (g.std() / np.sqrt(g.count())).to_numpy(),
                        "n": g.count().to_numpy()})
    out["is_fan_axis_max"] = out["row"].isin(schema.FAN_ROW_MAX)
    out["is_fan_axis_min"] = out["row"].isin(schema.FAN_ROW_MIN)
    return out.sort_values("row", ignore_index=True)


def band_column_profile(df: pd.DataFrame, value_col: str) -> dict:
    """[F2] ★★ 판결문 — 밴드별(상/중/하) 가로(A~L) 프로파일.

    상단≈하단 ≠ 중앙 이면 팬 확정(비분리형). 셋이 같으면 팬 기각(분리형=배선·기구물).
    """
    d = df.copy()
    d["_dev"] = tray_delta(d, value_col, stat="median")
    profiles = {}
    for band in schema.FAN_BANDS:
        r0, r1 = band["rows"]
        sub = d[(d["row"] >= r0) & (d["row"] <= r1)]
        g = sub.groupby("col")["_dev"]
        prof = pd.DataFrame({"col": g.mean().index, "mean": g.mean().to_numpy(),
                            "n": g.count().to_numpy()}).sort_values("col")
        prof["col_label"] = prof["col"].map(
            lambda c: chr(ord("A") + int(c) - 1) if pd.notna(c) else None)
        profiles[band["name"]] = prof
    return profiles


def band_similarity(profiles: dict) -> dict:
    """[F2 판정] 상단 vs 중앙, 하단 vs 중앙, 상단 vs 하단 프로파일 상관.

    상단↔하단은 높고 (상·하)↔중앙은 낮아야 '밴드마다 주기가 다르다'가 성립한다.
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
        "note": (f"'{top} vs {bot}' 상관이 높고 '~vs~{mid}' 상관이 낮으면 → "
                "밴드마다 가로 주기가 다름(비분리형) → 팬 확정에 부합"),
    }


def stage_field_reproducibility(df: pd.DataFrame, temp_cols: list[str]) -> pd.DataFrame:
    """[F0] 여러 단계의 온도 필드(트레이 편차)가 서로 재현되는가.

    팬은 상시 가동·고정 배열이므로 모든 스텝에서 같은 무늬여야 한다.
    144차원 위치 필드끼리의 상관행렬로 확인.
    """
    from .fields import pairwise_corr

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


def amplitude_vs_heat(df: pd.DataFrame, no_heat_cols: list[str],
                      heat_cols: list[str]) -> dict:
    """[F3] 무발열 스텝 vs 발열 스텝의 온도 필드 진폭(RMS) 비교.

    발열이 있어야 냉각 불균일이 드러난다 → 발열 스텝 진폭이 커야 팬 가설에 부합.
    """
    def rms_amp(col):
        if col not in df.columns:
            return np.nan
        d = df.copy()
        d["_dev"] = tray_delta(d, col, stat="median")
        f = position_field(d, "_dev", agg="rms").to_numpy()
        return float(np.nanmean(f))

    no_heat = {c: rms_amp(c) for c in no_heat_cols}
    heat = {c: rms_amp(c) for c in heat_cols}
    no_heat_mean = np.nanmean(list(no_heat.values())) if no_heat else np.nan
    heat_mean = np.nanmean(list(heat.values())) if heat else np.nan
    return {
        "no_heat_stage_amplitudes": no_heat, "heat_stage_amplitudes": heat,
        "no_heat_mean_rms": no_heat_mean, "heat_mean_rms": heat_mean,
        "ratio_heat_over_no_heat": (heat_mean / no_heat_mean
                                    if no_heat_mean and no_heat_mean > 0 else np.nan),
        "note": "비율이 1보다 뚜렷이 크면 F3(진폭∝발열량) 부합",
    }
