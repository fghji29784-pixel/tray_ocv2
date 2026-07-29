"""발표 산출물 그림 — docs/06_visualization_plan.md 설계를 코드로 옮긴다.

현재는 S1(전체 지문 갤러리, 4장)만 구현되어 있다. V0~V16, S3, S4 는
docs/06_visualization_plan.md §6.3 구현순서를 따라 순차적으로 추가한다.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter

from . import fan, fields, impact, indicators, patterns, schema, spatial, style, timing

# ---------------------------------------------------------------------------
# S1 — 전체 지문 갤러리 (OCV·온도 × 원본·정규화, 4장)
# ---------------------------------------------------------------------------


def _ocv_specs() -> list[tuple[str, str, str]]:
    """(key, 표시라벨, 값컬럼) — 전압 계열(OCV#01~07 + 전용PRVT#01~03)."""
    return [(s.key, s.label, f"v_{s.key}") for s in schema.STAGES]


def _temp_specs() -> list[tuple[str, str, str]]:
    """(key, 표시라벨, 값컬럼) — 온도 계열 전체(LCI·충전1~7·방전1~7·OCV/PRVT온도).

    충전/방전은 온도 스텝 하나당 min/mean/max 세 컬럼이 있지만, 위치 '지문' 갤러리는
    스텝당 한 패널이 목표이므로 대표값으로 mean 을 쓴다(진폭 자체는 F3/F4 에서 별도로 본다).
    """
    specs = []
    for t in schema.THERM_STEPS:
        stat = "mean" if t.phase in ("charge", "discharge") else "single"
        specs.append((t.key, t.label, f"temp_{t.key}_{stat}"))
    return specs


#: family → (스펙 목록 함수, 갤러리 열 수, 단위 표시용 라벨, 한글 계열명)
_FAMILY_META = {
    "ocv": {
        "specs_fn": _ocv_specs, "ncols": 5,
        "unit_label": "전압(원 단위 — V/mV 스케일 미확정, A9 참고)",
        "name_kr": "전압(OCV)",
    },
    "temp": {
        "specs_fn": _temp_specs, "ncols": 6,
        "unit_label": "°C", "name_kr": "온도",
    },
}


def _field_for_spec(df: pd.DataFrame, col: str, normalized: bool) -> pd.DataFrame:
    """단일 스펙(컬럼)에 대한 (12,12) 위치별 평균 필드. normalized=True 면 트레이 mode 차감."""
    values = fields.tray_delta(df, col, stat="mode") if normalized else df[col]
    tmp = df[["row", "col"]].copy()
    tmp["_v"] = values
    return fields.position_field(tmp, "_v", agg="mean")


def fig_s1_gallery(df: pd.DataFrame, family: str, normalized: bool, outdir: str) -> str:
    """S1 4장 중 1장 생성. family: 'ocv' | 'temp'.

    색 스케일은 서브패널마다 독립(diverging=normalized, 공유 스케일 안 씀) — 단계별
    절대 규모가 물리적으로 달라서(예: OCV#02 진폭이 docv7보다 10~100배 큼) 공유 스케일은
    무늬를 지워버린다(docs/06_visualization_plan.md §1 S1 참고).
    """
    if family not in _FAMILY_META:
        raise ValueError(f"family 는 'ocv' 또는 'temp' 여야 합니다: {family!r}")
    meta = _FAMILY_META[family]
    specs = [(k, l, c) for k, l, c in meta["specs_fn"]() if c in df.columns]
    if not specs:
        raise ValueError(f"'{family}' 계열에 해당하는 컬럼이 df 에 하나도 없습니다.")

    ncols = meta["ncols"]
    nrows = int(np.ceil(len(specs) / ncols))
    name_kr = meta["name_kr"]

    if normalized:
        title = f"트레이 mode를 빼고 보면, 여러 단계에서 같은 자리 무늬가 반복됩니다 ({name_kr})"
        how = ("패널마다 (셀 값 - 그 트레이의 mode)의 위치별 평균을 그렸습니다. 같은 자리가 "
               "여러 패널에서 반복해서 튀면 우연이 아닙니다. 색 스케일은 패널마다 독립입니다.")
    else:
        title = f"{name_kr} 절대값은 단계마다 다릅니다 — 정규화 전 원본입니다"
        how = ("패널마다 원본 값의 위치별 평균을 그대로 그렸습니다. 색 스케일은 패널마다 "
               "독립입니다(단계별 절대 규모가 물리적으로 달라서 공유 스케일은 무늬를 지웁니다).")

    fig, axes = style.kind_fig(
        title, how, figsize=(ncols * 2.7, nrows * 2.55 + 1.15),
        nrows=nrows, ncols=ncols, top=0.80,
        gridspec_kw={"wspace": 0.75, "hspace": 0.55})
    axes_flat = np.atleast_1d(axes).ravel()

    # 패널마다 색 스케일·컬러바가 독립이라 라벨은 짧게(단위만) — 긴 설명문은 부제에서
    # 이미 하므로 패널 30개에 반복하면 옆 패널을 침범해 오히려 못 읽는다(실측 확인).
    unit_short = "°C" if family == "temp" else "원단위"
    cbar_label = f"mode차 [{unit_short}]" if normalized else f"원본 [{unit_short}]"

    for ax, (_key, label, col) in zip(axes_flat, specs):
        grid_df = _field_for_spec(df, col, normalized)
        im = style.tray_heatmap(
            ax, grid_df.to_numpy(), unit=meta["unit_label"], diverging=normalized,
            show_margin_axis=False, cbar=True, cbar_label=cbar_label)
        cb = im.colorbar
        cb.ax.tick_params(labelsize=6.5)
        # 원본(비정규화) OCV 는 절대값이 ~3.5 근처라 matplotlib 이 "+3.499" 식 오프셋
        # 표기를 컬러바 위에 붙이는데, 그게 옆 패널 제목과 겹친다(실측 확인) — 꺼둔다.
        cb.ax.ticklabel_format(useOffset=False, style="plain")
        cb.set_label(cbar_label, fontsize=7.5, labelpad=4)
        ax.set_title(label, fontsize=9.5, pad=4)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=7.5)

    for ax in axes_flat[len(specs):]:
        ax.axis("off")

    style.observation_tag(fig)
    tag = "정규화" if normalized else "원본"
    path = os.path.join(outdir, f"S1_{family}_{tag}_갤러리.png")
    return style.save(fig, path)


def fig_s1_ocv_gallery(df: pd.DataFrame, normalized: bool, outdir: str) -> str:
    return fig_s1_gallery(df, "ocv", normalized, outdir)


def fig_s1_temp_gallery(df: pd.DataFrame, normalized: bool, outdir: str) -> str:
    return fig_s1_gallery(df, "temp", normalized, outdir)


def fig_s1_all(df: pd.DataFrame, outdir: str) -> dict:
    """S1 4장 전부 생성 — (OCV·온도) × (원본·정규화)."""
    os.makedirs(outdir, exist_ok=True)
    return {
        "ocv_raw": fig_s1_ocv_gallery(df, False, outdir),
        "ocv_norm": fig_s1_ocv_gallery(df, True, outdir),
        "temp_raw": fig_s1_temp_gallery(df, False, outdir),
        "temp_norm": fig_s1_temp_gallery(df, True, outdir),
    }


# ---------------------------------------------------------------------------
# V1 — E8 분산분해 (값이 달라지는 이유를 쪼갠다) 🔴
# ---------------------------------------------------------------------------


def fig_v1a_variance_bar(vardecomp: dict, outdir: str) -> str:
    """V1a — 100% 누적 막대: 위치고정 / 트레이별EOF / 셀잔차."""
    if not vardecomp.get("ok"):
        raise ValueError(f"variance_decomposition 결과가 유효하지 않습니다: {vardecomp}")

    fixed = vardecomp["frac_position_fixed"] * 100
    eof = vardecomp["frac_tray_position_eof"] * 100
    resid = vardecomp["frac_cell_residual"] * 100
    combined = fixed + eof

    fig, ax = style.kind_fig(
        f"전압 강하량이 셀마다 다른 이유의 {combined:.1f}%는 '어느 자리에 앉았나' 입니다",
        "가로 막대 전체가 100%. 왼쪽 두 조각이 자리 때문에 생기는 차이입니다.",
        figsize=(9.8, 4.2), top=0.72)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.35)
    ax.axis("off")

    # 라벨은 3줄(퍼센트+2줄 설명) 기준으로 만들었다 — 조각 폭이 좁으면(예 소수 랏에서
    # 고정성분이 몇 % 뿐인 경우) 안에 다 안 들어가고 옆 조각을 침범한다(실측 확인).
    # 폭이 좁은 조각은 대신 위쪽에 콜아웃(짧은 인출선)으로 뺀다.
    NARROW_PCT = 14.0
    segs = [
        (fixed, style.C["accent"], "모든 트레이\n공통 자리효과"),
        (eof, style.C["low_soft"], "트레이마다\n다른 자리효과"),
        (resid, style.C["neutral"], "셀 자체의 차이\n(진짜 불량 신호)"),
    ]
    x0 = 0.0
    for width, color, desc in segs:
        ax.add_patch(plt.Rectangle((x0, 0.32), width, 0.4, fc=color, ec="white",
                                   lw=1.6, zorder=2))
        text_color = "white" if color != style.C["neutral"] else "#22262B"
        cx = x0 + width / 2
        if width >= NARROW_PCT:
            ax.text(cx, 0.52, f"{width:.1f}%\n{desc}", ha="center", va="center",
                   fontsize=9, color=text_color, zorder=3)
        else:
            ax.annotate(f"{width:.1f}%\n{desc}", xy=(cx, 0.72), xytext=(cx, 1.18),
                       ha="center", va="bottom", fontsize=8.3, color=color,
                       arrowprops=dict(arrowstyle="-", color=color, lw=1.0), zorder=4)
        x0 += width

    ax.annotate("", xy=(combined, 0.80), xytext=(0, 0.80),
               arrowprops=dict(arrowstyle="-", color=style.C["accent"], lw=1.6))
    ax.text(combined / 2, 0.86, f"{combined:.1f}% 자리 때문", ha="center",
           fontsize=11, color=style.C["accent"], fontweight="bold")
    style.note(ax, "이 몫을 현재 판정은 하나도 보정하지 않습니다",
              xy=(combined, 0.32), xytext=(min(combined + 10, 70), 0.06),
              color=style.C["high"])

    style.verdict_badge(fig, "high")
    style.chain_strip(fig, "value")
    path = os.path.join(outdir, "V1a_분산분해_막대.png")
    return style.save(fig, path)


def fig_v1b_variance_maps(df: pd.DataFrame, value_col: str, outdir: str) -> str:
    """V1b — 12x12 히트맵 3장: ①위치고정 성분 ②대표 트레이 EOF 잔차 ③셀 잔차."""
    d = df.copy()
    d["_dev"] = fields.tray_delta(d, value_col, stat="median")
    valid = d["_dev"].notna() & d["row"].notna() & d["col"].notna()
    sub = d[valid].copy()
    if len(sub) < 200:
        raise ValueError("표본 부족 — V1b 계산 불가")

    pos_field = fields.position_field(sub, "_dev", agg="mean")
    sub["_resid_after_pos"] = sub["_dev"] - sub.groupby(["row", "col"])["_dev"].transform("mean")

    tray_ids, mat_resid = spatial.build_tray_matrix(
        sub, value_col="_resid_after_pos", already_relative=True)
    pca = spatial.pca_modes(mat_resid, n_modes=3)
    if not pca.get("ok"):
        raise ValueError(f"PCA 실패 — V1b 계산 불가: {pca}")

    valid_pos = pca["tray_index_valid"]
    scores = pca["scores"]
    mag = np.sum(scores ** 2, axis=1)
    i_valid = int(np.argmax(mag))
    orig_idx = int(valid_pos[i_valid])

    grid_after_pos = mat_resid[orig_idx].reshape(fields.N_ROWS, fields.N_COLS)
    recon = np.tensordot(scores[i_valid], pca["loadings"], axes=(0, 0))
    grid_cell_resid = np.where(np.isnan(grid_after_pos), np.nan, grid_after_pos - recon)

    vmax = float(np.nanmax(np.abs(np.stack([
        pos_field.to_numpy(), grid_after_pos, grid_cell_resid]))))

    fig, axes = style.kind_fig(
        "자리 효과가 실제로 어떻게 생겼는지 눈으로 봅니다",
        "왼쪽부터 ① 모든 트레이 공통 무늬 ② 어떤 트레이 하나의 개인 무늬 ③ 나머지(셀 자체). "
        "색 스케일은 세 장 모두 같습니다 — ③이 ①②보다 흐리고 무늬가 없어야 정상입니다.",
        figsize=(12.0, 4.6), nrows=1, ncols=3, top=0.74,
        gridspec_kw={"wspace": 0.55})

    titles = [
        "① 위치 고정 성분\n(모든 트레이 공통)",
        f"② 대표 트레이 1개 잔차\n({tray_ids[orig_idx]}, EOF 성분 최대)",
        "③ 셀 잔차\n(구조 없어야 정상)",
    ]
    grids = [pos_field.to_numpy(), grid_after_pos, grid_cell_resid]
    for ax, grid, ttl in zip(axes, grids, titles):
        style.tray_heatmap(ax, grid, unit="mV", vmax=vmax, cbar=True,
                          show_margin_axis=False, cbar_label="mV")
        ax.set_title(ttl, fontsize=10)

    style.verdict_badge(fig, "high")
    style.chain_strip(fig, "value")
    path = os.path.join(outdir, "V1b_분산분해_지도.png")
    return style.save(fig, path)


def fig_v1_all(df: pd.DataFrame, outdir: str, value_col: str = "docv7_raw") -> dict:
    os.makedirs(outdir, exist_ok=True)
    vardecomp = spatial.variance_decomposition(df, value_col)
    return {
        "v1a": fig_v1a_variance_bar(vardecomp, outdir),
        "v1b": fig_v1b_variance_maps(df, value_col, outdir),
    }


# ---------------------------------------------------------------------------
# V3 — I3 spike-in (판정이 실제로 흔들린다, 발표 클라이맥스) 🔴
# ---------------------------------------------------------------------------


def _flow_box(ax, xy, w, h, lines, fc="#EDF2F7", ec=None):
    ec = ec or style.C["accent"]
    ax.add_patch(plt.Rectangle(xy, w, h, fc=fc, ec=ec, lw=1.6, zorder=2))
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    ax.text(cx, cy, "\n".join(lines), ha="center", va="center",
           fontsize=9.3, color="#1C232C", zorder=3)


def fig_v3a_spike_concept(outdir: str, n_trays: int | None = None) -> str:
    """V3a — 인공불량 주입 절차 개념도. n_trays 를 주면 실제 랏의 트레이 수를 표시한다."""
    fig, ax = style.kind_fig(
        "정답을 아는 가짜 불량을 심어서, 현재 로직이 잡아내는지 시험했습니다",
        "현재 등급표는 현재 로직이 만든 것이라 순환입니다. 그래서 정답을 직접 만들어 "
        "비순환적으로 검증합니다.",
        figsize=(11.5, 4.4), top=0.72)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.3)
    ax.axis("off")

    w, h, y = 2.35, 1.5, 1.5
    boxes = [
        (0.2, ["① 정상 셀 1개 선택", "(트레이 144칸 중 무작위 위치)"]),
        (2.85, ["② 그 셀에만", "+0.8mV 강제로 더함", "-> 이 셀은 100% 불량"]),
        (5.5, ["③ 현재 판정 로직", "그대로 실행", "(트레이 mode + 0.8mV)"]),
        (8.15, ["④ 잡았나? 놓쳤나?", "위치별로 집계"]),
    ]
    for x, lines in boxes:
        _flow_box(ax, (x, y), w, h, lines)
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + w
        x2 = boxes[i + 1][0]
        ax.annotate("", xy=(x2, y + h / 2), xytext=(x1, y + h / 2),
                   arrowprops=dict(arrowstyle="->", color=style.C["accent"], lw=2))

    tray_txt = f"트레이 {n_trays}개" if n_trays else "트레이 전체"
    ax.annotate("", xy=(5.5 + w / 2, 1.05), xytext=(5.5 + w / 2, y),
               arrowprops=dict(arrowstyle="->", color=style.C["high"], lw=1.8))
    ax.text(5.5 + w / 2, 0.55,
           f"{tray_txt} 전부에서 반복 -> 위치(테두리 거리)별 검출률 비교\n"
           "(트레이당 1개만 심습니다 - 여러 개면 mode 자체가 흔들려 기준선이 무너집니다)",
           ha="center", va="center", fontsize=9.5, color=style.C["high"], fontweight="bold")

    style.verdict_badge(fig, "high")
    style.chain_strip(fig, "verdict")
    path = os.path.join(outdir, "V3a_인공불량_개념도.png")
    return style.save(fig, path)


def _pick_spike_entry(spike: dict, target_mv: float) -> dict:
    return min(spike["results"], key=lambda r: abs(r["magnitude_mv"] - target_mv))


def fig_v3b_detection_by_position(spike: dict, outdir: str,
                                  target_mv: float = schema.JUDGE_OFFSET_MV) -> str:
    """V3b — 판정 기준 크기(기본 0.8mV) 인공불량의 위치(테두리거리)별 검출률."""
    if not spike.get("ok"):
        raise ValueError(f"spike_in_test 결과가 유효하지 않습니다: {spike}")
    entry = _pick_spike_entry(spike, target_mv)
    by_edge = entry["detection_by_edge_distance"]
    edges = sorted(by_edge)
    rates = np.array([by_edge[e]["rate"] * 100 for e in edges])
    ns = [by_edge[e]["n"] for e in edges]
    cis = [style.binomial_ci(round(by_edge[e]["rate"] * by_edge[e]["n"]), by_edge[e]["n"])
          for e in edges]
    err_low = [max(0.0, rates[i] - cis[i][0] * 100) for i in range(len(edges))]
    err_high = [max(0.0, cis[i][1] * 100 - rates[i]) for i in range(len(edges))]

    overall = entry["overall_detection_rate"] * 100
    rmax, rmin = float(rates.max()), float(rates.min())
    ratio_txt = (f"최대 {rmax / rmin:.1f}배" if rmin > 0 else "뚜렷한")

    fig, ax = style.kind_fig(
        f"똑같은 크기({entry['magnitude_mv']}mV)의 인공불량인데 자리에 따라 "
        f"검출률이 {ratio_txt} 차이납니다",
        "세로축은 '심어둔 불량을 실제로 잡아낸 비율'입니다. 평평해야 정상입니다. "
        "세로선은 이항 신뢰구간(표본이 작을수록 넓습니다).",
        figsize=(9.5, 6.6), top=0.80)

    x = np.arange(len(edges))
    ax.bar(x, rates, color=style.C["accent"], width=0.62, zorder=3,
          yerr=[err_low, err_high], capsize=4, ecolor="#333333",
          error_kw=dict(lw=1.3, zorder=4))
    ax.axhline(overall, color=style.C["rule"], ls="--", lw=1.5, zorder=2,
              label=f"전체 평균 {overall:.1f}%")
    for xi, r, n, eh in zip(x, rates, ns, err_high):
        style.annotate_count(ax, xi, r + eh, n)

    ax.set_xticks(x, [str(e) for e in edges])
    ax.set_xlabel("테두리로부터의 거리 (0=가장자리 ... 5=한가운데)")
    ax.set_ylabel("검출률 (%)")
    ax.set_ylim(0, max(105, float(rates.max() + max(err_high)) + 15))
    ax.legend(loc="lower right")

    imax, imin = int(np.argmax(rates)), int(np.argmin(rates))
    if imax != imin:
        style.note(ax, f"거리{edges[imax]}: {rates[imax]:.1f}%", (x[imax], rates[imax]),
                  xytext=(x[imax] - 1.4, min(rates[imax] + 22, 95)))
        style.note(ax, f"거리{edges[imin]}: {rates[imin]:.1f}%", (x[imin], rates[imin]),
                  xytext=(x[imin] + 0.8, min(rates[imin] + 28, 98)), color=style.C["high"])

    style.verdict_badge(fig, "high")
    style.chain_strip(fig, "verdict")
    path = os.path.join(outdir, "V3b_위치별_검출률.png")
    return style.save(fig, path)


def fig_v3c_magnitude_sensitivity(spike: dict, outdir: str,
                                  target_mv: float = schema.JUDGE_OFFSET_MV) -> str:
    """V3c — 주입 크기별(작음/기준/큼) 위치-검출률 곡선. 경계 크기에서만 자리 차이가 드러난다."""
    if not spike.get("ok"):
        raise ValueError(f"spike_in_test 결과가 유효하지 않습니다: {spike}")

    fig, ax = style.kind_fig(
        "판정 기준 근처에서만 자리 차이가 뚜렷하게 드러납니다",
        "선마다 서로 다른 크기의 인공 불량입니다. 작은 불량은 어디서든 못 잡고, 큰 불량은 "
        "어디서든 잡습니다 — 문제는 그 사이 경계 크기입니다.",
        figsize=(9.5, 6.2), top=0.80)

    palette = [style.C["low"], style.C["accent"], style.C["high"]]
    results_sorted = sorted(spike["results"], key=lambda r: r["magnitude_mv"])
    closest = _pick_spike_entry(spike, target_mv)
    for i, entry in enumerate(results_sorted):
        by_edge = entry["detection_by_edge_distance"]
        edges = sorted(by_edge)
        rates = [by_edge[e]["rate"] * 100 for e in edges]
        is_target = entry is closest
        ax.plot(edges, rates, marker="o", lw=2.4 if is_target else 1.6,
               color=palette[i % len(palette)],
               label=f"{entry['magnitude_mv']}mV 주입" + (" (판정 기준)" if is_target else ""))

    ax.set_xlabel("테두리로부터의 거리 (0=가장자리 ... 5=한가운데)")
    ax.set_ylabel("검출률 (%)")
    ax.set_ylim(-5, 108)
    ax.legend(loc="center right")

    style.verdict_badge(fig, "high")
    style.chain_strip(fig, "verdict")
    path = os.path.join(outdir, "V3c_주입크기별_민감도.png")
    return style.save(fig, path)


def fig_v3_all(df: pd.DataFrame, outdir: str) -> dict:
    os.makedirs(outdir, exist_ok=True)
    spike = impact.spike_in_test(df)
    n_trays = int(df["tray_id"].nunique()) if "tray_id" in df.columns else None
    return {
        "v3a": fig_v3a_spike_concept(outdir, n_trays=n_trays),
        "v3b": fig_v3b_detection_by_position(spike, outdir),
        "v3c": fig_v3c_magnitude_sensitivity(spike, outdir),
    }


# ---------------------------------------------------------------------------
# V2 — L6 재현성 + null 대조 (랏 1개 한계 방어) 🔴
# ---------------------------------------------------------------------------


def fig_v2_split_half(df: pd.DataFrame, value_col: str, outdir: str, seed: int = 0) -> str:
    """V2 — 트레이를 무작위 반반으로 나눈 두 지도 + 실제 산점도 + null(위치 셔플) 대조."""
    d = df.copy()
    d["_dev"] = fields.tray_delta(d, value_col, stat="median")
    tray_ids = d["tray_id"].dropna().unique()
    if len(tray_ids) < 20:
        raise ValueError("트레이 수 부족(<20) - V2 계산 불가")

    rng = np.random.default_rng(seed)
    shuffled_trays = rng.permutation(tray_ids)
    half = len(shuffled_trays) // 2
    g1, g2 = set(shuffled_trays[:half]), set(shuffled_trays[half:])

    f1 = fields.position_field(d[d["tray_id"].isin(g1)], "_dev", agg="mean")
    f2 = fields.position_field(d[d["tray_id"].isin(g2)], "_dev", agg="mean")
    a1, a2 = f1.to_numpy().ravel(), f2.to_numpy().ravel()
    ok = np.isfinite(a1) & np.isfinite(a2)
    if ok.sum() < 20:
        raise ValueError("유효 위치 부족 - V2 계산 불가")
    idx = np.where(ok)[0]
    r_real = float(np.corrcoef(a1[idx], a2[idx])[0, 1])

    a2_shuffled = a2[idx].copy()
    rng.shuffle(a2_shuffled)
    r_null = float(np.corrcoef(a1[idx], a2_shuffled)[0, 1])

    vmax = float(np.nanmax(np.abs(np.stack([f1.to_numpy(), f2.to_numpy()]))))

    fig, axes = style.kind_fig(
        f"트레이를 무작위로 반씩 나눠도 같은 무늬가 재현됩니다 (일치도 r={r_real:.3f})",
        "왼쪽 둘은 무작위로 나눈 두 그룹 각각의 평균 위치 지문입니다. 오른쪽 두 산점도 중 "
        "'실제'가 직선을 이루고 '무작위 대조'가 구름 모양이어야 우연이 아니라는 뜻입니다.",
        figsize=(14.5, 4.4), nrows=1, ncols=4, top=0.76,
        gridspec_kw={"wspace": 0.55})

    style.tray_heatmap(axes[0], f1.to_numpy(), unit="mV", vmax=vmax,
                      show_margin_axis=False, cbar_label="mV")
    axes[0].set_title(f"그룹1 ({len(g1)}개 트레이)", fontsize=10)
    style.tray_heatmap(axes[1], f2.to_numpy(), unit="mV", vmax=vmax,
                      show_margin_axis=False, cbar_label="mV")
    axes[1].set_title(f"그룹2 ({len(g2)}개 트레이)", fontsize=10)

    lims = [float(min(a1[idx].min(), a2[idx].min())),
           float(max(a1[idx].max(), a2[idx].max()))]
    axes[2].scatter(a1[idx], a2[idx], s=18, color=style.C["accent"], alpha=0.75, zorder=3)
    axes[2].plot(lims, lims, ls="--", color=style.C["rule"], lw=1.2, zorder=2)
    axes[2].set_title(f"실제: r={r_real:.3f}", fontsize=10.5,
                      color=style.C["accent"], fontweight="bold")
    axes[2].set_xlabel("그룹1 값 (mV)")
    axes[2].set_ylabel("그룹2 값 (mV)")

    axes[3].scatter(a1[idx], a2_shuffled, s=18, color=style.C["neutral"], alpha=0.75, zorder=3)
    axes[3].plot(lims, lims, ls="--", color=style.C["rule"], lw=1.2, zorder=2)
    axes[3].set_title(f"무작위 대조: r={r_null:.3f}", fontsize=10.5,
                      color=style.C["neutral"], fontweight="bold")
    axes[3].set_xlabel("그룹1 값 (mV)")
    axes[3].set_ylabel("그룹2 값 (위치 섞음, mV)")

    style.verdict_badge(fig, "high")
    style.chain_strip(fig, "value")
    path = os.path.join(outdir, "V2_재현성_반반나누기.png")
    return style.save(fig, path)


# ---------------------------------------------------------------------------
# V14/V15 — L 그룹: 패턴 유형·비율 / 고정보정 한계 ("왜 적응형이 필요한가")
# ---------------------------------------------------------------------------


def fig_v14_pattern_ratio(df: pd.DataFrame, value_col: str, outdir: str) -> str:
    """V14 — 패턴 유형 비율 막대 + 유형별 대표 트레이 지도 3장."""
    pattern_df = patterns.classify_patterns(df, value_col)
    ratio = patterns.pattern_ratio(pattern_df)
    if not ratio.get("ok"):
        raise ValueError(f"패턴 분류 실패 - V14 계산 불가: {ratio}")

    d = df.copy()
    d["_dev"] = fields.tray_delta(d, value_col, stat="median")

    labels = ["링(ring)", "역링(anti-ring)", "약함/무구조"]
    reps: dict[str, tuple] = {}
    for lbl in labels:
        sub = pattern_df[pattern_df["pattern"] == lbl]
        if sub.empty:
            continue
        idx = sub["mode0_score"].abs().idxmax()
        tray_id = sub.loc[idx, "tray_id"]
        grid = fields.to_grid(d[d["tray_id"] == tray_id], "_dev")
        reps[lbl] = (tray_id, grid)

    style._ensure_init()
    fig = plt.figure(figsize=(11.8, 8.2))
    fig.suptitle("위치 무늬는 한 종류가 아닙니다 - 정반대 무늬가 섞여 있습니다",
                fontsize=15.5, fontweight="bold", color=style.C["accent"],
                x=0.02, ha="left", y=0.985)
    fig.text(0.02, 0.930,
            "위 막대는 트레이 전체 중 각 무늬 유형의 비율입니다. 아래 3장은 그 유형을 "
            "가장 뚜렷하게 보여주는 실제 트레이 예시입니다.",
            fontsize=11, color="#3C4149", ha="left", va="top", linespacing=1.5)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.9, 1.6], hspace=0.65, wspace=0.5,
                          top=0.80, bottom=0.20, left=0.08, right=0.96)

    ax_bar = fig.add_subplot(gs[0, :])
    pcts = [ratio["ratio"].get(lbl, 0.0) * 100 for lbl in labels]
    colors = [style.C["high"], style.C["low"], style.C["neutral"]]
    ax_bar.barh(labels, pcts, color=colors, height=0.55)
    for y, p in enumerate(pcts):
        ax_bar.text(p + 1.2, y, f"{p:.1f}%", va="center", fontsize=10, fontweight="bold")
    ax_bar.set_xlim(0, max(pcts) * 1.25 if pcts else 1)
    ax_bar.set_xlabel("트레이 중 비율 (%)")
    ax_bar.invert_yaxis()

    if reps:
        vmax = float(np.nanmax([np.nanmax(np.abs(g)) for _, g in reps.values()]))
    else:
        vmax = 1.0
    for i, lbl in enumerate(labels):
        ax = fig.add_subplot(gs[1, i])
        if lbl in reps:
            tray_id, grid = reps[lbl]
            style.tray_heatmap(ax, grid, unit="mV", vmax=vmax, show_margin_axis=False,
                              cbar_label="mV")
            ax.set_title(f"{lbl}\n({tray_id})", fontsize=9.5)
        else:
            ax.axis("off")
            ax.set_title(f"{lbl}\n(예시 없음)", fontsize=9.5)

    style.verdict_badge(fig, "high")
    style.chain_strip(fig, "value")
    path = os.path.join(outdir, "V14_패턴유형비율.png")
    return style.save(fig, path)


def fig_v15_fixed_correction_limit(vardecomp: dict, outdir: str) -> str:
    """V15 — 고정 보정(트레이 공통 지도) vs 적응형 보정이 잡는 분산 비율 비교."""
    flimit = patterns.fixed_correction_limit(vardecomp)
    if not flimit.get("ok"):
        raise ValueError(f"고정보정한계 계산 실패 - V15 계산 불가: {flimit}")

    fixed_pct = flimit["frac_position_fixed_pct"]
    adaptive_pct = round((vardecomp["frac_position_fixed"]
                         + vardecomp["frac_tray_position_eof"]) * 100, 1)

    fig, ax = style.kind_fig(
        f"모든 트레이에 같은 보정을 하면 {fixed_pct:.1f}%밖에 못 잡습니다",
        "왼쪽 막대는 '모든 트레이에 똑같은 보정 지도 1장'을 썼을 때 잡히는 몫, 오른쪽은 "
        "트레이마다 다른 무늬까지 반영(적응형)했을 때 잡히는 몫입니다.",
        figsize=(7.6, 6.2), top=0.78)

    xs = ["고정 보정\n(트레이 공통 지도 1장)", "적응형 보정\n(트레이별 무늬까지 반영)"]
    ys = [fixed_pct, adaptive_pct]
    ax.bar(xs, ys, color=[style.C["neutral"], style.C["accent"]], width=0.55, zorder=3)
    for x, y in zip(xs, ys):
        ax.text(x, y + max(ys) * 0.03, f"{y:.1f}%", ha="center", fontsize=12.5,
               fontweight="bold")
    ax.set_ylabel("잡히는 분산 비율 (%)")
    ax.set_ylim(0, max(ys) * 1.3)

    style.verdict_badge(fig, "high")
    style.chain_strip(fig, "value")
    path = os.path.join(outdir, "V15_고정보정한계.png")
    return style.save(fig, path)


def fig_v14_v15_all(df: pd.DataFrame, outdir: str, value_col: str = "docv7_raw") -> dict:
    os.makedirs(outdir, exist_ok=True)
    vardecomp = spatial.variance_decomposition(df, value_col)
    return {
        "v14": fig_v14_pattern_ratio(df, value_col, outdir),
        "v15": fig_v15_fixed_correction_limit(vardecomp, outdir),
    }


# ---------------------------------------------------------------------------
# S3 — 전압×온도 상관 (현 상황, 관찰) — R² 로 온도 설명력의 상한을 보인다
# ---------------------------------------------------------------------------


def fig_s3_ocv_temp_corr(df: pd.DataFrame, outdir: str, docv7_col: str = "docv7_raw") -> str:
    """S3 — OCV#01~03 온도 및 온도차(temp7) vs docv7 산점도 + r/R²."""
    if docv7_col not in df.columns:
        raise ValueError(f"{docv7_col} 컬럼 없음 - S3 계산 불가")

    candidates = [("t_ocv1", "OCV#01 온도"), ("t_ocv2", "OCV#02 온도"),
                 ("t_ocv3", "OCV#03 온도")]
    panels = [(c, l) for c, l in candidates if c in df.columns]
    has_temp7 = "t_ocv1" in df.columns and "t_ocv3" in df.columns
    if has_temp7:
        panels.append(("_temp7", "온도차 (OCV1온도-OCV3온도)"))
    if not panels:
        raise ValueError("온도 컬럼 없음 - S3 계산 불가")

    d = df.copy()
    if has_temp7:
        d["_temp7"] = pd.to_numeric(d["t_ocv1"], errors="coerce") - pd.to_numeric(
            d["t_ocv3"], errors="coerce")

    n = len(panels)
    fig, axes = style.kind_fig(
        "온도는 docv7 변동의 일부만 설명합니다 - 나머지는 온도와 무관합니다",
        "점 하나가 셀 하나입니다. R²가 클수록 온도가 값 차이를 더 많이 설명한다는 뜻입니다.",
        figsize=(n * 4.0, 4.7), nrows=1, ncols=n, top=0.78,
        gridspec_kw={"wspace": 0.45})
    axes = np.atleast_1d(axes)

    y = pd.to_numeric(d[docv7_col], errors="coerce")
    for ax, (col, label) in zip(axes, panels):
        x = pd.to_numeric(d[col], errors="coerce")
        ok = x.notna() & y.notna()
        xv, yv = x[ok].to_numpy(), y[ok].to_numpy()
        r = (float(np.corrcoef(xv, yv)[0, 1])
            if len(xv) > 2 and np.std(xv) > 1e-9 else np.nan)
        ax.scatter(xv, yv, s=6, alpha=0.25, color=style.C["accent"], zorder=2)
        if np.isfinite(r):
            slope, intercept = (float(v) for v in np.polyfit(xv, yv, 1))
            xs_line = np.linspace(xv.min(), xv.max(), 50)
            ax.plot(xs_line, slope * xs_line + intercept, color=style.C["high"],
                   lw=2, zorder=3)
            ax.set_title(f"{label}\nr={r:.3f}, R²={r ** 2:.3f}", fontsize=9.5)
        else:
            ax.set_title(f"{label}\n(계산 불가)", fontsize=9.5)
        ax.set_xlabel(f"{label} (°C)")
        ax.set_ylabel("docv7 (mV)")

    style.observation_tag(fig)
    path = os.path.join(outdir, "S3_전압온도_상관_산점도.png")
    return style.save(fig, path)


# ---------------------------------------------------------------------------
# V4 — F3 온도라는 물리적 배경 🟠 (원인까지는 확정, 판정까지의 직접연결은 미확정)
# ---------------------------------------------------------------------------


def _default_rise_stat_cols() -> list[tuple]:
    cols = []
    for prefix, n_max in (("charge", 7), ("discharge", 7)):
        for n in range(1, n_max + 1):
            key = f"{prefix}{n}"
            cols.append((f"temp_{key}_mean", f"temp_{key}_min", f"temp_{key}_max", key))
    return cols


def fig_v4a_heat_vs_amplitude(reg: dict, outdir: str) -> str:
    """V4a — 발열량(연속량) vs 위치필드 진폭(RMS) 산점도+회귀."""
    if not reg.get("ok"):
        raise ValueError(f"heat_vs_amplitude_regression 실패 - V4a 계산 불가: {reg}")
    table = reg["table"]

    fig, ax = style.kind_fig(
        f"열이 많이 날수록 자리별 온도차가 커집니다 ({len(table)}개 공정 단계, r={reg['corr']:.2f})",
        "점 하나가 공정 단계 하나입니다. 오른쪽으로 갈수록 열이 많이 나는 단계입니다.",
        figsize=(8.6, 6.4), top=0.80)

    x = table["median_rise_c"].to_numpy()
    y = table["field_rms"].to_numpy()
    ax.scatter(x, y, s=45, color=style.C["accent"], zorder=3)
    for xi, yi, lbl in zip(x, y, table["stage"]):
        ax.annotate(str(lbl), (xi, yi), textcoords="offset points", xytext=(5, 4),
                   fontsize=8, color="#4A4F57")
    xs_line = np.linspace(float(x.min()), float(x.max()), 50)
    ax.plot(xs_line, reg["slope"] * xs_line + reg["intercept"], color=style.C["high"],
           lw=2, ls="--", zorder=2)
    ax.set_xlabel("발열량 (최고-최저 온도, °C)")
    ax.set_ylabel("자리별 온도 불균일 (RMS, °C)")

    style.verdict_badge(fig, "high_soft")
    style.chain_strip(fig, "cause")
    path = os.path.join(outdir, "V4a_발열량_대_불균일.png")
    return style.save(fig, path)


def fig_v4b_heat_map_compare(df: pd.DataFrame, reg: dict, rise_stat_cols: list[tuple],
                             outdir: str) -> str:
    """V4b — 저발열 단계 vs 고발열 단계 위치 지도 비교(동일 색스케일)."""
    if not reg.get("ok"):
        raise ValueError(f"heat_vs_amplitude_regression 실패 - V4b 계산 불가: {reg}")
    table = reg["table"]
    low_row = table.loc[table["median_rise_c"].idxmin()]
    high_row = table.loc[table["median_rise_c"].idxmax()]
    label_to_col = {label: mean_col for mean_col, _mn, _mx, label in rise_stat_cols}
    low_col, high_col = label_to_col[low_row["stage"]], label_to_col[high_row["stage"]]

    d = df.copy()
    d["_dev_low"] = fields.tray_delta(d, low_col, stat="median")
    d["_dev_high"] = fields.tray_delta(d, high_col, stat="median")
    f_low = fields.position_field(d, "_dev_low", agg="mean")
    f_high = fields.position_field(d, "_dev_high", agg="mean")
    vmax = float(np.nanmax(np.abs(np.stack([f_low.to_numpy(), f_high.to_numpy()]))))

    fig, axes = style.kind_fig(
        "열이 적을 때는 고른데, 열이 많을 때 자리별로 갈립니다",
        "같은 트레이, 같은 자리 배치. 다른 건 그 공정에서 열이 얼마나 났는지뿐입니다.",
        figsize=(9.8, 4.7), nrows=1, ncols=2, top=0.78, gridspec_kw={"wspace": 0.5})

    style.tray_heatmap(axes[0], f_low.to_numpy(), unit="°C", vmax=vmax,
                      show_margin_axis=False, cbar_label="°C")
    axes[0].set_title(f"저발열: {low_row['stage']} (발열 {low_row['median_rise_c']:.1f}°C)",
                      fontsize=10)
    style.tray_heatmap(axes[1], f_high.to_numpy(), unit="°C", vmax=vmax,
                      show_margin_axis=False, cbar_label="°C")
    axes[1].set_title(f"고발열: {high_row['stage']} (발열 {high_row['median_rise_c']:.1f}°C)",
                      fontsize=10)

    style.verdict_badge(fig, "high_soft")
    style.chain_strip(fig, "cause")
    path = os.path.join(outdir, "V4b_저발열_고발열_지도비교.png")
    return style.save(fig, path)


def fig_v4_all(df: pd.DataFrame, outdir: str) -> dict:
    os.makedirs(outdir, exist_ok=True)
    rise_cols = _default_rise_stat_cols()
    reg = fan.heat_vs_amplitude_regression(df, rise_cols)
    out = {}
    if reg.get("ok"):
        out["v4a"] = fig_v4a_heat_vs_amplitude(reg, outdir)
        out["v4b"] = fig_v4b_heat_map_compare(df, reg, rise_cols, outdir)
    if "docv7_raw" in df.columns:
        out["s3"] = fig_s3_ocv_temp_corr(df, outdir)
    return out


# ---------------------------------------------------------------------------
# V5/V6/V7 — "영향 없음 / 미확정" 정직성 그림 ⚪⚫
# ---------------------------------------------------------------------------


def fig_v5_time_bias(k3: dict, outdir: str) -> str:
    """V5 — 트레이별 에이징 실제 경과시간과 불량률 상관 (기울기 없어야 정상)."""
    if not k3.get("ok") or "table" not in k3:
        raise ValueError(f"aging_duration_bias_check 실패 - V5 계산 불가: {k3}")
    table = k3["table"]

    fig, ax = style.kind_fig(
        "기다린 시간은 판정에 영향을 주지 않았습니다",
        "점 하나가 트레이 하나입니다. 기울기가 없으면(평평하면) 영향이 없다는 뜻입니다.",
        figsize=(8.6, 6.2), top=0.80)

    x = table["hours"].to_numpy()
    y = table["fail_rate"].to_numpy() * 100
    ax.scatter(x, y, s=22, color=style.C["accent"], alpha=0.65, zorder=3)
    if np.std(x) > 1e-9:
        slope, intercept = (float(v) for v in np.polyfit(x, y, 1))
        xs_line = np.linspace(float(x.min()), float(x.max()), 50)
        ax.plot(xs_line, slope * xs_line + intercept, color=style.C["rule"], lw=2,
               ls="--", zorder=2, label=f"추세선 (r={k3['corr_hours_vs_failrate']:.2f})")
        ax.legend(loc="upper right")
    lo, hi = k3["hours_range"]
    span_pct = (hi - lo) / lo * 100 if lo else float("nan")
    ax.text(0.02, 0.96, f"트레이 간 시간차 {hi - lo:.1f}h ({span_pct:.0f}%)",
           transform=ax.transAxes, fontsize=9.5, color=style.C["neutral"], va="top")
    ax.set_xlabel("트레이 실제 에이징 경과시간 (h)")
    ax.set_ylabel("트레이 불량률 (%)")

    style.verdict_badge(fig, "neutral")
    style.chain_strip(fig, ("cause", "verdict"))
    path = os.path.join(outdir, "V5_시간편향_없음.png")
    return style.save(fig, path)


def fig_v6_indicator_premise(df: pd.DataFrame, outdir: str) -> str:
    """V6 — S_A/S_B/S_C 직전 휴지시간(이완 여부) + 실제 강하량 크기(로그축) 2패널."""
    rest = indicators.rest_time_report(df)
    if rest.empty or "지표" not in rest.columns:
        raise ValueError("rest_time_report 결과 없음 - V6 계산 불가")

    keys = [i.key for i in schema.INDICATORS]
    mags = {k: float(pd.to_numeric(df[k], errors="coerce").abs().median())
           for k in keys if k in df.columns}
    rest_col = "직전휴지_중앙값(h)"
    rest_map = {row["지표"]: row[rest_col] for _, row in rest.iterrows()
               if rest_col in row and pd.notna(row.get(rest_col))}
    if not rest_map or not mags:
        raise ValueError("휴지시간/강하량 계산 불가 - V6 계산 불가")

    fig, axes = style.kind_fig(
        "자기방전을 3번 잰다고 생각했지만, 실제로는 일부만 자기방전입니다",
        "왼쪽은 측정 직전 쉰 시간, 오른쪽은 그래서 나온 값의 크기(중앙값, 로그축)입니다. "
        "휴지시간이 짧으면(점선 아래) 자기방전이 아니라 전압 안정화를 잰 것일 수 있습니다.",
        figsize=(10.6, 5.0), nrows=1, ncols=2, top=0.76, gridspec_kw={"wspace": 0.4})

    labels = list(rest_map)
    axes[0].bar(labels, [rest_map[k] for k in labels], color=style.C["accent"], zorder=3)
    axes[0].axhline(3, color=style.C["rule"], ls="--", lw=1.3, zorder=2,
                    label="이완 의심 기준 (3h)")
    axes[0].set_ylabel("측정 직전 휴지시간 (h)")
    axes[0].legend(loc="upper left")

    mlabels = list(mags)
    mvals = [mags[k] for k in mlabels]
    axes[1].bar(mlabels, mvals, color=style.C["accent"], zorder=3)
    axes[1].set_yscale("log")
    # matplotlib 로그축 막대는 기본적으로 축 하단을 '데이터 최솟값 바로 아래'로
    # 딱 맞춰 자동확대한다 — 세 지표가 실제로는 같은 자릿수(예 25% 이내)인데도
    # 막대 하나만 축 하단에 거의 붙어 나머지보다 훨씬 작아 보이는 착시가 생긴다
    # (실측 확인). 최솟값보다 한 자릿수 아래로 하단을 고정해 시각적으로 정직하게 만든다.
    bottom = 10 ** (np.floor(np.log10(min(mvals))) - 1)
    axes[1].set_ylim(bottom=bottom)
    # 로그축 기본 지수표기(예 10^-1, 보조눈금 포함)는 mathtext 로 그려지는데, 이
    # mathtext 폰트가 unicode 마이너스(U+2212) 글리프를 못 찾아 경고+깨진 기호로
    # 나온다(실측 확인 — 주눈금만 바꿔선 안 없어지고 보조눈금 포매터도 꺼야 했다).
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
    axes[1].yaxis.set_minor_formatter(NullFormatter())
    axes[1].set_ylabel("실제 강하량 중앙값 |값| (mV, 로그축)")

    style.verdict_badge(fig, "neutral")
    style.chain_strip(fig, "cause")
    path = os.path.join(outdir, "V6_자기방전3지표_전제붕괴.png")
    return style.save(fig, path)


def fig_v7_fan_mismatch(minima: dict, outdir: str) -> str:
    """V7 — 도면 예측 냉각열(예측 차가운 열) vs 나머지 열 온도 잔차 비교."""
    band_results = {k: v for k, v in minima.items() if k != "note"}
    if not band_results:
        raise ValueError("predicted_minima_check 결과 없음 - V7 계산 불가")

    fig, ax = style.kind_fig(
        "냉각팬이 원인이라는 가설은 이번 데이터로 확인되지 않았습니다",
        "도면에서 미리 예측한 '차가울 자리'와 실제 측정을 비교했습니다. 예측 열이 "
        "더 차가워야(막대가 낮아야) 가설이 맞습니다.",
        figsize=(9.2, 6.2), top=0.80)

    names = list(band_results)
    x = np.arange(len(names))
    pred = [band_results[n]["predicted_mean"] for n in names]
    other = [band_results[n]["other_cols_mean"] for n in names]
    w = 0.35
    ax.bar(x - w / 2, pred, width=w, color=style.C["low"], label="예측한 차가운 열", zorder=3)
    ax.bar(x + w / 2, other, width=w, color=style.C["neutral"], label="나머지 열", zorder=3)
    ax.set_xticks(x, names)
    ax.set_ylabel("등방성분 제거 후 온도 잔차 (°C)")
    ax.legend()

    n_match = sum(1 for n in names if band_results[n]["colder_as_predicted"])
    ax.text(0.02, 0.96, f"{n_match}/{len(names)} 밴드에서 예측대로 더 차가움",
           transform=ax.transAxes, fontsize=10, color=style.C["high"],
           fontweight="bold", va="top")

    style.verdict_badge(fig, "accent")
    style.chain_strip(fig, "cause")
    path = os.path.join(outdir, "V7_팬_예측_불일치.png")
    return style.save(fig, path)


def fig_v567_all(df: pd.DataFrame, outdir: str, main_temp_col: str | None = None) -> dict:
    os.makedirs(outdir, exist_ok=True)
    out = {}
    k3 = timing.aging_duration_bias_check(df)
    if k3.get("ok"):
        out["v5"] = fig_v5_time_bias(k3, outdir)
    try:
        out["v6"] = fig_v6_indicator_premise(df, outdir)
    except ValueError:
        pass

    if main_temp_col is None:
        main_temp_col = next((c for c in ("t_ocv2", "t_ocv4") if c in df.columns), None)
    if main_temp_col:
        profiles = fan.band_column_profile(df, main_temp_col, detrend=True)
        minima = fan.predicted_minima_check(profiles)
        out["v7"] = fig_v7_fan_mismatch(minima, outdir)
    return out
