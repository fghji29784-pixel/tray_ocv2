"""발표 산출물 그림 — docs/06_visualization_plan.md 설계를 코드로 옮긴다.

현재는 S1(전체 지문 갤러리, 4장)만 구현되어 있다. V0~V16, S3, S4 는
docs/06_visualization_plan.md §6.3 구현순서를 따라 순차적으로 추가한다.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import fields, impact, schema, spatial, style

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
