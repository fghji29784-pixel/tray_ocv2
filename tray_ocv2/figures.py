"""발표 산출물 그림 — docs/06_visualization_plan.md 설계를 코드로 옮긴다.

현재는 S1(전체 지문 갤러리, 4장)만 구현되어 있다. V0~V16, S3, S4 는
docs/06_visualization_plan.md §6.3 구현순서를 따라 순차적으로 추가한다.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import fields, schema, style

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
