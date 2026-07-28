"""발표용 시각화 기반 — docs/03_viz_guidelines.md 의 규칙을 코드로 강제한다.

핵심 사용법:
    from tray_ocv2 import style
    style.init()                                   # 한글 폰트·기본 스타일
    fig, ax = style.kind_fig("결론 문장", "이 그림 읽는 법")
    style.tray_heatmap(ax, grid, unit="mV")
    style.save(fig, "reports/figures/E1_docv7_지문.png")

설계 원칙
  - 제목은 '결론 문장', 부제는 '읽는 법' → kind_fig 가 두 개를 모두 요구한다.
  - 전압 수치는 항상 판정 기준(0.8 mV) 대비 %를 병기 → mv_text() / 컬러바 이중 눈금.
  - 발산형 컬러맵은 항상 0 중심 대칭 → tray_heatmap 이 강제.
"""
from __future__ import annotations

import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

#: 현행 판정 오프셋 (mV). 모든 크기 비교의 기준자.
JUDGE_MARGIN_MV = 0.8

#: 트레이 격자
N_ROWS, N_COLS = 12, 12
ROW_LABELS = list("ABCDEFGHIJKL")          # 행 A~L
COL_LABELS = [str(i) for i in range(1, 13)]  # 열 1~12

#: 충방전기 냉각팬 배열 (docs/00_facts.md §3.1). (행중심, [열중심...])
#: 도면 픽셀 판독값 — CAD 검증 전. 직경은 미확정이라 표시용 기본값만 둔다.
FAN_LAYOUT = [
    (2.6, [1.9, 5.2, 8.5, 11.5]),      # 상단 4개
    (6.4, [2.0, 4.3, 6.6, 8.9, 11.2]),  # 중앙 5개
    (10.2, [1.9, 5.1, 8.4, 11.4]),     # 하단 4개
]
FAN_DIAMETER_CELLS = 2.2   # 미확정. 오버레이 표시용일 뿐 분석에 쓰지 않는다.

#: 색 언어 (docs/03_viz_guidelines.md §2)
C = {
    "high": "#C1272D",      # 높음/위험/불량 방향
    "high_soft": "#E8837F",
    "low": "#1F5FA9",       # 낮음/안전 방향
    "low_soft": "#7FA8D4",
    "neutral": "#8A8F98",   # 정상/배경
    "accent": "#12355B",    # 이 슬라이드의 주인공
    "rule": "#111111",      # 기준선
    "noise": "#DDDDDD",     # 잡음 밴드
    "grid": "#D8DBE0",
    "missing": "#FFFFFF",
}

#: 발산형 컬러맵 (파랑–흰–빨강, 색맹 안전). 무지개 금지.
CMAP_DIV = LinearSegmentedColormap.from_list(
    "tray_div", ["#1F5FA9", "#7FA8D4", "#EDF2F7", "#E8837F", "#C1272D"])
#: 순차형 (파랑 단색)
CMAP_SEQ = LinearSegmentedColormap.from_list(
    "tray_seq", ["#F4F7FB", "#7FA8D4", "#1F5FA9", "#12355B"])

_KOREAN_FONT_CANDIDATES = [
    "Malgun Gothic", "NanumGothic", "Noto Sans KR",
    "Noto Sans CJK KR", "Gulim", "Batang",
]

_initialized = False


# ---------------------------------------------------------------------------
# 초기화
# ---------------------------------------------------------------------------
def find_korean_font() -> str | None:
    """설치된 한글 폰트 중 첫 번째를 반환. 없으면 None."""
    import matplotlib.font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    for name in _KOREAN_FONT_CANDIDATES:
        if name in available:
            return name
    return None


def init(font: str | None = None) -> str | None:
    """한글 폰트 + 발표용 기본 스타일 적용. 반환: 사용된 폰트명(없으면 None)."""
    global _initialized
    chosen = font or find_korean_font()
    if chosen is None:
        warnings.warn(
            "한글 폰트를 찾지 못했습니다. 그림 라벨이 깨질 수 있습니다. "
            "Malgun Gothic 또는 NanumGothic 설치를 권합니다.")
    else:
        mpl.rcParams["font.family"] = chosen

    mpl.rcParams.update({
        "axes.unicode_minus": False,     # 한글 폰트에서 마이너스 깨짐 방지
        "figure.dpi": 110,
        "savefig.dpi": 200,              # R9: 회의실 프로젝터 기준
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,                 # R9
        "axes.titlesize": 13,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#5A6069",
        "axes.grid": True,
        "grid.color": C["grid"],
        "grid.linewidth": 0.7,
        "grid.alpha": 0.8,
        "lines.linewidth": 2.0,
        "patch.linewidth": 0.6,
    })
    _initialized = True
    return chosen


def _ensure_init():
    if not _initialized:
        init()


# ---------------------------------------------------------------------------
# 단위 · 텍스트 (R3: 항상 판정 기준 대비 %)
# ---------------------------------------------------------------------------
def margin_pct(mv) -> float:
    """mV → 판정 기준(0.8 mV) 대비 %."""
    return np.asarray(mv, dtype=float) / JUDGE_MARGIN_MV * 100.0


def mv_text(mv: float, digits: int = 3) -> str:
    """R3·R7 — '0.048 mV (판정 기준의 6 %)' 형식 문자열."""
    return f"{mv:.{digits}g} mV (판정 기준의 {margin_pct(mv):.0f} %)"


# ---------------------------------------------------------------------------
# 그림 생성 (R1·R2 강제)
# ---------------------------------------------------------------------------
def kind_fig(conclusion: str, how_to_read: str, figsize=(9.5, 6.6),
             nrows: int = 1, ncols: int = 1, top: float = 0.845, **kw):
    """'친절한 그림' 생성기.

    conclusion  : 제목 = 결론 문장 (방법 이름 금지)  [R1]
    how_to_read : 부제 = 이 그림 읽는 법 1~2줄       [R2]
    top         : 본문 영역 상단 위치. 부제가 길거나 x축 라벨이 위에 오면 낮춘다.

    반환: (fig, ax) 또는 (fig, axes배열)
    """
    _ensure_init()
    if not conclusion or not how_to_read:
        raise ValueError("R1·R2: 결론 제목과 '읽는 법' 부제는 둘 다 필수입니다.")

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kw)
    fig.suptitle(conclusion, fontsize=15.5, fontweight="bold",
                 color=C["accent"], x=0.02, ha="left", y=0.985)
    fig.text(0.02, 0.930, how_to_read, fontsize=11, color="#3C4149",
             ha="left", va="top", linespacing=1.5)
    fig.subplots_adjust(top=top)
    return fig, axes


def note(ax, text: str, xy, xytext=None, color=None):
    """R10 — 그림 안에 화살표 주석을 직접 단다."""
    color = color or C["accent"]
    ax.annotate(
        text, xy=xy, xytext=xytext or (xy[0] + 0.6, xy[1] + 0.6),
        fontsize=10.5, color=color, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=color, lw=1.6,
                        connectionstyle="arc3,rad=0.15"),
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=color, alpha=0.9))


def margin_rule(ax, value_mv: float = JUDGE_MARGIN_MV, axis: str = "y",
                label: str | None = None):
    """R6 — 판정 기준선을 긋는다."""
    label = label or f"판정 기준 {value_mv} mV"
    fn = ax.axhline if axis == "y" else ax.axvline
    fn(value_mv, color=C["rule"], ls="--", lw=1.5, label=label, zorder=5)


def noise_band(ax, level_mv: float, axis: str = "y",
               label: str = "측정 잡음 수준 (이 아래는 의미 없음)"):
    """R6 — 잡음 바닥을 회색 밴드로. 이 아래 신호는 해석하지 않는다."""
    if axis == "y":
        ax.axhspan(-level_mv, level_mv, color=C["noise"], alpha=0.7,
                   zorder=0, label=label)
    else:
        ax.axvspan(-level_mv, level_mv, color=C["noise"], alpha=0.7,
                   zorder=0, label=label)


# ---------------------------------------------------------------------------
# 12×12 트레이 히트맵 (docs/03_viz_guidelines.md §3 표준형)
# ---------------------------------------------------------------------------
def tray_heatmap(ax, grid, *, unit: str = "mV", diverging: bool = True,
                 vmax: float | None = None, show_fans: bool = False,
                 show_margin_axis: bool = True, cbar: bool = True,
                 cbar_label: str | None = None):
    """트레이 12×12 필드 히트맵 (표준형).

    grid : (12, 12) 배열. NaN = 결측 → 흰색 + 해칭.
    diverging=True 면 0 중심 대칭 스케일을 **강제**한다 (착시 방지).
    show_margin_axis=True 면 컬러바에 '판정 기준 대비 %' 이중 눈금을 붙인다.
    """
    _ensure_init()
    g = np.asarray(grid, dtype=float)
    if g.shape != (N_ROWS, N_COLS):
        raise ValueError(f"grid 는 ({N_ROWS}, {N_COLS}) 여야 합니다: {g.shape}")

    finite = np.isfinite(g)
    if vmax is None:
        vmax = float(np.nanmax(np.abs(g))) if finite.any() else 1.0
        vmax = vmax if vmax > 0 else 1.0

    if diverging:
        cmap, vmin = CMAP_DIV, -vmax   # 0 중심 대칭 강제
    else:
        cmap, vmin = CMAP_SEQ, float(np.nanmin(g)) if finite.any() else 0.0

    # 결측 셀: 흰 바탕 + 해칭
    ax.set_facecolor("white")
    for r, c in zip(*np.where(~finite)):
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=True,
                                   facecolor=C["missing"], edgecolor="#B8BDC4",
                                   hatch="///", lw=0.6, zorder=1))

    im = ax.imshow(np.ma.masked_invalid(g), cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", zorder=2)

    # 행 A~L / 열 1~12 라벨 (열 번호는 아래쪽 — 위쪽은 부제 자리)
    ax.set_xticks(range(N_COLS), COL_LABELS)
    ax.set_yticks(range(N_ROWS), ROW_LABELS)
    ax.set_xlabel("열 (트레이 가로 위치)")
    ax.set_ylabel("행 (트레이 세로 위치)")
    ax.set_xticks(np.arange(-0.5, N_COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N_ROWS, 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", lw=1.0)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)

    if show_fans:
        _overlay_fans(ax)

    if cbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
        cb.outline.set_visible(False)
        # 기본 눈금(mV) 라벨은 왼쪽으로 빼서 이중 눈금과 충돌시키지 않는다.
        cb.ax.yaxis.set_label_position("left")
        cb.set_label(cbar_label or f"트레이 중앙값 대비 차이 [{unit}]",
                     fontsize=10.5, labelpad=8)
        if show_margin_axis and unit == "mV":
            # R3 — 판정 기준 대비 % 이중 눈금 (오른쪽)
            sec = cb.ax.secondary_yaxis(
                1.0, functions=(lambda v: v / JUDGE_MARGIN_MV * 100,
                                lambda p: p * JUDGE_MARGIN_MV / 100))
            sec.set_ylabel(f"판정 기준({JUDGE_MARGIN_MV} mV) 대비 [%]",
                           fontsize=10, labelpad=6)
            sec.tick_params(labelsize=9.5)
    return im


def _overlay_fans(ax):
    """충방전기 냉각팬 위치 오버레이. 원인 파트에서만 사용."""
    r_cells = FAN_DIAMETER_CELLS / 2.0
    for row_c, cols in FAN_LAYOUT:
        for col_c in cols:
            # 셀 좌표(1-based) → imshow 인덱스(0-based)
            circ = plt.Circle((col_c - 1, row_c - 1), r_cells, fill=False,
                              ec="#2F3337", lw=1.8, ls="--", alpha=0.85,
                              zorder=6)
            ax.add_patch(circ)
    ax.plot([], [], ls="--", color="#2F3337", lw=1.8,
            label="냉각팬 위치 (직경은 추정)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=1)


# ---------------------------------------------------------------------------
# 저장
# ---------------------------------------------------------------------------
def save(fig, path, close: bool = True):
    """그림 저장 (경로 자동 생성)."""
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path)
    if close:
        plt.close(fig)
    return path
