"""시각화 스타일 데모 — 합성 데이터로 표준 그림 3종을 생성한다.

실행: python -m tests.demo_style
산출: reports/figures/demo/*.png
"""
from __future__ import annotations

import numpy as np

from tray_ocv2 import style
from tray_ocv2.style import C, JUDGE_MARGIN_MV, N_COLS, N_ROWS


def _synthetic_field(seed: int = 0) -> np.ndarray:
    """테두리 高 링 + 팬 밴드 구조 + 고립 핫셀 + 결측 을 섞은 합성 필드 (mV)."""
    rng = np.random.default_rng(seed)
    rr, cc = np.mgrid[1:N_ROWS + 1, 1:N_COLS + 1]

    # 1) 테두리 高 링 (팬 배열 세로 바깥인 1·12행에서 특히 강함)
    edge_dist = np.minimum.reduce([rr - 1, N_ROWS - rr, cc - 1, N_COLS - cc])
    ring = 0.16 * np.exp(-edge_dist / 1.6)

    # 2) 팬 밴드 구조: 바깥 밴드는 열 주기 3.2, 중앙 밴드는 2.3
    band = np.zeros_like(ring)
    outer = (rr <= 4) | (rr >= 9)
    band[outer] = 0.05 * np.cos(2 * np.pi * cc[outer] / 3.2)
    band[~outer] = 0.05 * np.cos(2 * np.pi * cc[~outer] / 2.3)

    field = ring + band + rng.normal(0, 0.02, ring.shape)

    # 3) 고립 핫셀 (진짜 불량) + 결측
    for r, c in [(4, 7), (9, 2)]:
        field[r, c] += 0.9
    for r, c in [(2, 10), (7, 3), (11, 6)]:
        field[r, c] = np.nan

    return field - np.nanmedian(field)


def demo_heatmap(field, show_fans, tag, conclusion, how_to_read):
    fig, ax = style.kind_fig(conclusion, how_to_read, figsize=(9.2, 7.8),
                             top=0.86)
    style.tray_heatmap(ax, field, unit="mV", show_fans=show_fans)
    hot = np.unravel_index(np.nanargmax(field), field.shape)
    style.note(ax, f"이 셀만 유독 높음\n{style.mv_text(field[hot])}",
               xy=(hot[1], hot[0]), xytext=(hot[1] - 4.6, hot[0] + 2.8))
    return style.save(fig, f"reports/figures/demo/{tag}.png")


def demo_scale_bar():
    """V1 — 0.8 mV 가 얼마나 작은지 감각을 주는 그림."""
    items = [
        ("셀 전압 (완충 시)", 3600.0, C["neutral"]),
        ("판정 기준", JUDGE_MARGIN_MV, C["rule"]),
        ("자리에 따른 차이 (트레이별)", 0.286, C["high"]),
        ("모든 트레이 공통 무늬", 0.048, C["low"]),
    ]
    fig, ax = style.kind_fig(
        "판정 기준 0.8 mV 는 셀 전압의 0.02 % 에 불과한 아주 작은 값입니다",
        "가로축은 로그 눈금입니다. 오른쪽으로 한 칸 갈 때마다 10배 커집니다.\n"
        "우리가 다루는 '자리에 따른 차이'는 판정 기준의 6~36 % 크기입니다.",
        figsize=(9.6, 4.6))
    names = [n for n, _, _ in items]
    vals = [v for _, v, _ in items]
    cols = [c for _, _, c in items]
    y = np.arange(len(items))[::-1]
    ax.barh(y, vals, color=cols, height=0.55)
    for yi, (n, v, _) in zip(y, items):
        pct = v / JUDGE_MARGIN_MV * 100
        txt = f"  {v:,.3g} mV" + (f"  (판정 기준의 {pct:.0f} %)" if v < 100 else "")
        ax.text(v * 1.15, yi, txt, va="center", fontsize=11)
    ax.set_yticks(y, names)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 3e5)
    ax.set_xlabel("크기 [mV] — 로그 눈금")
    ax.grid(axis="y", visible=False)
    return style.save(fig, "reports/figures/demo/V1_스케일감각.png")


def main():
    font = style.init()
    print(f"[폰트] {font}")
    field = _synthetic_field()
    paths = [
        demo_heatmap(
            field, False, "E1_트레이지문",
            "트레이 가장자리 셀이 중앙보다 전압 강하량이 큽니다",
            "칸 하나가 셀 하나입니다. 빨간색일수록 전압이 많이 떨어진 셀입니다.\n"
            "색은 같은 트레이 안 중앙값과의 차이이며, 오른쪽 눈금은 판정 기준 대비 비율입니다."),
        demo_heatmap(
            field, True, "F2_팬오버레이",
            "가장자리가 높은 이유 — 세로 1번과 12번 줄은 냉각팬 배열 바깥에 놓입니다",
            "점선 원이 충방전기 냉각팬 위치입니다 (직경은 도면 확인 전 추정값).\n"
            "팬은 세로 방향으로 3개 밴드(4개-5개-4개)를 이룹니다. "
            "팬 아래는 잘 식고, 팬 사이와 배열 바깥은 덜 식습니다."),
        demo_scale_bar(),
    ]
    for p in paths:
        print(f"[저장] {p}")


if __name__ == "__main__":
    main()
