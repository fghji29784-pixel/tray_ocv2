"""명령행 진입점.

사용법:
    python -m tray_ocv2.cli inspect --input <파일>
    python -m tray_ocv2.cli run     --input <파일> --outdir reports
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from . import fan, indicators, io_load, qc, schema, style, timing
from .fields import position_field

# Windows 콘솔 기본 인코딩(cp949)은 —, ★ 같은 유니코드 문자에서 죽는다.
# UTF-8로 강제해 크래시를 막는다 (터미널에 깨져 보이면 `chcp 65001` 권장).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def _print_dict(d: dict, indent: int = 0):
    pad = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{pad}{k}:")
            _print_dict(v, indent + 1)
        else:
            print(f"{pad}{k}: {v}")


def cmd_inspect(args):
    info = io_load.inspect(args.input)
    print(f"[행 수] {info['n_rows']}  [원본 컬럼] {info['n_cols_actual']}  "
         f"[기대 컬럼] {info['n_cols_expected']}")
    print(f"\n[기대했으나 파일에 없는 컬럼] {len(info['missing_expected'])}개")
    for c in info["missing_expected"]:
        print(f"  - {c}")
    print(f"\n[파일에 있으나 매핑 안 된 컬럼] {len(info['unmapped_actual'])}개")
    for c in info["unmapped_actual"][:40]:
        print(f"  - {c}")
    if len(info["unmapped_actual"]) > 40:
        print(f"  ... 외 {len(info['unmapped_actual']) - 40}개")


def cmd_run(args):
    style.init()
    os.makedirs(args.outdir, exist_ok=True)
    fig_dir = os.path.join(args.outdir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"[로딩] {args.input}")
    raw = io_load.read_any(args.input)
    df = io_load.build_cell_table(raw)
    print(f"[셀 수] {len(df)}  [트레이 수] {df['tray_id'].nunique() if 'tray_id' in df else '?'}")

    report_lines = ["# 자동 진단 리포트\n"]

    # --- A: 신뢰성 ---------------------------------------------------------
    print("\n=== [A] 데이터 신뢰성 ===")
    report_lines.append("## A. 데이터 신뢰성\n")

    a6 = schema.verify_position_consistency(raw)
    print("[A6] 위치 파싱 일관성:", a6.get("ok"))
    report_lines.append(f"- A6 위치 일관성: {a6}\n")

    sc = qc.score_card(df)
    print(sc.to_string(index=False))
    report_lines.append("### 스코어카드\n\n" + sc.to_markdown(index=False) + "\n")

    a7 = qc.grade_consistency(df)
    print("\n[A7] 등급 구성 분해:")
    _print_dict(a7)
    report_lines.append(f"### A7 등급 구성 분해\n\n```\n{a7}\n```\n")

    a9 = qc.docv7_unit_check(df)
    print("\n[A9] docv7 단위/정의 대조:")
    _print_dict(a9)
    report_lines.append(f"### A9 단위 대조\n\n```\n{a9}\n```\n")

    dup = qc.duplicate_cells(df)
    print(f"\n[A5] 중복 셀 행: {len(dup)}건")

    clock = timing.clock_sanity(df)
    print("\n[A8] 시계 정합성 (음수 구간 있으면 이상):")
    if len(clock):
        print(clock.to_string(index=False))
    report_lines.append("### A8 시계 정합성\n\n" +
                        (clock.to_markdown(index=False) if len(clock) else "(계산 불가)") + "\n")

    # --- K: 시간 축 ----------------------------------------------------------
    print("\n=== [K] 공정 순서 / 시간 ===")
    order = timing.infer_process_order(df)
    print(order.to_string(index=False) if len(order) else "(타임스탬프 컬럼 없음)")
    report_lines.append("## K. 공정 순서 실측 (infer_process_order)\n\n" +
                        (order.to_markdown(index=False) if len(order) else "(없음)") + "\n")

    # --- C: 3지표 삼각검증 ----------------------------------------------------
    print("\n=== [C] 자기방전 3지표 삼각검증 (최우선 관문) ===")
    df = indicators.compute(df)
    df = indicators.tray_relative(df)

    rest = indicators.rest_time_report(df)
    print("\n[K10] OCV 직전 휴지시간 (S_A/S_B 가 이완인지 판별):")
    print(rest.to_string(index=False) if len(rest) else "(휴지시간 계산 불가)")
    report_lines.append("## C. 자기방전 3지표\n\n### K10 직전 휴지시간\n\n" +
                        (rest.to_markdown(index=False) if len(rest) else "(없음)") + "\n")

    tri = indicators.triangulate(df)
    print("\n[C1] within-tray 상관행렬:")
    print(tri.to_string() if len(tri) else "(계산 불가 — S_A/S_B/S_C 컬럼 부족)")
    report_lines.append("### C1 상관행렬\n\n" +
                        (tri.to_markdown() if len(tri) else "(없음)") + "\n")

    c2 = indicators.cross_check_grade(df)
    print("\n[C2] 등급 E 셀의 S_A/S_B 준독립 대조:")
    _print_dict(c2)
    report_lines.append(f"### C2 준독립 대조\n\n```\n{c2}\n```\n")

    # --- F: 팬 서명 (온도) ----------------------------------------------------
    print("\n=== [F] 팬 서명 검증 (온도 우선) ===")
    temp_cols = [c for c in df.columns if c.startswith("t_ocv") or c.startswith("temp_")]
    if temp_cols:
        f0 = fan.stage_field_reproducibility(df, temp_cols)
        print("[F0] 단계 간 온도 필드 재현성 상관행렬 (일부):")
        print(f0.iloc[:8, :8].to_string() if len(f0) else "(없음)")
        report_lines.append("## F. 팬 서명 검증\n\n### F0 온도 필드 재현성\n\n" +
                            (f0.to_markdown() if len(f0) else "(없음)") + "\n")

        main_temp = "t_ocv2" if "t_ocv2" in df.columns else temp_cols[0]
        f1 = fan.row_profile(df, main_temp)
        print(f"\n[F1] 세로 프로파일 ({main_temp}):")
        print(f1.to_string(index=False))
        report_lines.append(f"### F1 세로 프로파일 ({main_temp})\n\n" +
                            f1.to_markdown(index=False) + "\n")

        profiles = fan.band_column_profile(df, main_temp)
        sim = fan.band_similarity(profiles)
        print(f"\n[F2] ★판결문 — 밴드 유사도 ({main_temp}):")
        _print_dict(sim)
        report_lines.append(f"### F2 밴드 유사도 ({main_temp})\n\n```\n{sim}\n```\n")

        f3 = fan.amplitude_vs_heat(
            df, no_heat_cols=[c for c in schema_no_heat_cols(df)],
            heat_cols=[c for c in ("temp_charge1_mean", "temp_discharge1_mean")
                      if c in df.columns])
        print("\n[F3] 발열 유무에 따른 진폭 비교:")
        _print_dict(f3)
        report_lines.append(f"### F3 진폭 vs 발열\n\n```\n{f3}\n```\n")

        # 그림: OCV2 온도 필드 (평균 + 팬 오버레이)
        field = position_field(
            df.assign(_dev=df[main_temp] - df.groupby("tray_id")[main_temp].transform("median")),
            "_dev", agg="mean")
        fig, ax = style.kind_fig(
            f"셀 온도({main_temp})가 자리마다 다릅니다",
            "칸 하나가 셀 하나. 색은 같은 트레이 안 중앙값과의 차이(°C)입니다.\n"
            "점선 원은 충방전기 냉각팬 위치(직경은 추정)입니다.",
            figsize=(9.2, 7.8), top=0.86)
        style.tray_heatmap(ax, field.to_numpy(), unit="°C", show_fans=True,
                          show_margin_axis=False,
                          cbar_label="트레이 중앙값 대비 온도차 [°C]")
        style.save(fig, os.path.join(fig_dir, "F_온도지문.png"))
        print(f"\n[저장] {fig_dir}/F_온도지문.png")
    else:
        print("(온도 컬럼이 없어 F 그룹을 건너뜁니다)")

    with open(os.path.join(args.outdir, "report_auto.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n[저장] {args.outdir}/report_auto.md")

    out_csv = os.path.join(args.outdir, "processed_cell_table.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[저장] {out_csv}")


def schema_no_heat_cols(df: pd.DataFrame):
    for key in schema.NO_HEAT_STEPS:
        for cand in (f"t_{key}", f"temp_{key}_single", f"temp_{key}_mean"):
            if cand in df.columns:
                yield cand
                break


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tray_ocv2")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_inspect = sub.add_parser("inspect", help="컬럼 매핑 대조")
    p_inspect.add_argument("--input", required=True)
    p_inspect.set_defaults(func=cmd_inspect)

    p_run = sub.add_parser("run", help="전체 진단 파이프라인 실행")
    p_run.add_argument("--input", required=True)
    p_run.add_argument("--outdir", default="reports")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
