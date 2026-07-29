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

from . import fan, impact, indicators, io_load, patterns, qc, schema, spatial, style, timing
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
    print("\n[C1] within-tray 상관행렬 (원본 — 신뢰도 보정 전, 감쇠 가능):")
    print(tri.to_string() if len(tri) else "(계산 불가 — S_A/S_B/S_C 컬럼 부족)")
    report_lines.append("### C1 상관행렬 (원본)\n\n" +
                        (tri.to_markdown() if len(tri) else "(없음)") + "\n")

    rel = indicators.reliability_report(df)
    print("\n[C1-신뢰도] 오류8 — 상관계수는 신뢰도가 낮으면 자동으로 0쪽으로 감쇠된다:")
    _print_dict(rel)
    report_lines.append(f"### C1 신뢰도 추정 (오류8)\n\n```\n{rel}\n```\n")

    if len(tri):
        dis = indicators.disattenuated_matrix(tri, rel)
        print("\n[C1-보정] 신뢰도로 보정한 상관행렬 (진짜 성분 상관의 추정치, 과대추정 가능):")
        print(dis.to_string())
        report_lines.append("### C1 감쇠보정 상관행렬\n\n" + dis.to_markdown() + "\n")

    tail = indicators.tail_consistency_check(df)
    print("\n[C1-꼬리/오류11] ★희귀사건용 — 판정 대상 꼬리가 두 반분 모두에서 일관되게 높은가:")
    _print_dict(tail)
    report_lines.append(f"### C1 꼬리 일관성 (오류11)\n\n```\n{tail}\n```\n")

    tail_rate = indicators.tail_rate_check(df)
    print("\n[C1-꼬리율] ★가속 열화인가 구간길이 차이인가 — 율(mV/day)로 재확인:")
    _print_dict(tail_rate)
    report_lines.append(f"### C1 꼬리 가속열화 판별\n\n```\n{tail_rate}\n```\n")

    hot_summary = {
        ind.key: {
            "고온노출_중앙값(h)": round(float(df[f"{ind.key}_hot_hours"].median()), 2)
            if f"{ind.key}_hot_hours" in df.columns else None,
            "율_원본_중앙값(mV/day)": round(float(df[f"{ind.key}_rate"].median()), 4)
            if f"{ind.key}_rate" in df.columns and df[f"{ind.key}_rate"].notna().any() else None,
            "율_아레니우스등가_중앙값(mV/day)": round(float(df[f"{ind.key}_rate_arrhenius"].median()), 4)
            if f"{ind.key}_rate_arrhenius" in df.columns and df[f"{ind.key}_rate_arrhenius"].notna().any() else None,
        } for ind in schema.INDICATORS
    }
    print("\n[K5/오류9] 지표별 고온노출시간·아레니우스 등가율 (Ea=54kJ/mol 가정, 민감도 확인 필요):")
    _print_dict(hot_summary)
    report_lines.append(f"### 아레니우스 등가율 (오류9, Ea=54kJ/mol 가정)\n\n```\n{hot_summary}\n```\n")

    c2 = indicators.cross_check_grade(df)
    print("\n[C2] 등급 E 셀의 S_A/S_B 준독립 대조:")
    _print_dict(c2)
    report_lines.append(f"### C2 준독립 대조\n\n```\n{c2}\n```\n")

    k3 = timing.aging_duration_bias_check(df)
    print("\n[K3] 트레이별 에이징 실제시간 편차 → 불량률 상관 (시간 축 판정 편향):")
    _print_dict(k3)
    report_lines.append(f"### K3 시간 편향\n\n```\n{k3}\n```\n")

    # --- F: 팬 서명 (온도) ----------------------------------------------------
    print("\n=== [F] 팬 서명 검증 (온도 우선) ===")
    temp_cols = [c for c in df.columns if c.startswith("t_ocv") or c.startswith("temp_")]
    if temp_cols:
        f0 = fan.stage_field_reproducibility(df, temp_cols)
        print("[F0] 단계 간 온도 필드 재현성 상관행렬 (일부):")
        print(f0.iloc[:8, :8].to_string() if len(f0) else "(없음)")
        report_lines.append("## F. 팬 서명 검증\n\n### F0 온도 필드 재현성 (원본)\n\n" +
                            (f0.to_markdown() if len(f0) else "(없음)") + "\n")

        cluster = fan.cluster_reproducibility(f0) if len(f0) else {}
        print("\n[F0-판정/오류7] 군집별 재현성 — '모든 스텝이 같아야' 가 아니라 "
             "'열이력이 같은 스텝끼리만' 재현되어야 한다:")
        _print_dict(cluster)
        report_lines.append(f"### F0 판정 — 군집별 재현성 (오류7)\n\n```\n{cluster}\n```\n")

        temp_groups = fan.classify_temp_cols(temp_cols)
        recent_heat_cols = [c for c in temp_groups.get("recent_heat", []) if c.startswith("t_")]
        main_temp = (recent_heat_cols[0] if recent_heat_cols
                    else ("t_ocv2" if "t_ocv2" in df.columns else temp_cols[0]))

        f1 = fan.row_profile(df, main_temp)
        print(f"\n[F1] 세로 프로파일 ({main_temp}, recent_heat 단계 중 선택):")
        print(f1.to_string(index=False))
        report_lines.append(f"### F1 세로 프로파일 ({main_temp})\n\n" +
                            f1.to_markdown(index=False) + "\n")

        profiles = fan.band_column_profile(df, main_temp, detrend=True)
        sim = fan.band_similarity(profiles)
        print(f"\n[F2] 밴드 유사도 ({main_temp}, 테두리거리 detrend 적용):")
        _print_dict(sim)
        report_lines.append(f"### F2 밴드 유사도 ({main_temp})\n\n```\n{sim}\n```\n")

        minima = fan.predicted_minima_check(profiles)
        print(f"\n[F2-판결문/오류6] 도면 예측 열(FAN_COL_COLD_OUTER/MIDDLE) 직접 검정:")
        _print_dict(minima)
        report_lines.append(f"### F2 판결문 — 예측 열 검정 (오류6)\n\n```\n{minima}\n```\n")

        radial = fan.fan_radial_profile(df, main_temp)
        radial_summary = {k: v for k, v in radial.items() if k != "table"}
        print("\n[F4/오류12] ★부호 논쟁 판별 — 팬 중심 거리별 반경 프로파일:")
        print(radial.get("table").to_string(index=False) if "table" in radial else "(없음)")
        _print_dict(radial_summary)
        report_lines.append(
            "### F4 반경 프로파일 — 부호 논쟁 판별 (오류12)\n\n"
            + (radial["table"].to_markdown(index=False) if "table" in radial else "")
            + f"\n\n```\n{radial_summary}\n```\n")

        aniso = fan.fan_radial_profile_anisotropic(df, main_temp)
        print("\n[F4-이방성] ★행/열 방향 분리 — 밴드경계 효과 vs 팬간격 효과 판별:")
        if aniso.get("ok"):
            print("  [세로축(허브 정체점 가설)]")
            print(aniso["row_axis_profile"].to_string(index=False))
            print("  ", aniso["row_axis_summary"])
            print("  [가로축(팬간격 가설, 밴드 내부만)]")
            print(aniso["col_axis_profile"].to_string(index=False))
            print("  ", aniso["col_axis_summary"])
        else:
            print(" ", aniso)
        report_lines.append(
            "### F4 이방성 분리 (행/열)\n\n"
            + (("**세로축**\n\n" + aniso["row_axis_profile"].to_markdown(index=False)
               + f"\n\n```\n{aniso['row_axis_summary']}\n```\n\n"
               + "**가로축**\n\n" + aniso["col_axis_profile"].to_markdown(index=False)
               + f"\n\n```\n{aniso['col_axis_summary']}\n```\n")
              if aniso.get("ok") else f"```\n{aniso}\n```\n"))

        no_heat_cols = [c for c in schema_no_heat_cols(df)
                        if fan.classify_temp_cols([c]).get("separate_instrument") is None]
        f3 = fan.amplitude_vs_heat(
            df, no_heat_cols=no_heat_cols,
            heat_cols=[c for c in ("temp_charge1_mean", "temp_discharge1_mean")
                      if c in df.columns])
        print("\n[F3-구버전] 발열 유무 이분법 진폭 비교 (참고용, LCI 등 별도설비 제외):")
        _print_dict(f3)
        report_lines.append(f"### F3 진폭 vs 발열 — 이분법 참고용\n\n```\n{f3}\n```\n")

        rise_cols = []
        for prefix, n_max in (("charge", 7), ("discharge", 7)):
            for n in range(1, n_max + 1):
                key = f"{prefix}{n}"
                rise_cols.append((f"temp_{key}_mean", f"temp_{key}_min",
                                 f"temp_{key}_max", key))
        reg = fan.heat_vs_amplitude_regression(df, rise_cols)
        reg_summary = {k: v for k, v in reg.items() if k != "table"}
        print("\n[F3-판정/오류10] ★발열량 대 진폭 연속 회귀 (이분법 대체):")
        print(reg.get("table").to_string(index=False) if "table" in reg else "(없음)")
        _print_dict(reg_summary)
        report_lines.append(
            "### F3 판정 — 발열량 vs 진폭 회귀 (오류10)\n\n"
            + (reg["table"].to_markdown(index=False) if "table" in reg else "")
            + f"\n\n```\n{reg_summary}\n```\n")

        # 그림: 대표 recent_heat 단계 온도 필드 (평균 + 팬 오버레이)
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

    # --- E: 공간 지문 (발표 파트 I 핵심) ---------------------------------------
    print("\n=== [E] 공간 지문 · 분산 분해 ===")
    docv7_col = "docv7_raw"
    if docv7_col in df.columns:
        from .fields import tray_delta as _tray_delta
        _dev_tmp = df.copy()
        _dev_tmp["_dev"] = _tray_delta(_dev_tmp, docv7_col, stat="median")
        mean_field = position_field(_dev_tmp, "_dev", agg="mean")
        rms_field = position_field(_dev_tmp, "_dev", agg="rms")
        print("[E1] docv7 위치별 평균 필드 (일부):")
        print(mean_field.iloc[:4].to_string())
        print("[E1] docv7 위치별 RMS 필드 (부호무관 진폭, 일부):")
        print(rms_field.iloc[:4].to_string())
        report_lines.append(
            "## E. 공간 지문\n\n### E1 평균/RMS 필드\n\n**평균**\n\n"
            + mean_field.to_markdown() + "\n\n**RMS**\n\n" + rms_field.to_markdown() + "\n")

        qspread = spatial.quantile_spread_report(df, docv7_col)
        print("\n[E2] 위치별 분위수 확산 — 평행이동인가 꼬리 집중인가:")
        _print_dict(qspread)
        report_lines.append(f"### E2 분위수 확산\n\n```\n{qspread}\n```\n")

        vardecomp = spatial.variance_decomposition(df, docv7_col)
        print("\n[E8, 오류5 수정] ★분산 분해 — 고정보정 한계의 정식 수치:")
        _print_dict(vardecomp)
        report_lines.append(f"### E8 분산 분해 (오류5 수정)\n\n```\n{vardecomp}\n```\n")

        fig, ax = style.kind_fig(
            "전압 강하량(docv7)이 자리마다 다릅니다",
            "칸 하나가 셀 하나. 색은 같은 트레이 안 중앙값과의 차이(mV)입니다.",
            figsize=(9.2, 7.8), top=0.86)
        style.tray_heatmap(ax, mean_field.to_numpy(), unit="mV", show_fans=False)
        style.save(fig, os.path.join(fig_dir, "E1_docv7_지문.png"))
        print(f"\n[저장] {fig_dir}/E1_docv7_지문.png")

        # --- L: 패턴 유형·비율 (원인 불문 성립하는 문제점) ---------------------
        print("\n=== [L] 패턴 유형 · 비율 ===")
        pattern_df = patterns.classify_patterns(df, docv7_col)
        ratio = patterns.pattern_ratio(pattern_df)
        print("[L1·L2] 트레이별 패턴 유형 분류 + 비율:")
        _print_dict(ratio)
        report_lines.append(f"## L. 패턴 유형·비율\n\n### L1·L2 유형 비율\n\n```\n{ratio}\n```\n")

        gsize = patterns.pattern_gradient_size(df, docv7_col, pattern_df)
        print("\n[L3] 유형별 구배 크기(p2p, 판정기준 대비 %):")
        _print_dict(gsize)
        report_lines.append(f"### L3 유형별 구배 크기\n\n```\n{gsize}\n```\n")

        pfail = patterns.pattern_failure_rate(df, pattern_df)
        print("\n[L4] 유형별 불량 판정률:")
        _print_dict(pfail)
        report_lines.append(f"### L4 유형별 불량률\n\n```\n{pfail}\n```\n")

        flimit = patterns.fixed_correction_limit(vardecomp)
        print("\n[L5, 오류5 수정] 고정 보정의 한계 (정식 분산분해 기반):")
        _print_dict(flimit)
        report_lines.append(f"### L5 고정보정 한계\n\n```\n{flimit}\n```\n")

        splithalf = patterns.split_half_reproducibility(df, docv7_col)
        print("\n[L6] 랏 1개 재현성 — 무작위 절반 분할 비교:")
        _print_dict(splithalf)
        report_lines.append(f"### L6 split-half 재현성\n\n```\n{splithalf}\n```\n")

        # --- I: 판정 영향 (발표 클라이맥스) ------------------------------------
        print("\n=== [I] 판정 영향 정량화 ===")
        pfr = impact.position_failure_rate(df)
        print("[I2] ★위치별 불량 판정률 배수:")
        _print_dict({k: v for k, v in pfr.items() if k not in ("field", "ratio_field")})
        report_lines.append(
            "## I. 판정 영향\n\n### I2 위치별 불량 판정률\n\n```\n"
            + str({k: v for k, v in pfr.items() if k not in ("field", "ratio_field")})
            + "\n```\n")
        if pfr.get("ok"):
            fig, ax = style.kind_fig(
                f"어떤 자리는 평균보다 최대 {pfr['max_ratio']:.1f}배 더 자주 불량 판정됩니다",
                "칸 하나가 셀 위치 하나. 색은 트레이 평균 불량률 대비 배수입니다.",
                figsize=(9.2, 7.8), top=0.86)
            style.tray_heatmap(ax, pfr["ratio_field"].to_numpy(), unit="배",
                              show_fans=False, show_margin_axis=False,
                              cbar_label="평균 대비 불량 판정 배수")
            style.save(fig, os.path.join(fig_dir, "I2_위치별불량률.png"))
            print(f"\n[저장] {fig_dir}/I2_위치별불량률.png")

        spike = impact.spike_in_test(df)
        print("\n[I3] ★★유일한 비순환 성능지표 — 인공 결함 위치별 검출률:")
        _print_dict(spike)
        report_lines.append(f"### I3 spike-in 검출률\n\n```\n{spike}\n```\n")
    else:
        print("(docv7 컬럼이 없어 E/L/I 그룹을 건너뜁니다)")

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
