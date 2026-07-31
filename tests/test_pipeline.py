"""파이프라인 스모크 테스트 — 합성 데이터로 전 모듈이 끝까지 도는지 확인한다.

실제 검증이 아니라 "코드가 죽지 않고 계획된 산출물 모양이 나오는가"를 본다.
실행: python -m tests.test_pipeline
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from tray_ocv2 import fan, fields, figures, impact, indicators, io_load, patterns, qc, schema, spatial, timing
from tray_ocv2.io_load import _to_datetime
from tests.make_synthetic import make_export

# cli.py 와 동일한 이유(cp949 콘솔이 —,★ 등에서 죽음) — 실패 상세 메시지 출력 시 필요.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

FAILS = []


def check(name, cond, detail=""):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def check_korean_ampm_parsing():
    """2026-07-29 실사례: 'YYYY-MM-DD 오전/오후 H:MM:SS' 형식 파싱 회귀 방지."""
    s = pd.Series([
        "2026-07-08 오전 5:39:34", "2026-07-08 오전 5:37:24",
        "2026-01-01 오전 12:00:00",   # 자정 -> 00:00:00
        "2026-01-01 오후 12:00:00",   # 정오 -> 12:00:00
        "2026-01-01 오후 1:05:09",    # -> 13:05:09
        "2026-01-01 오후 11:59:59",   # -> 23:59:59
        None, "garbage",
    ])
    expected = [
        pd.Timestamp("2026-07-08 05:39:34"), pd.Timestamp("2026-07-08 05:37:24"),
        pd.Timestamp("2026-01-01 00:00:00"), pd.Timestamp("2026-01-01 12:00:00"),
        pd.Timestamp("2026-01-01 13:05:09"), pd.Timestamp("2026-01-01 23:59:59"),
        pd.NaT, pd.NaT,
    ]
    out = _to_datetime(s)
    ok = all((pd.isna(a) and pd.isna(b)) or a == b for a, b in zip(out, expected))
    check("한글 오전/오후 시각 파싱", ok, str(list(out)))

    already_dt = pd.Series(pd.to_datetime(["2026-07-08 05:39:34", "2026-07-08 06:00:00"]))
    out2 = _to_datetime(already_dt)
    check("이미 datetime64 인 컬럼 무변형 통과", (out2 == already_dt).all())


def check_disattenuated_corr():
    """[오류8] 신뢰도 감쇠보정 공식 직접 검증 (docs/04_logic_audit.md 오류8)."""
    from tray_ocv2.indicators import disattenuated_corr

    check("감쇠보정: 1.0 초과시 클립",
         abs(disattenuated_corr(0.5, 0.25, 0.25) - 1.0) < 1e-9)
    check("감쇠보정: 정상 케이스 0.3/0.5=0.6",
         abs(disattenuated_corr(0.3, 0.5, 0.5) - 0.6) < 1e-9)
    check("감쇠보정: 신뢰도 NaN이면 NaN", np.isnan(disattenuated_corr(0.3, np.nan, 0.5)))


def check_fan_col_predictions():
    """[신규 발견] 열 축 예측(FAN_COL_COLD_*)이 팬 중심 열을 가리키는지 확인.

    이전 버전은 팬 '사이(gap)' 열(D,G,J)을 손으로 잘못 계산해 '냉각 최강'으로
    라벨링했었다(행 축 FAN_ROW_MAX 의 '팬 중심=차가움' 규약과 불일치). 합성 데이터
    검증 중 발견해 schema._fan_center_columns() 로 계산식 기반으로 교체했다.
    """
    check("열 축 예측(상단): 팬 중심 열", schema.FAN_COL_COLD_OUTER == ["B", "E", "H", "L"],
         str(schema.FAN_COL_COLD_OUTER))
    check("열 축 예측(중앙): 팬 중심 열",
         schema.FAN_COL_COLD_MIDDLE == ["B", "D", "G", "I", "K"],
         str(schema.FAN_COL_COLD_MIDDLE))


def check_charger_from_box():
    """[2026-07-31] 충·방전 박스 문자열 → 호기 파싱 (§00_facts 2.3d/3.3)."""
    cases = {
        "CDC #MP1 -2-BOX #03-06 (1-3-6)": 2,
        "CDC #MP2 -1-BOX #06-07 (1-6-7)": 3,
        "CDC #MP2 -2-BOX #06-07 (1-6-7)": 4,
        "CDC #MP2 -3-BOX #06-07 (1-6-7)": 5,
        "#MP1-2": 2,                      # 공백 없는 변형
        "CDC #MP9 -9-BOX (미등록)": None,  # 미등록 토큰 → None
        "가짜값": None, None: None,
    }
    for raw, expected in cases.items():
        got = schema.charger_from_box(raw)
        check(f"박스→호기 파싱: {raw!r}→{expected}", got == expected, f"got={got}")

    # 연·단까지 뽑는 parse_box (예: 4호 4연 5단)
    info = schema.parse_box("CDC #MP2 -2-BOX #04-05 (1-4-5)")
    check("parse_box: 4호 4연 5단",
         info == {"charger": 4, "bay": 4, "tier": 5}, str(info))
    info2 = schema.parse_box("CDC #MP1 -2-BOX #03-06 (1-3-6)")
    check("parse_box: 2호 3연 6단",
         info2 == {"charger": 2, "bay": 3, "tier": 6}, str(info2))
    check("parse_box: 미등록 호기라도 연·단은 뽑음",
         schema.parse_box("CDC #MP9 -9-BOX #07-02") == {"charger": None, "bay": 7, "tier": 2})
    check("parse_box: 완전 무관 문자열 → None", schema.parse_box("가짜값") is None)


def main():
    check_korean_ampm_parsing()
    check_disattenuated_corr()
    check_fan_col_predictions()
    check_charger_from_box()

    raw = make_export(n_trays=30, seed=1)
    check("합성 데이터 생성", len(raw) == 30 * 144, f"len={len(raw)}")

    df = io_load.build_cell_table(raw)
    check("셀 테이블 구성", len(df) == len(raw))
    check("row/col 파싱", df["row"].notna().mean() > 0.95,
         f"결측률={df['row'].isna().mean():.3f}")

    a6 = schema.verify_position_consistency(raw)
    check("A6 위치 일관성", a6.get("ok") is True, str(a6))

    lsb = qc.lsb_estimate(df["t_ocv1"])
    check("A1 LSB 추정 실행", np.isfinite(lsb))

    miss = qc.missing_by_position(df, "t_ocv2")
    check("A2 결측 위치 맵", miss.shape == (12, 12))

    dup = qc.duplicate_cells(df)
    check("A5 중복 검사 실행", isinstance(dup, type(df)))

    a7 = qc.grade_consistency(df)
    check("A7 등급 구성 분해", a7.get("ok") is True, str(a7))
    if a7.get("ok"):
        check("A7 규칙 설명 비율 범위", 0 <= a7["pct_grade_explained_by_rule"] <= 100)

    a9 = qc.docv7_unit_check(df)
    check("A9 단위 대조", a9.get("ok") is True, str(a9))
    if a9.get("ok"):
        scale = a9["best_scale_candidate"]["scale"]
        check("A9 스케일 후보=1000 근접", abs(scale - 1000.0) < 1e-6, f"scale={scale}")

    clock = timing.clock_sanity(df)
    check("A8 시계 정합성 실행", len(clock) > 0)
    check("A8 음수 구간 없음", (clock["n_negative"] == 0).all(),
         clock[clock["n_negative"] > 0].to_string())

    order = timing.infer_process_order(df)
    check("K0 공정순서 추정", len(order) > 5)

    df = indicators.compute(df)
    check("S_A/S_B/S_C 계산", all(k in df.columns for k in ("S_A", "S_B", "S_C")))
    check("지표 시간(h) 계산", df["S_C_hours"].notna().mean() > 0.9,
         f"결측률={df['S_C_hours'].isna().mean():.3f}")
    check("지표 율(mV/day) 계산", df["S_C_rate"].notna().mean() > 0.9)

    df = indicators.tray_relative(df)
    check("트레이 편차 계산", "S_C_dev" in df.columns)

    rest = indicators.rest_time_report(df)
    check("K10 휴지시간 리포트", len(rest) >= 2)

    tri = indicators.triangulate(df)
    check("C1 상관행렬 계산", tri.shape[0] >= 2, str(tri))
    if tri.shape[0] >= 2 and "S_A_dev" in tri.index and "S_C_dev" in tri.index:
        r = tri.loc["S_A_dev", "S_C_dev"]
        check("C1 S_A~S_C 상관 존재(합성설계상 양의 상관 기대)", np.isfinite(r), f"r={r}")

    rel = indicators.reliability_report(df)
    check("오류8 신뢰도 리포트(S_A/S_B/S_C 전부)",
         all(k in rel for k in ("S_A", "S_B", "S_C")), str(rel))
    if len(tri):
        dis = indicators.disattenuated_matrix(tri, rel)
        check("오류8 감쇠보정 행렬 shape 일치", dis.shape == tri.shape)

    check("오류9 지표별 고온노출시간 컬럼", "S_A_hot_hours" in df.columns
         and "S_B_hot_hours" in df.columns)
    check("오류9 아레니우스 등가율 컬럼", "S_A_rate_arrhenius" in df.columns)

    tail = indicators.tail_consistency_check(df)
    check("오류11 꼬리 일관성 검사 실행", tail.get("ok") is True, str(tail))
    if tail.get("ok"):
        # 합성데이터는 핫셀 누설을 두 구간에 지속되게 설계 -> 꼬리가 양쪽에서 높아야 함
        check("오류11 꼬리가 양쪽 반분에서 높음 (합성설계 재현)",
             tail["frac_tail_high_in_both_halves"] > tail["expected_if_random"] * 10,
             str(tail))

    tail_rate = indicators.tail_rate_check(df)
    check("꼬리 가속열화 판별 실행", tail_rate.get("ok") is True, str(tail_rate))
    if tail_rate.get("ok"):
        # 합성데이터는 RT5=RT6=24h, half1≈half2 로 설계 -> 비율이 1 근처여야 함
        check("꼬리 율 재계산: 합성설계(대칭)에서 비율 1 근처",
             0.5 < tail_rate["rate_ratio_2_over_1"] < 2.0, str(tail_rate))

    c2 = indicators.cross_check_grade(df)
    check("C2 준독립 대조 실행", c2.get("ok") is True, str(c2))

    temp_cols = [c for c in df.columns if c.startswith("t_ocv") or c.startswith("temp_")]
    check("온도 컬럼 존재", len(temp_cols) > 5, f"n={len(temp_cols)}")

    f0 = fan.stage_field_reproducibility(df, temp_cols)
    check("F0 재현성 행렬 계산", f0.shape[0] >= 2)
    if len(f0):
        cluster = fan.cluster_reproducibility(f0)
        check("오류7 군집별 재현성 계산", "within_by_category" in cluster, str(cluster))

    f1 = fan.row_profile(df, "t_ocv2")
    check("F1 세로 프로파일 12행", len(f1) == 12, f"len={len(f1)}")
    if len(f1):
        max_rows_mean = f1.loc[f1["is_fan_axis_max"], "mean"].mean()
        min_rows_mean = f1.loc[f1["is_fan_axis_min"], "mean"].mean()
        check("F1 팬축 온도 < 팬사이 온도 (합성설계 재현)",
             max_rows_mean < min_rows_mean,
             f"팬축={max_rows_mean:.3f} 팬사이={min_rows_mean:.3f}")

    profiles = fan.band_column_profile(df, "t_ocv2", detrend=True)
    check("F2 밴드 프로파일 3개(detrend)", len(profiles) == 3, f"밴드={list(profiles)}")
    sim = fan.band_similarity(profiles)
    check("F2 유사도 계산", sim.get("ok") is True, str(sim))

    minima = fan.predicted_minima_check(profiles)
    band_results = {k: v for k, v in minima.items() if k != "note"}
    check("오류6 예측열 검정 실행", len(band_results) == 3, str(minima))
    check("오류6 예측열 검정: 합성설계 전 밴드에서 팬중심 열이 실제로 더 차가움",
         all(v["colder_as_predicted"] for v in band_results.values()), str(minima))

    f3 = fan.amplitude_vs_heat(
        df, no_heat_cols=["t_ocv1"], heat_cols=["temp_charge1_mean"])
    check("F3-구버전 진폭 비교 실행", np.isfinite(f3.get("ratio_heat_over_no_heat", np.nan)),
         str(f3))

    rise_cols = [(f"temp_charge{n}_mean", f"temp_charge{n}_min", f"temp_charge{n}_max",
                 f"charge{n}") for n in range(1, 8)]
    reg = fan.heat_vs_amplitude_regression(df, rise_cols)
    check("오류10 발열량-진폭 회귀 실행", reg.get("ok") is True, str(reg))

    radial = fan.fan_radial_profile(df, "t_ocv2")
    check("F4 반경 프로파일 실행", radial.get("ok") is True, str(radial))
    if radial.get("ok"):
        # 합성데이터는 팬 중심에서 지수감쇠(단조)로 설계 -> 전반적 추세가 뚜렷해야 함.
        # is_monotonic(구간별 부호 전부 일치)은 끝단 잡음 하나에도 뒤집히므로
        # (실제로 발견된 문제 — 아래 참조) trend_corr(거리-값 선형상관)로 확인한다.
        check("F4 반경-값 추세 뚜렷함 (합성설계 재현)",
             abs(radial["trend_corr"]) > 0.8, str(radial))
        check("F4 중심이 가장 차가움 (합성설계 재현)",
             radial["center_mean"] < radial["extremum_mean"], str(radial))

    aniso = fan.fan_radial_profile_anisotropic(df, "t_ocv2")
    check("F4 이방성 분리 실행", aniso.get("ok") is True, str(aniso))
    if aniso.get("ok"):
        # 합성데이터는 등방(단일 로브) 설계 -> 행/열 축 둘 다 추세가 뚜렷해야 함.
        # (세로축 is_monotonic 이 끝단 구간 하나의 잡음만으로 False 가 되는 것을
        #  실제로 발견 -> trend_corr 로 판정하도록 보강. docs/04_logic_audit.md 참조)
        check("F4 이방성: 세로축 추세 뚜렷함 (합성설계 재현)",
             abs(aniso["row_axis_summary"].get("trend_corr", 0)) > 0.7, str(aniso))
        check("F4 이방성: 가로축 추세 뚜렷함 (합성설계 재현)",
             abs(aniso["col_axis_summary"].get("trend_corr", 0)) > 0.7, str(aniso))

    k3 = timing.aging_duration_bias_check(df)
    check("K3 시간편향 검사 실행 (표본부족이어도 죽지 않아야)",
         "ok" in k3, str(k3))

    grid = fields.to_grid(df[df["tray_id"] == df["tray_id"].iloc[0]], "t_ocv2")
    check("격자 변환 shape", grid.shape == (12, 12))
    ring = fields.ring_score(grid)
    check("ring_score 계산", np.isfinite(ring))

    field = fields.position_field(df, "t_ocv2", agg="mean")
    check("position_field shape", field.shape == (12, 12))

    # --- mode 전환 인프라 (원칙 0.6) ---
    fd_bin = fields.freedman_diaconis_bin(df["docv7_raw"])
    check("Freedman-Diaconis bin 폭 계산", np.isfinite(fd_bin) and fd_bin > 0, f"bin={fd_bin}")

    dev_mode = fields.tray_delta(df, "docv7_raw", stat="mode")
    check("tray_delta(mode, 자동bin) 실행", dev_mode.notna().mean() > 0.9)
    dev_mode_fixed = fields.tray_delta(df, "docv7_raw", stat="mode",
                                       mode_bin_mv=schema.JUDGE_MODE_BIN_MV)
    check("tray_delta(mode, 고정bin) 실행", dev_mode_fixed.notna().mean() > 0.9)

    sens = fields.sensitivity_median_vs_mode(df, "docv7_raw")
    check("median vs mode 민감도 검증 shape",
         list(sens.columns[:1]) == ["tray_id"] and len(sens) == df["tray_id"].nunique(),
         f"len={len(sens)}")
    check("민감도 검증 abs_diff 내림차순 정렬",
         sens["abs_diff"].is_monotonic_decreasing, str(sens.head(3)))

    # --- E: 공간 지문 / 분산 분해 ---
    docv7_col = "docv7_raw"
    qspread = spatial.quantile_spread_report(df, docv7_col)
    check("E2 분위수 확산 실행", qspread.get("ok") is True, str(qspread))

    vardecomp = spatial.variance_decomposition(df, docv7_col)
    check("E8 분산분해 실행", vardecomp.get("ok") is True, str(vardecomp))
    if vardecomp.get("ok"):
        total_frac = (vardecomp["frac_position_fixed"]
                     + vardecomp["frac_tray_position_eof"]
                     + vardecomp["frac_cell_residual"])
        check("E8 분산분해 성분 합=1.0", abs(total_frac - 1.0) < 1e-6, f"합={total_frac}")
        check("E8 각 성분 0~1 범위",
             all(0 <= vardecomp[k] <= 1 for k in
                 ("frac_position_fixed", "frac_tray_position_eof", "frac_cell_residual")),
             str(vardecomp))

    # --- L: 패턴 유형·비율 ---
    pattern_df = patterns.classify_patterns(df, docv7_col)
    check("L1 패턴 분류 실행", not pattern_df.empty, f"len={len(pattern_df)}")

    ratio = patterns.pattern_ratio(pattern_df)
    check("L2 유형 비율 실행", ratio.get("ok") is True, str(ratio))
    if ratio.get("ok"):
        check("L2 비율 합=1.0", abs(sum(ratio["ratio"].values()) - 1.0) < 1e-6, str(ratio))

    gsize = patterns.pattern_gradient_size(df, docv7_col, pattern_df)
    check("L3 유형별 구배크기 실행", gsize.get("ok") is True, str(gsize))

    pfail = patterns.pattern_failure_rate(df, pattern_df)
    check("L4 유형별 불량률 실행", pfail.get("ok") is True, str(pfail))

    flimit = patterns.fixed_correction_limit(vardecomp)
    check("L5 고정보정한계 실행", flimit.get("ok") is True, str(flimit))
    if flimit.get("ok") and vardecomp.get("ok"):
        check("L5 값이 E8 과 일치",
             abs(flimit["frac_position_fixed"] - vardecomp["frac_position_fixed"]) < 1e-9)

    splithalf = patterns.split_half_reproducibility(df, docv7_col)
    check("L6 split-half 실행", splithalf.get("ok") is True, str(splithalf))
    if splithalf.get("ok"):
        check("L6 상관계수 -1~1 범위", -1 <= splithalf["split_half_corr"] <= 1)

    # --- I: 판정 영향 ---
    pfr = impact.position_failure_rate(df)
    check("I2 위치별 불량률 실행", pfr.get("ok") is True, str(pfr))
    if pfr.get("ok"):
        # 합성데이터 핫셀 위치: (row=4,col=7=G)->G4, (row=9,col=2=B)->B9
        check("I2 최대배수 위치가 합성 핫셀 위치와 일치",
             pfr["max_ratio_position"] in ("G4", "B9"), str(pfr))
        check("I2 최대배수 > 1 (평균보다 더 자주 걸림)", pfr["max_ratio"] > 1, str(pfr))

    spike = impact.spike_in_test(df, magnitudes_mv=(0.5, 0.8, 1.5), seed=7)
    check("I3 spike-in 실행", spike.get("ok") is True, str(spike))
    if spike.get("ok"):
        rates = [r["overall_detection_rate"] for r in spike["results"]]
        check("I3 검출률이 주입크기에 따라 증가(비감소)",
             all(a <= b + 1e-9 for a, b in zip(rates, rates[1:])), str(rates))
        check("I3 최대 주입크기(1.5mV)에서 검출률 높음", rates[-1] > 0.8, str(rates))

    # --- S1: 전체 지문 갤러리 (4장) ---
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = figures.fig_s1_all(df, tmpdir)
        check("S1 4장 전부 생성", len(paths) == 4, str(paths))
        check("S1 파일 실제로 저장됨",
             all(os.path.exists(p) for p in paths.values()), str(paths))

    # --- V1: E8 분산분해 그림 ---
    with tempfile.TemporaryDirectory() as tmpdir:
        v1_paths = figures.fig_v1_all(df, tmpdir)
        check("V1 2장 전부 생성", len(v1_paths) == 2, str(v1_paths))
        check("V1 파일 실제로 저장됨",
             all(os.path.exists(p) for p in v1_paths.values()), str(v1_paths))

    # --- V3: I3 spike-in 그림 ---
    with tempfile.TemporaryDirectory() as tmpdir:
        v3_paths = figures.fig_v3_all(df, tmpdir)
        check("V3 3장 전부 생성", len(v3_paths) == 3, str(v3_paths))
        check("V3 파일 실제로 저장됨",
             all(os.path.exists(p) for p in v3_paths.values()), str(v3_paths))

    # --- V2: L6 재현성 + null 대조 ---
    with tempfile.TemporaryDirectory() as tmpdir:
        v2_path = figures.fig_v2_split_half(df, "docv7_raw", tmpdir)
        check("V2 파일 실제로 저장됨", os.path.exists(v2_path), v2_path)

    # --- V14/V15: 패턴유형·고정보정한계 ---
    with tempfile.TemporaryDirectory() as tmpdir:
        v1415_paths = figures.fig_v14_v15_all(df, tmpdir)
        check("V14/V15 2장 전부 생성", len(v1415_paths) == 2, str(v1415_paths))
        check("V14/V15 파일 실제로 저장됨",
             all(os.path.exists(p) for p in v1415_paths.values()), str(v1415_paths))

    # --- V4/S3: 온도 물리적 배경 ---
    with tempfile.TemporaryDirectory() as tmpdir:
        v4_paths = figures.fig_v4_all(df, tmpdir)
        check("V4/S3 그림 생성(v4a/v4b/s3)",
             all(k in v4_paths for k in ("v4a", "v4b", "s3")), str(v4_paths))
        check("V4/S3 파일 실제로 저장됨",
             all(os.path.exists(p) for p in v4_paths.values()), str(v4_paths))

    # --- V5/V6/V7: 영향없음/미확정 ---
    with tempfile.TemporaryDirectory() as tmpdir:
        v567_paths = figures.fig_v567_all(df, tmpdir)
        check("V6/V7 최소 생성", all(k in v567_paths for k in ("v6", "v7")), str(v567_paths))
        check("V5/V6/V7 파일 실제로 저장됨",
             all(os.path.exists(p) for p in v567_paths.values()), str(v567_paths))

    # --- V8/V9/V10: 데이터 신뢰성 ---
    with tempfile.TemporaryDirectory() as tmpdir:
        v8910_paths = figures.fig_v8_v9_v10_all(df, tmpdir)
        check("V8/V10 최소 생성", all(k in v8910_paths for k in ("v8", "v10")),
             str(v8910_paths))
        check("V8/V9/V10 파일 실제로 저장됨",
             all(os.path.exists(p) for p in v8910_paths.values()), str(v8910_paths))

    # --- V11: 공정 타임라인 ---
    with tempfile.TemporaryDirectory() as tmpdir:
        v11_path = figures.fig_v11_process_timeline(df, tmpdir)
        check("V11 파일 실제로 저장됨", os.path.exists(v11_path), v11_path)

    # --- V12/V13: 자기방전 지표 ---
    with tempfile.TemporaryDirectory() as tmpdir:
        v1213_paths = figures.fig_v12_v13_all(df, tmpdir)
        check("V12/V13 2장 전부 생성", len(v1213_paths) == 2, str(v1213_paths))
        check("V12/V13 파일 실제로 저장됨",
             all(os.path.exists(p) for p in v1213_paths.values()), str(v1213_paths))

    # --- V16/S4 ---
    with tempfile.TemporaryDirectory() as tmpdir:
        v16s4_paths = figures.fig_v16_s4_all(df, tmpdir)
        check("V16/S4 2장 전부 생성", len(v16s4_paths) == 2, str(v16s4_paths))
        check("V16/S4 파일 실제로 저장됨",
             all(os.path.exists(p) for p in v16s4_paths.values()), str(v16s4_paths))

    # --- V0: 종합 대시보드 ---
    with tempfile.TemporaryDirectory() as tmpdir:
        v0_path = figures.fig_v0_dashboard(df, tmpdir)
        check("V0 파일 실제로 저장됨", os.path.exists(v0_path), v0_path)

    print(f"\n{'='*40}")
    if FAILS:
        print(f"FAIL {len(FAILS)}건: {FAILS}")
        sys.exit(1)
    print("모든 스모크 테스트 통과")


if __name__ == "__main__":
    main()
