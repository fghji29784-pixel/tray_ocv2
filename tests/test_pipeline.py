"""파이프라인 스모크 테스트 — 합성 데이터로 전 모듈이 끝까지 도는지 확인한다.

실제 검증이 아니라 "코드가 죽지 않고 계획된 산출물 모양이 나오는가"를 본다.
실행: python -m tests.test_pipeline
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from tray_ocv2 import fan, fields, indicators, io_load, qc, schema, timing
from tray_ocv2.io_load import _to_datetime
from tests.make_synthetic import make_export

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


def main():
    check_korean_ampm_parsing()
    check_disattenuated_corr()
    check_fan_col_predictions()

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

    grid = fields.to_grid(df[df["tray_id"] == df["tray_id"].iloc[0]], "t_ocv2")
    check("격자 변환 shape", grid.shape == (12, 12))
    ring = fields.ring_score(grid)
    check("ring_score 계산", np.isfinite(ring))

    field = fields.position_field(df, "t_ocv2", agg="mean")
    check("position_field shape", field.shape == (12, 12))

    print(f"\n{'='*40}")
    if FAILS:
        print(f"FAIL {len(FAILS)}건: {FAILS}")
        sys.exit(1)
    print("모든 스모크 테스트 통과")


if __name__ == "__main__":
    main()
