"""파이프라인 스모크 테스트 — 합성 데이터로 전 모듈이 끝까지 도는지 확인한다.

실제 검증이 아니라 "코드가 죽지 않고 계획된 산출물 모양이 나오는가"를 본다.
실행: python -m tests.test_pipeline
"""
from __future__ import annotations

import sys

import numpy as np

from tray_ocv2 import fan, fields, indicators, io_load, qc, schema, timing
from tests.make_synthetic import make_export

FAILS = []


def check(name, cond, detail=""):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def main():
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

    c2 = indicators.cross_check_grade(df)
    check("C2 준독립 대조 실행", c2.get("ok") is True, str(c2))

    temp_cols = [c for c in df.columns if c.startswith("t_ocv") or c.startswith("temp_")]
    check("온도 컬럼 존재", len(temp_cols) > 5, f"n={len(temp_cols)}")

    f0 = fan.stage_field_reproducibility(df, temp_cols)
    check("F0 재현성 행렬 계산", f0.shape[0] >= 2)

    f1 = fan.row_profile(df, "t_ocv2")
    check("F1 세로 프로파일 12행", len(f1) == 12, f"len={len(f1)}")
    if len(f1):
        max_rows_mean = f1.loc[f1["is_fan_axis_max"], "mean"].mean()
        min_rows_mean = f1.loc[f1["is_fan_axis_min"], "mean"].mean()
        check("F1 팬축 온도 < 팬사이 온도 (합성설계 재현)",
             max_rows_mean < min_rows_mean,
             f"팬축={max_rows_mean:.3f} 팬사이={min_rows_mean:.3f}")

    profiles = fan.band_column_profile(df, "t_ocv2")
    check("F2 밴드 프로파일 3개", len(profiles) == 3, f"밴드={list(profiles)}")
    sim = fan.band_similarity(profiles)
    check("F2 유사도 계산", sim.get("ok") is True, str(sim))

    f3 = fan.amplitude_vs_heat(
        df, no_heat_cols=["t_ocv1"], heat_cols=["temp_charge1_mean"])
    check("F3 진폭 비교 실행", np.isfinite(f3.get("ratio_heat_over_no_heat", np.nan)),
         str(f3))

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
