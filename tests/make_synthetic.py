"""합성 Export 데이터 생성 — 실 데이터 없이 파이프라인을 끝까지 검증하기 위함.

schema.py 의 실제 컬럼명을 그대로 채워, 팬 서명(밴드별 다른 열 주기) +
자기방전(트레이별 mode) + 결측 + 고립 핫셀을 심어둔다. 검증 결과가 계획대로
나오는지 보는 것이 목적이지, 실제 공정을 정확히 흉내내는 것이 목적이 아니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tray_ocv2 import schema

N_TRAYS = 40
N_ROWS, N_COLS = 12, 12
BASE_TIME = pd.Timestamp("2026-01-01 08:00:00")


def _fan_temp_offset(row: int, col: int) -> float:
    """팬 기하를 반영한 합성 온도 편차(°C). row=세로(1~12), col=가로(A~L=1~12)."""
    band = next(b for b in schema.FAN_BANDS if b["rows"][0] <= row <= b["rows"][1])
    row_c, col_cs = band["row_center"], band["col_centers"]
    d_row = abs(row - row_c)
    d_col = min(abs(col - c) for c in col_cs)
    dist = np.sqrt(d_row ** 2 + d_col ** 2)
    return -0.35 * np.exp(-dist / 1.8)   # 팬 축 가까울수록 차가움(음수)


def make_export(n_trays: int = N_TRAYS, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    t_cursor = BASE_TIME

    for ti in range(n_trays):
        tray_id = f"TRAY{ti:04d}"
        tray_sign = 1.0 if ti % 3 else -1.0          # 역링 트레이 일부 섞기
        tray_selfdis = rng.normal(1.3, 0.15)          # mV, 트레이 mode 될 값
        t0 = t_cursor

        for n in range(1, N_ROWS * N_COLS + 1):
            col = (n - 1) // 12 + 1   # A~L (1~12)
            row = (n - 1) % 12 + 1    # 1~12
            col_letter = chr(ord("A") + col - 1)
            cell_pos = f"{col_letter}{row:02d}"

            fan_dev = _fan_temp_offset(row, col)
            noise = rng.normal(0, 0.05)
            grade = "A"
            is_hotcell = (ti == 5 and row == 4 and col == 7) or (ti == 12 and row == 9 and col == 2)

            r = {
                schema.ID_COLS["route"]: "RT01",
                schema.ID_COLS["product_lot"]: "PRODX N99S",
                schema.ID_COLS["tray_id"]: tray_id,
                schema.ID_COLS["cell_id"]: f"C{ti:04d}{n:04d}",
                schema.ID_COLS["can_id"]: f"CAN{ti % 5}",
                schema.ID_COLS["vent_id"]: f"VENT{ti % 3}",
                schema.ID_COLS["cell_no"]: n,
                schema.ID_COLS["cell_pos"]: cell_pos,
            }

            # --- LCI (무발열) ---
            for k in (1, 2):
                p = f"Low Current Inspection #{k:02d}"
                r[f"{p} 시작시간"] = t0 + pd.Timedelta(minutes=5 * k)
                r[f"{p} 종료시간"] = r[f"{p} 시작시간"] + pd.Timedelta(minutes=3)
                r[f"{p} 작업시간"] = 3.0
                r[f"{p} 온도"] = 25.0 + noise * 0.3

            # --- Charge/DisCharge (발열 있음, 팬 온도장 직접 반영) ---
            step_t = t0 + pd.Timedelta(hours=1)
            for prefix, count in (("Charge", 7), ("DisCharge", 7)):
                for k in range(1, count + 1):
                    p = f"{prefix} #{k:02d}"
                    dur_h = rng.uniform(0.8, 1.5)
                    r[f"{p} 시작시간"] = step_t
                    r[f"{p} 종료시간"] = step_t + pd.Timedelta(hours=dur_h)
                    r[f"{p} 작업시간"] = dur_h * 60  # 분 단위로 저장(현실적 단위 혼용 재현)
                    tmin = 25.0 + fan_dev * 0.6 + rng.normal(0, 0.1)
                    tmax = tmin + rng.uniform(0.8, 2.0)
                    r[f"{p} 최저 온도"] = tmin
                    r[f"{p} 최고 온도"] = tmax
                    r[f"{p} 평균온도"] = (tmin + tmax) / 2
                    if prefix == "Charge" and k in (1, 2):
                        r[f"End Voltage_{prefix} #{k:02d}"] = 4.2 + rng.normal(0, 0.001)
                    if prefix == "DisCharge" and k in (1, 2, 3):
                        r[f"End Voltage_{prefix} #{k:02d}"] = 3.0 + rng.normal(0, 0.001)
                    step_t = r[f"{p} 종료시간"] + pd.Timedelta(minutes=10)

            # --- OCV / 고온에이징 / 상온에이징 (공정순서대로 시간 진행) ---
            def add_ocv(n_ocv, soc, extra_mv=0.0, rest_hours=0.5):
                nonlocal step_t
                p = f"OCV #{n_ocv:02d}"
                r[f"{p} 시작시간"] = step_t + pd.Timedelta(hours=rest_hours)
                r[f"{p} 종료시간"] = r[f"{p} 시작시간"] + pd.Timedelta(minutes=2)
                r[f"{p} 작업시간"] = 2.0
                r[f"{p}_온도"] = 25.0 + fan_dev + noise
                r[f"{p} OCV"] = 3.5 + soc * 0.003 + tray_sign * fan_dev * 0.01 + extra_mv / 1000.0
                step_t = r[f"{p} 종료시간"]

            add_ocv(1, 0.0, rest_hours=0.2)
            add_ocv(2, 10.0, rest_hours=0.3)   # 충전 직후 → 휴지 짧음(오류1 재현)

            r["Start Time_High Temp Aging #01"] = step_t + pd.Timedelta(hours=0.5)
            r["End Time_High Temp Aging #01"] = r["Start Time_High Temp Aging #01"] + pd.Timedelta(hours=48)
            r["Work BOX_High Temp Aging #01"] = f"BOX{ti % 4}"
            step_t = r["End Time_High Temp Aging #01"]
            r["Start Time_Room Temp Aging #02"] = step_t
            r["End Time_Room Temp Aging #02"] = step_t + pd.Timedelta(hours=24)
            step_t = r["End Time_Room Temp Aging #02"]

            selfdis_a = tray_selfdis + rng.normal(0, 0.08) + (0.9 if is_hotcell else 0.0)
            add_ocv(3, 10.0, extra_mv=-selfdis_a, rest_hours=0.1)

            add_ocv(4, 70.0, rest_hours=0.3)
            r["Start Time_High Temp Aging #02"] = step_t + pd.Timedelta(hours=0.5)
            r["End Time_High Temp Aging #02"] = r["Start Time_High Temp Aging #02"] + pd.Timedelta(hours=48)
            r["Work BOX_High Temp Aging #02"] = f"BOX{ti % 4}"
            step_t = r["End Time_High Temp Aging #02"]
            r["Start Time_Room Temp Aging #03"] = step_t
            r["End Time_Room Temp Aging #03"] = step_t + pd.Timedelta(hours=24)
            step_t = r["End Time_Room Temp Aging #03"]

            selfdis_b = tray_selfdis * 0.8 + rng.normal(0, 0.08)
            add_ocv(5, 70.0, extra_mv=-selfdis_b, rest_hours=0.1)
            add_ocv(6, 100.0, rest_hours=0.3)
            add_ocv(7, 30.0, rest_hours=0.3)

            for k, label in ((1, "RT #01"), (4, "RT #04")):
                pass  # RT#01,04~06 자리표시(합성 미사용) — 실데이터엔 존재

            r["Start Time_Room Temp Aging #04"] = step_t
            r["End Time_Room Temp Aging #04"] = step_t + pd.Timedelta(hours=24)
            step_t = r["End Time_Room Temp Aging #04"]

            # --- PRVT (별도 계측기, 온도 영향 없음 가정) ---
            def add_prvt(n_p, extra_mv=0.0, rest_hours=0.2):
                nonlocal step_t
                p = f"PRIVT OCV #{n_p:02d}"
                r[f"{p} 시작시간"] = step_t + pd.Timedelta(hours=rest_hours)
                r[f"{p} 온도"] = 25.0 + rng.normal(0, 0.15)
                r[f"{p} OCV"] = 3.500 + extra_mv / 1000.0
                step_t = r[f"{p} 시작시간"] + pd.Timedelta(minutes=2)

            selfdis_c = tray_selfdis + rng.normal(0, 0.06) + (1.1 if is_hotcell else 0.0)
            add_prvt(1, extra_mv=0.0)
            r["Start Time_Room Temp Aging #05"] = step_t
            r["End Time_Room Temp Aging #05"] = step_t + pd.Timedelta(hours=24)
            step_t = r["End Time_Room Temp Aging #05"]
            add_prvt(2, extra_mv=-selfdis_c / 2)
            r["Start Time_Room Temp Aging #06"] = step_t
            r["End Time_Room Temp Aging #06"] = step_t + pd.Timedelta(hours=24)
            step_t = r["End Time_Room Temp Aging #06"]
            add_prvt(3, extra_mv=-selfdis_c)

            docv7 = (r["PRIVT OCV #01 OCV"] - r["PRIVT OCV #03 OCV"]) * 1000.0  # mV
            r["Delta OCV #07 DOCV"] = docv7
            if is_hotcell or docv7 > 1.3 + schema.JUDGE_OFFSET_MV:
                grade = "E"
            r[schema.ID_COLS["grade"]] = grade
            r["샘플링 이전 등급"] = grade

            # 결측 셀 몇 개 심기
            if (ti, row, col) in [(3, 2, 10), (7, 7, 3), (20, 11, 6)]:
                r["OCV #02_온도"] = np.nan
                r["Charge #01 최저 온도"] = np.nan

            rows.append(r)

        t_cursor = t0 + pd.Timedelta(hours=6)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = make_export(n_trays=8)
    print(df.shape)
    print(df.head(3))
