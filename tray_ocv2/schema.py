"""데이터 스키마 — 실제 Export 컬럼명을 도메인 개념에 매핑한다.

docs/00_facts.md 의 사실을 코드로 옮긴 것. 컬럼명이 바뀌면 여기만 고친다.

핵심 개념
  - STAGES     : 전압을 측정하는 단계 (OCV #01~#07, PRIVT OCV #01~#03) + SOC + 순서
  - THERM_STEPS: 온도를 갖는 공정 스텝 (LCI, Charge, DisCharge, OCV, PRIVT)
  - AGING      : 에이징 구간 (HT #01~#02, RT #01~#06). RT #07 이후는 사용하지 않는다.
  - INDICATORS : 자기방전 3지표 S_A / S_B / S_C
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 식별 · 위치
# ---------------------------------------------------------------------------
ID_COLS = {
    "route": "공정경로",
    "product": "Product",
    "lot": "Lot",
    "tray_id": "Tray ID",
    "cell_id": "Cell ID",       # 식별용. 제조 순번과 무관 → 순서 분석에 쓰지 않는다.
    "can_id": "Can ID",
    "vent_id": "Vent ID",
    "grade": "등급",             # A=양품 / E=불량. ★현행 판정 로직 산물 → 순환 라벨
    "cell_no": "Cell No",        # 1~144
    "cell_pos": "Cell 위치",     # A01~L12
}
#: 사용하지 않는 컬럼 (사용자 지시)
IGNORED_COLS = ["샘플링 이전 등급"]

N_ROWS, N_COLS = 12, 12
CELL_POS_RE = re.compile(r"^\s*([A-La-l])\s*(\d{1,2})\s*$")

GRADE_GOOD, GRADE_FAIL = "A", "E"

# ---------------------------------------------------------------------------
# 좌표계 방향 — ✅ 확정 (2026-07, 설비 도면 주석으로 확인)
# ---------------------------------------------------------------------------
#            A  B  C  D  E  F  G  H  I  J  K  L      ← 알파벳 = 가로 = 열(column)
#         1 ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
#         2 │   ○     ○     ○     ○   │  ← 팬 상단 밴드 4개
#         3 │                          │
#         4 ├──────────────────────────┤
#         5 │  ○   ○   ○   ○   ○       │  ← 팬 중앙 밴드 5개
#   숫자  6 │                          │
#   =세로 7 │                          │
#   =행   8 ├──────────────────────────┤
#         9 │   ○     ○     ○     ○   │  ← 팬 하단 밴드 4개
#        10 │                          │
#        11 │                          │
#        12 └──────────────────────────┘
#
# → 알파벳 A~L = **열(가로)**, 숫자 1~12 = **행(세로)**
# → 팬 3밴드는 **숫자 축(세로)을 따라 쌓여 있고**, 밴드 내 4-5-4 는 **알파벳 축(가로)**
#
# Cell No 규칙: n → col=(n-1)//12+1 (A~L), row=(n-1)%12+1 (1~12)
#   즉 A01→A02→…→A12→B01… 순으로 **세로로 먼저** 증가한다. (A6 에서 실데이터 검증)
#
# ⚠️ 주의: 12×12 정사각이라 축을 뒤집어도 에러가 나지 않는다. 전치된 채 조용히 계산되고,
#          팬 배열이 행/열 비대칭(3밴드 vs 4-5-4)이라 F1·F2 결론이 정반대가 된다.
#          ※ tray_ocv(구 저장소) 문서에는 '알파벳=행'으로 잘못 적혀 있으니 그대로 믿지 말 것.
ALPHABET_AXIS = "col"   # ✅ 확정

#: 팬 3밴드가 쌓인 축 = 숫자(1~12) = row
FAN_BAND_AXIS = "row"

#: 팬 기하 (셀 좌표. row=숫자 1~12 세로, col=알파벳 A~L 가로)
#: 도면 픽셀 판독값 — CAD 검증 전. 직경은 미확정.
FAN_BANDS = [
    {"name": "상단", "row_center": 2.6, "col_centers": [1.9, 5.2, 8.5, 11.5],
     "rows": (1, 4), "n_fans": 4, "col_pitch": 3.2},
    {"name": "중앙", "row_center": 6.4, "col_centers": [2.0, 4.3, 6.6, 8.9, 11.2],
     "rows": (5, 8), "n_fans": 5, "col_pitch": 2.3},
    {"name": "하단", "row_center": 10.2, "col_centers": [1.9, 5.1, 8.4, 11.4],
     "rows": (9, 12), "n_fans": 4, "col_pitch": 3.2},
]

#: [F1] 예측 — 숫자 축(세로) 프로파일. 직경 무관.
FAN_ROW_MAX = [3, 6, 10]        # 팬 축 아래 = 냉각 최강 = 차가움
FAN_ROW_MIN = [1, 4, 5, 8, 12]  # 팬 사이(4.5·8.3) + 배열 바깥(1·12) = 냉각 최약

#: [F2] 예측 — 알파벳 축(가로) 프로파일. **밴드마다 주기가 다른 것이 판별점.**
#:   상단·하단 밴드(주기 3.2): 극소 ≈ C/D, G, J
#:   중앙 밴드(주기 2.3)     : 극소 ≈ C, E, H, J   ← 상단·하단과 달라야 팬 확정
FAN_COL_MIN_OUTER = ["D", "G", "J"]
FAN_COL_MIN_MIDDLE = ["C", "E", "H", "J"]


def parse_cell_pos(s: str) -> tuple[int, int] | None:
    """'A01' → (row, col). ALPHABET_AXIS 설정에 따라 해석이 바뀐다. 실패 시 None."""
    m = CELL_POS_RE.match(str(s))
    if not m:
        return None
    alpha = ord(m.group(1).upper()) - ord("A") + 1
    num = int(m.group(2))
    row, col = (alpha, num) if ALPHABET_AXIS == "row" else (num, alpha)
    if 1 <= row <= N_ROWS and 1 <= col <= N_COLS:
        return row, col
    return None


def parse_cell_no(n) -> tuple[int, int] | None:
    """Cell No 1~144 → (row, col).

    ⚠️ 이 매핑은 `Cell 위치` 와의 실제 대응을 데이터에서 검증한 뒤 확정한다(A6).
    현재는 '12개마다 알파벳이 바뀐다'는 가정을 쓴다.
    """
    try:
        n = int(n)
    except (TypeError, ValueError):
        return None
    if not 1 <= n <= N_ROWS * N_COLS:
        return None
    alpha, num = (n - 1) // 12 + 1, (n - 1) % 12 + 1
    row, col = (alpha, num) if ALPHABET_AXIS == "row" else (num, alpha)
    return row, col


def verify_position_consistency(df) -> dict:
    """[A6] `Cell No` 와 `Cell 위치` 가 같은 좌표를 가리키는지 검증.

    둘이 어긋나면 parse_cell_no 의 가정이 틀린 것이다. 좌표계 확정의 1단계.
    """
    import pandas as pd  # 지연 import (schema 는 의존성 최소화)

    pos_col, no_col = ID_COLS["cell_pos"], ID_COLS["cell_no"]
    if pos_col not in df.columns or no_col not in df.columns:
        return {"ok": False, "reason": "위치 컬럼 없음"}
    a = df[pos_col].map(parse_cell_pos)
    b = df[no_col].map(parse_cell_no)
    both = a.notna() & b.notna()
    if not both.any():
        return {"ok": False, "reason": "파싱 가능한 행 없음"}
    match = (a[both] == b[both])
    return {
        "ok": bool(match.all()),
        "n_checked": int(both.sum()),
        "n_mismatch": int((~match).sum()),
        "example": None if match.all() else
        df.loc[both & ~match.reindex(df.index, fill_value=False),
               [no_col, pos_col]].head(5).to_dict("records"),
    }


# ---------------------------------------------------------------------------
# 전압 측정 단계
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Stage:
    key: str            # 내부 키
    label: str          # 표시용 (발표 그림)
    order: int          # 공정 순서
    soc: float | None   # SOC [%]
    value_col: str      # 전압 값 컬럼
    temp_col: str | None    # 측정 시점 온도 컬럼
    start_col: str | None   # 측정 시작 시각
    in_charger: bool    # True=충방전기 내부(팬 가동) / False=별도 OCV 계측기


def _ocv(n: int, order: int, soc: float) -> Stage:
    return Stage(f"ocv{n}", f"OCV #{n:02d}", order, soc,
                 f"OCV #{n:02d} OCV", f"OCV #{n:02d}_온도",
                 f"OCV #{n:02d} 시작시간", True)


def _prvt(n: int, order: int) -> Stage:
    return Stage(f"prvt{n}", f"전용OCV #{n:02d}", order, 30.0,
                 f"PRIVT OCV #{n:02d} OCV", f"PRIVT OCV #{n:02d} 온도",
                 f"PRIVT OCV #{n:02d} 시작시간", False)


#: 전압 측정 단계 (공정 순서). SOC 는 docs/00_facts.md §1.2
STAGES: list[Stage] = [
    _ocv(1, 10, 0.0),      # 충전 전
    _ocv(2, 20, 10.0),     # C1·C2 직후, HT1 전   ← matched-SOC 짝: ocv3
    _ocv(3, 30, 10.0),     # HT1·RT2 후          ← matched-SOC 짝: ocv2
    _ocv(4, 40, 70.0),     # C3·C4 직후, HT2 전   ← matched-SOC 짝: ocv5
    _ocv(5, 50, 70.0),     # HT2·RT3 후          ← matched-SOC 짝: ocv4
    _ocv(6, 60, 100.0),    # 만충
    _ocv(7, 70, 30.0),     # 방전 후
    _prvt(1, 80),
    _prvt(2, 90),
    _prvt(3, 100),
]
STAGE_BY_KEY = {s.key: s for s in STAGES}

#: 판정값 직접 컬럼 (= PRVT1 − PRVT3)
DOCV7_COL = "Delta OCV #07 DOCV"

#: 단위. 구 저장소에서 OCV 는 V, Delta OCV 는 이미 mV 였음 → 수령 후 A6 에서 검증.
UNIT_SCALE_TO_MV = {"stage": 1000.0, "docv7": 1.0}


# ---------------------------------------------------------------------------
# 온도를 갖는 공정 스텝
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ThermStep:
    key: str
    label: str
    order: int
    phase: str                    # lci | charge | discharge | ocv_meas
    temp_cols: dict = field(default_factory=dict)   # {'min'/'mean'/'max'/'single': 컬럼}
    start_col: str | None = None
    end_col: str | None = None
    dur_col: str | None = None    # 작업시간
    self_heating: bool = False    # 자기발열이 있는 스텝인가 (F3 검증용)


def _cd(prefix: str, n: int, order: int, phase: str) -> ThermStep:
    p = f"{prefix} #{n:02d}"
    return ThermStep(
        f"{phase}{n}", p, order, phase,
        {"min": f"{p} 최저 온도", "mean": f"{p} 평균온도", "max": f"{p} 최고 온도"},
        f"{p} 시작시간", f"{p} 종료시간", f"{p} 작업시간", self_heating=True)


def _lci(n: int, order: int) -> ThermStep:
    p = f"Low Current Inspection #{n:02d}"
    return ThermStep(f"lci{n}", p, order, "lci", {"single": f"{p} 온도"},
                     f"{p} 시작시간", f"{p} 종료시간", f"{p} 작업시간",
                     self_heating=False)


#: 온도 보유 스텝 전체. order 는 공정 순서(전압 단계와 같은 축).
THERM_STEPS: list[ThermStep] = (
    [_lci(1, 1), _lci(2, 2)]
    + [_cd("Charge", 1, 12, "charge"), _cd("Charge", 2, 14, "charge")]
    + [_cd("Charge", 3, 32, "charge"), _cd("Charge", 4, 34, "charge")]
    + [_cd("Charge", n, 50 + n, "charge") for n in (5, 6, 7)]
    + [_cd("DisCharge", n, 60 + n, "discharge") for n in range(1, 8)]
    + [ThermStep(s.key, s.label, s.order, "ocv_meas",
                 {"single": s.temp_col}, s.start_col, None, None, False)
       for s in STAGES if s.temp_col]
)

#: ★ 자기발열이 전혀 없는 대조군 — F3(진폭 ∝ 발열량) 검증의 baseline
NO_HEAT_STEPS = ["lci1", "lci2", "ocv1"]


# ---------------------------------------------------------------------------
# 에이징 구간
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Aging:
    key: str
    label: str
    kind: str          # high | room
    start_col: str
    end_col: str
    box_col: str | None = None   # 설비(챔버) ID


#: 고온 에이징 60°C. box_col 은 설비 메타데이터로 추정 — 정체 확인 필요.
AGING_HIGH = [
    Aging(f"ht{n}", f"고온에이징 #{n:02d}", "high",
          f"Start Time_High Temp Aging #{n:02d}",
          f"End Time_High Temp Aging #{n:02d}",
          f"Work BOX_High Temp Aging #{n:02d}")
    for n in (1, 2)
]
#: 상온 에이징. ★#07 이후는 사용하지 않는다 (사용자 지시).
RT_MAX = 6
AGING_ROOM = [
    Aging(f"rt{n}", f"상온에이징 #{n:02d}", "room",
          f"Start Time_Room Temp Aging #{n:02d}",
          f"End Time_Room Temp Aging #{n:02d}")
    for n in range(1, RT_MAX + 1)
]
AGING = AGING_HIGH + AGING_ROOM

HIGH_TEMP_C = 60.0   # 고온 에이징 설정 온도
ROOM_TEMP_C = 25.0   # 상온 기준


# ---------------------------------------------------------------------------
# 종료전압 (분극 프록시용)
# ---------------------------------------------------------------------------
#: {종료전압 컬럼: 짝이 되는 직후 전압 단계 key}
#: 분극 P = V_end − 직후 OCV ≈ 내부저항 + 확산분극  (docs/01_analysis_plan.md D7)
END_VOLTAGE_COLS = {
    "End Voltage_Charge #01": None,   # C1 다음이 C2(연속) → 짝 없음
    "End Voltage_Charge #02": "ocv2",  # ★ C2 종료 → OCV#02 : 분극 계산 가능
    "End Voltage_DisCharge #01": None,
    "End Voltage_DisCharge #02": None,
    "End Voltage_DisCharge #03": None,  # D1~D7 연속 → 사이에 OCV 없음
}
#: 추가 요청 대상: End Voltage_DisCharge #07 (있으면 ocv7 과 짝 → 최고 감도 저항 프록시)


# ---------------------------------------------------------------------------
# 자기방전 3지표 (docs/01_analysis_plan.md C 그룹)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Indicator:
    key: str
    label: str
    from_stage: str
    to_stage: str
    soc: float
    hot_aging: str | None   # 구간에 포함된 고온 에이징 key (아레니우스 분해용)


INDICATORS = [
    Indicator("S_A", "자기방전 A (고온1 구간)", "ocv2", "ocv3", 10.0, "ht1"),
    Indicator("S_B", "자기방전 B (고온2 구간)", "ocv4", "ocv5", 70.0, "ht2"),
    Indicator("S_C", "자기방전 C (전용OCV = 현 판정값)", "prvt1", "prvt3", 30.0, None),
]

# ---------------------------------------------------------------------------
# 판정 로직 (현행)
# ---------------------------------------------------------------------------
JUDGE_OFFSET_MV = 0.8     # 트레이 mode + 0.8 mV 초과 → 불량
JUDGE_MODE_BIN_MV = 0.01  # mode 계산용 히스토그램 bin (수령 후 A1 에서 재확인)


def all_expected_columns() -> list[str]:
    """스키마가 기대하는 전체 컬럼 목록 (inspect 단계에서 대조용)."""
    cols = list(ID_COLS.values()) + [DOCV7_COL] + list(END_VOLTAGE_COLS)
    for s in STAGES:
        cols += [c for c in (s.value_col, s.temp_col, s.start_col) if c]
    for t in THERM_STEPS:
        cols += [c for c in t.temp_cols.values() if c]
        cols += [c for c in (t.start_col, t.end_col, t.dur_col) if c]
    for a in AGING:
        cols += [c for c in (a.start_col, a.end_col, a.box_col) if c]
    return sorted(set(cols))
