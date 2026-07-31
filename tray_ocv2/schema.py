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
    "product_lot": "Product Lot",  # 실제 파일엔 Product/Lot 이 분리 안 되고 한 컬럼(확인됨)
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


def _fan_center_columns(col_centers: list[float]) -> list[str]:
    """도면 팬 중심 좌표(숫자)를 가장 가까운 열 라벨(A~L)로 반올림한다.

    ★수정 이력: 이전 버전은 팬 '사이(gap)' 열(예: D,G,J)을 손으로 계산해 '냉각 최강'으로
    잘못 라벨링했다. 행 축(FAN_ROW_MAX)은 처음부터 '팬 중심 = 차가움' 규약을 썼는데
    열 축만 반대(gap=차가움)로 만든 것 — 두 축의 물리 규약이 안 맞았다.
    이 함수로 col_centers 에서 직접 계산해 **행 축과 동일한 규약**(팬 중심에 가장 가까운
    열이 냉각 최강)을 강제한다. 합성 데이터(정답을 아는 검증)로 이 불일치를 발견했다.
    """
    cols = []
    for c in col_centers:
        idx = max(1, min(N_COLS, int(round(c))))
        cols.append(chr(ord("A") + idx - 1))
    return cols


#: [F2] 예측 — 알파벳 축(가로) 프로파일에서 **냉각 최강(차가움)** 으로 예측되는 열.
#: 팬 중심에 가장 가까운 열 = 차가움 (FAN_ROW_MAX 와 동일 규약). col_centers 에서
#: 직접 계산 — 밴드마다 주기가 다른 것이 판별점(상/하단은 3.2 주기, 중앙은 2.3 주기).
FAN_COL_COLD_OUTER = _fan_center_columns(FAN_BANDS[0]["col_centers"])   # 상·하단 밴드
FAN_COL_COLD_MIDDLE = _fan_center_columns(FAN_BANDS[1]["col_centers"])  # 중앙 밴드


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
    end_col: str | None = None   # 측정 종료 시각 (현재 미사용 — _ocv() 참조)
    dur_col: str | None = None   # 측정 작업시간 (현재 미사용 — _ocv() 참조)
    box_col: str | None = None   # 호기 박스 ID (in_charger=True 인 OCV 만. §3.3 팬분석 주력)


def _ocv(n: int, order: int, soc: float) -> Stage:
    """OCV 측정 자체는 ~1초라 종료시간·작업시간을 요청/사용하지 않는다.

    이유: ① K9(CV종료 스텝 소요시간=저항프록시)는 정전압 종료 메커니즘 전제라 그냥
    전압을 읽는 OCV엔 성립 안 함. ② K6(스캔순서 드리프트)도 1초 안에 144셀을 다
    찍는다면 물리적으로 드리프트가 생길 여지가 없음. 현재 코드에서 end_col/dur_col의
    유일한 소비처였던 timing.clock_sanity() 도 진단용일 뿐 핵심 분석이 아니었다.
    OCV#01~03 은 실 파일에 종료/작업시간이 이미 존재해 그걸로 '정말 짧은지' 확인 가능
    (요청 불필요, 있는 데이터로 검증).
    """
    p = f"OCV #{n:02d}"
    # OCV 는 충방전기 내부에서 측정 → work box = 그 순간의 호기. 팬 서명(F그룹)의 주력
    # 온도장이라, 이 컬럼만으로 호기별 층화가 가능하다(충·방전 박스 없어도 됨, §3.3).
    return Stage(f"ocv{n}", p, order, soc, f"{p} OCV", f"{p}_온도",
                 f"{p} 시작시간", True, box_col=f"Work BOX_{p}")


def _prvt(n: int, order: int) -> Stage:
    p = f"PRIVT OCV #{n:02d}"
    return Stage(f"prvt{n}", f"전용OCV #{n:02d}", order, 30.0, f"{p} OCV",
                 f"{p} 온도", f"{p} 시작시간", False)


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
    box_col: str | None = None    # 충·방전 설비(호기) 박스 ID 컬럼 (§00_facts 2.3d/3.3)


def _cd(prefix: str, n: int, order: int, phase: str) -> ThermStep:
    p = f"{prefix} #{n:02d}"
    return ThermStep(
        f"{phase}{n}", p, order, phase,
        {"min": f"{p} 최저 온도", "mean": f"{p} 평균온도", "max": f"{p} 최고 온도"},
        f"{p} 시작시간", f"{p} 종료시간", f"{p} 작업시간", self_heating=True,
        box_col=f"Work BOX_{prefix} #{n:02d}")


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
# [오류7] OCV 단계의 직전 열이력 분류 — docs/04_logic_audit.md 오류7
# ---------------------------------------------------------------------------
# F0(온도 필드 재현성)의 옛 기준("모든 스텝에서 같은 무늬여야 한다")은 F3(발열이 있어야
# 구배가 드러난다)과 모순된다. 실측(Run #1)도 OCV1·3·5(에이징/휴지 직후 열평형) 와
# OCV2·4·6·7(충방전 직후 잔열) 두 군집으로 깨끗이 갈렸다 — F3이 맞고 옛 F0 기준이 틀렸다.
# 이 분류는 "그때 열이 있었는가"라는 공정 순서상의 사실에서 정하며, 상관행렬 결과를
# 보고 나중에 끼워맞추지 않는다(순환 방지).
STAGE_THERMAL_STATE: dict = {
    "ocv1": "equilibrium",       # 충전 전
    "ocv2": "recent_heat",       # C1·C2 직후
    "ocv3": "equilibrium",       # HT1 + RT2 후
    "ocv4": "recent_heat",       # C3·C4 직후
    "ocv5": "equilibrium",       # HT2 + RT3 후
    "ocv6": "recent_heat",       # C5·C6·C7 직후 (만충)
    "ocv7": "recent_heat",       # D1~D7 직후
    "prvt1": "separate_instrument",
    "prvt2": "separate_instrument",
    "prvt3": "separate_instrument",
}


def classify_thermal_state(therm_key: str) -> str:
    """[오류7] 컬럼명에서 뽑은 bare key(예: 'ocv2', 'charge1')로 열이력 분류를 얻는다.

    반환: 'recent_heat' | 'equilibrium' | 'separate_instrument' | 'unknown'
    """
    if therm_key in STAGE_THERMAL_STATE:
        return STAGE_THERMAL_STATE[therm_key]
    step = next((t for t in THERM_STEPS if t.key == therm_key), None)
    if step is None:
        return "unknown"
    if step.phase == "lci":
        # 실측(F3) 근거: LCI 진폭이 다른 모든 계열과 무관하게 유독 크다(±4.6~4.8) →
        # 충방전기와 무관한 별도 검사설비 고유 패턴으로 추정.
        return "separate_instrument"
    if step.self_heating:
        return "recent_heat"
    return "unknown"

# ---------------------------------------------------------------------------
# 충·방전 설비 박스 → 호기 매핑 (§00_facts 2.3d/3.3, 2026-07-31 확인)
# ---------------------------------------------------------------------------
# 박스 값 예: "CDC #MP1 -2-BOX #03-06 (1-3-6)" 에서 `#MP{a}-{b}` 토큰을 뽑아 호기로 매핑.
# 호기별 기류가 다르므로(2·3호 바닥6×6+앞뒤외기 / 4·5호 윗4-5-4+바닥4×4+측면외기)
# 팬 서명(F 그룹)은 호기별로 분리해 분석해야 한다.
_BOX_MP_TOKEN_RE = re.compile(r"#\s*MP\s*(\d+)\s*-\s*(\d+)", re.IGNORECASE)
#: "BOX #{연:02d}-{단:02d}" — 연(bay, 가로 1~14) · 단(tier, 높이 1~10). "4연 5단" 예: `#04-05`
_BOX_SLOT_RE = re.compile(r"BOX\s*#\s*(\d+)\s*-\s*(\d+)", re.IGNORECASE)

#: (MP앞자리, MP뒷자리) → 호기 번호. #MP1-1(=1호 추정)은 값 미제공이라 뺐다 → unknown 처리.
CHARGER_BY_MP_TOKEN = {(1, 2): 2, (2, 1): 3, (2, 2): 4, (2, 3): 5}

#: 호기별 바닥팬 격자 (§3.3). 8셀 이상·기류 분석에서 참조.
CHARGER_BOTTOM_FAN_GRID = {2: (6, 6), 3: (6, 6), 4: (4, 4), 5: (4, 4)}
#: 호기별 추가 외기 방향. 'col_A'=A쪽 횡류, 'col_L'=L쪽 횡류, 'row_ends'=앞뒤(세로) 외기.
CHARGER_EXTRA_AIRFLOW = {2: "row_ends", 3: "row_ends", 4: "col_A", 5: "col_L"}
#: 랙 크기 — 호기당 연(가로) 1~14, 단(높이) 1~10.
BAY_MAX, TIER_MAX = 14, 10


def parse_box(box_value) -> dict | None:
    """박스 문자열 → {'charger': 호기(2~5), 'bay': 연(1~14), 'tier': 단(1~10)}.

    예: "CDC #MP2 -2-BOX #04-05 (1-4-5)" → {'charger':4, 'bay':4, 'tier':5} (4호 4연 5단).
    - 호기: `#MP{a}-{b}` 토큰(CHARGER_BY_MP_TOKEN). 미등록이면 charger=None.
    - 연·단: `BOX #{연}-{단}` 슬롯. 없으면 bay/tier=None.
    셋 다 못 뽑으면 None. 트레이의 랙 내 위치(연·단)는 호기별 기류 분석의 층화 축이다(§3.3).
    """
    if box_value is None:
        return None
    s = str(box_value)
    charger = bay = tier = None
    m = _BOX_MP_TOKEN_RE.search(s)
    if m:
        charger = CHARGER_BY_MP_TOKEN.get((int(m.group(1)), int(m.group(2))))
    m2 = _BOX_SLOT_RE.search(s)
    if m2:
        bay, tier = int(m2.group(1)), int(m2.group(2))
    if charger is None and bay is None and tier is None:
        return None
    return {"charger": charger, "bay": bay, "tier": tier}


def charger_from_box(box_value) -> int | None:
    """박스 문자열 → 충방전기 호기 번호(2~5). 실패/미등록이면 None. `parse_box` 편의 래퍼.

    호기별로 냉각 기류 형상이 다르므로(§3.3), 온도 지문을 이 값으로 층화해야 한다.
    """
    info = parse_box(box_value)
    return info["charger"] if info else None


# ---------------------------------------------------------------------------
# 판정 로직 (현행)
# ---------------------------------------------------------------------------
JUDGE_OFFSET_MV = 0.8     # 트레이 mode + 0.8 mV 초과 → 불량
JUDGE_MODE_BIN_MV = 0.01  # mode 계산용 히스토그램 bin (수령 후 A1 에서 재확인)


def all_expected_columns() -> list[str]:
    """스키마가 기대하는 전체 컬럼 목록 (inspect 단계에서 대조용)."""
    cols = list(ID_COLS.values()) + [DOCV7_COL] + list(END_VOLTAGE_COLS)
    for s in STAGES:
        cols += [c for c in (s.value_col, s.temp_col, s.start_col,
                             s.end_col, s.dur_col, s.box_col) if c]
    for t in THERM_STEPS:
        cols += [c for c in t.temp_cols.values() if c]
        cols += [c for c in (t.start_col, t.end_col, t.dur_col, t.box_col) if c]
    for a in AGING:
        cols += [c for c in (a.start_col, a.end_col, a.box_col) if c]
    return sorted(set(cols))
