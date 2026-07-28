"""타임스탬프 기반 분석 — docs/01_analysis_plan.md K 그룹.

RT #01~#09 실측 순서가 기존 문서 이해와 다르다는 것이 이미 확인됐다(00_facts §2.3d).
그래서 "RT2 는 OCV3 앞이다" 같은 라벨 매핑을 코드에 하드코딩하지 않고,
`infer_process_order()` 로 실측 타임스탬프에서 순서를 직접 확인하는 것을 우선한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema


def to_hours(start: pd.Series, end: pd.Series) -> pd.Series:
    return (pd.to_datetime(end, errors="coerce")
            - pd.to_datetime(start, errors="coerce")).dt.total_seconds() / 3600.0


def infer_process_order(df: pd.DataFrame) -> pd.DataFrame:
    """[K0/A10] 모든 스텝의 중앙값 시작시각으로 실제 공정 순서를 역추정한다.

    RT#01~#09, HT#01~#02, OCV/PRVT, Charge/DisCharge 가 실제로 어느 순서인지
    타임스탬프로 직접 확인 — 라벨(번호)을 믿지 않고 데이터로 검증한다.
    """
    rows = []
    for a in schema.AGING:
        col = f"start_{a.key}"
        if col in df.columns:
            s = df[col]
            rows.append({"key": a.key, "label": a.label, "kind": "aging",
                        "median_start": s.median(), "n": int(s.notna().sum())})
    for s in schema.STAGES:
        col = f"start_{s.key}"
        if col in df.columns:
            v = df[col]
            rows.append({"key": s.key, "label": s.label, "kind": "voltage",
                        "median_start": v.median(), "n": int(v.notna().sum())})
    for t in schema.THERM_STEPS:
        col = f"start_{t.key}"
        if col in df.columns and t.phase in ("charge", "discharge", "lci"):
            v = df[col]
            rows.append({"key": t.key, "label": t.label, "kind": t.phase,
                        "median_start": v.median(), "n": int(v.notna().sum())})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("median_start", ignore_index=True)
        out["order_rank"] = np.arange(1, len(out) + 1)
    return out


def order_consistency(df: pd.DataFrame, key_a: str, key_b: str) -> float:
    """[검증] 트레이마다 key_a 가 key_b 보다 항상 먼저인지 (일관성 비율 0~1).

    infer_process_order 는 중앙값 기준이라 트레이별로 뒤집힐 수 있다 — 그 비율을 본다.
    """
    ca, cb = f"start_{key_a}", f"start_{key_b}"
    if ca not in df.columns or cb not in df.columns:
        return np.nan
    a, b = pd.to_datetime(df[ca], errors="coerce"), pd.to_datetime(df[cb], errors="coerce")
    ok = a.notna() & b.notna()
    if ok.sum() == 0:
        return np.nan
    return float((a[ok] < b[ok]).mean())


def aging_hours(df: pd.DataFrame, aging_key: str) -> pd.Series:
    """[K1] 에이징 구간 실제 경과시간(시간 단위)."""
    a = next((x for x in schema.AGING if x.key == aging_key), None)
    if a is None:
        raise KeyError(aging_key)
    s, e = f"start_{a.key}", f"end_{a.key}"
    if s not in df.columns or e not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return to_hours(df[s], df[e])


def high_temp_exposure_hours(df: pd.DataFrame) -> pd.Series:
    """[K5/K8 입력] HT#01 + HT#02 총 노출시간. 아레니우스 분해의 핵심 입력."""
    total = pd.Series(0.0, index=df.index)
    any_present = False
    for a in schema.AGING_HIGH:
        h = aging_hours(df, a.key)
        if h.notna().any():
            total = total.add(h.fillna(0), fill_value=0)
            any_present = True
    return total if any_present else pd.Series(np.nan, index=df.index)


def indicator_hours(df: pd.DataFrame, indicator_key: str) -> pd.Series:
    """[K1/K2] 지표 구간의 실제 경과시간 — 두 단계의 시작시각 차이로 직접 계산.

    OCV/PRVT 는 전부 시작시각을 확보했으므로, 사이의 에이징 스텝을 몰라도
    '전체 경과시간'은 바로 나온다 (에이징 세부 구성은 K8에서 별도로 본다).
    """
    ind = next((i for i in schema.INDICATORS if i.key == indicator_key), None)
    if ind is None:
        raise KeyError(indicator_key)
    s_from = f"start_{ind.from_stage}"
    s_to = f"start_{ind.to_stage}"
    if s_from not in df.columns or s_to not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return to_hours(df[s_from], df[s_to])


def rate_mv_per_day(delta_mv: pd.Series, hours: pd.Series) -> pd.Series:
    """[K2] mV → mV/day. 0 또는 음수 시간은 NaN 처리(시계 오류 방지)."""
    h = hours.where(hours > 0)
    return delta_mv / (h / 24.0)


# ---------------------------------------------------------------------------
# [K10] OCV 측정 직전 휴지시간 — "S_A/S_B 가 이완인지 자기방전인지" 판별의 전제
# ---------------------------------------------------------------------------
#: 신뢰도 높은 것만 명시. ocv3/ocv5 는 사이 에이징 라벨(RT#0x)이 실제로 몇 번인지
#: infer_process_order() 로 확인 후 채운다 — 잘못된 매핑을 하드코딩하지 않는다.
REST_BEFORE_STAGE: dict = {
    "ocv2": "charge2",   # Charge#02 종료 -> OCV#02 시작
    "ocv4": "charge4",
    "ocv6": "charge7",
    "ocv7": "discharge7",
    "prvt1": "discharge7",  # OCV7 이후 RT4 상당 구간 포함 — 세부 미분해, 총량은 정확
}


def rest_before_ocv(df: pd.DataFrame, stage_key: str) -> pd.Series:
    """[K10] 해당 OCV/PRVT 직전, 직전 공정 종료부터의 경과시간(시간).

    짧으면(수 시간 이내) 해당 단계는 전압 이완이 아직 안 끝난 상태 → 오류 감사
    (docs/04_logic_audit.md 오류 1) 의 직접 검증.
    """
    prior_key = REST_BEFORE_STAGE.get(stage_key)
    if prior_key is None:
        return pd.Series(np.nan, index=df.index)
    end_col = f"end_{prior_key}"
    start_col = f"start_{stage_key}"
    if end_col not in df.columns or start_col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return to_hours(df[end_col], df[start_col])


def step_wait_time(df: pd.DataFrame, therm_key: str) -> pd.Series:
    """[K11] (종료−시작) − 작업시간 = 설비 점유 후 대기시간(라인 정체 프록시)."""
    s, e, d = f"start_{therm_key}", f"end_{therm_key}", f"dur_{therm_key}"
    if not all(c in df.columns for c in (s, e, d)):
        return pd.Series(np.nan, index=df.index)
    wall = to_hours(df[s], df[e])
    dur_h = pd.to_numeric(df[d], errors="coerce")
    # dur_col 단위가 분/초일 수 있음 — 상대적 스케일을 wall 과 비교해 자동 추정
    if dur_h.notna().any() and wall.notna().any():
        ratio = float(np.nanmedian(wall / dur_h.replace(0, np.nan)))
        if np.isfinite(ratio) and abs(ratio - 1 / 60) < abs(ratio - 1):
            dur_h = dur_h / 60.0
        elif np.isfinite(ratio) and abs(ratio - 1 / 3600) < abs(ratio - 1):
            dur_h = dur_h / 3600.0
    return wall - dur_h


def clock_sanity(df: pd.DataFrame) -> pd.DataFrame:
    """[A8] 시작<종료 여부 전수 검사. 음수 구간은 시계 불일치 의심."""
    rows = []
    pairs = ([(f"start_{s.key}", f"end_{s.key}", s.label) for s in schema.STAGES
             if s.end_col]
            + [(f"start_{t.key}", f"end_{t.key}", t.label) for t in schema.THERM_STEPS
               if t.end_col]
            + [(f"start_{a.key}", f"end_{a.key}", a.label) for a in schema.AGING])
    for s_col, e_col, label in pairs:
        if s_col not in df.columns or e_col not in df.columns:
            continue
        h = to_hours(df[s_col], df[e_col])
        rows.append({"key": label, "n": int(h.notna().sum()),
                    "n_negative": int((h < 0).sum()),
                    "median_hours": float(h.median()) if h.notna().any() else np.nan})
    return pd.DataFrame(rows)
