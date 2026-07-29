"""원본 Export 파일 로딩 + 셀 단위 표준 테이블 구성.

원본은 wide 포맷(1행 = 셀 1개)이라 그대로 읽고, `schema.py` 매핑을 적용해
분석에 쓰기 좋은 컬럼명(row/col/v_ocv1 등)으로 정리한다. 원본 컬럼은 버리지 않고
표준 컬럼 옆에 유지한다(디버깅·재검증 용도).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import schema


def read_any(path: str) -> pd.DataFrame:
    """xlsx/xls/csv 자동 판별 로딩. csv 는 한글 인코딩을 순서대로 시도한다."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".csv":
        last_err = None
        for enc in ("utf-8-sig", "cp949", "utf-8"):
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError as e:
                last_err = e
        raise last_err
    raise ValueError(f"지원하지 않는 파일 형식: {ext}")


def inspect(path: str) -> dict:
    """[0단계] 파일 컬럼과 스키마 기대 컬럼을 대조한다. 매핑 확정 전 먼저 실행."""
    df = read_any(path)
    expected = set(schema.all_expected_columns())
    actual = set(df.columns)
    return {
        "n_rows": len(df),
        "n_cols_actual": len(df.columns),
        "n_cols_expected": len(expected),
        "missing_expected": sorted(expected - actual),
        "unmapped_actual": sorted(actual - expected),
    }


def _to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


#: '2026-07-08 오전 5:39:34' 형식 — pandas 기본 파서가 인식 못 하는 한글 오전/오후.
#: (에이징 시각 컬럼은 엑셀 datetime 셀로 저장돼 있어 이 문제가 없다. OCV/LCI/Charge/
#:  DisCharge 시각은 문자열로 저장돼 있어 이 파서가 필요하다 — 2026-07 실사례로 확인)
_KOREAN_AMPM_RE = r"^(\d{4}-\d{2}-\d{2})\s*(오전|오후)\s*(\d{1,2}):(\d{2}):(\d{2})$"


def _parse_korean_ampm(s: pd.Series) -> pd.Series:
    ext = s.astype(str).str.strip().str.extract(_KOREAN_AMPM_RE)
    ext.columns = ["date", "ampm", "h", "m", "sec"]
    ok = ext["date"].notna()
    if not ok.any():
        return pd.Series(pd.NaT, index=s.index)

    h = pd.to_numeric(ext["h"], errors="coerce")
    is_pm = ext["ampm"] == "오후"
    h24 = h.where(~((~is_pm) & (h == 12)), 0)      # 오전 12시(자정) → 0시
    h24 = h24.where(~(is_pm & (h != 12)), h + 12)   # 오후 1~11시 → +12시 (오후 12시는 그대로)

    combined = (ext["date"] + " " + h24.astype("Int64").astype(str).str.zfill(2)
               + ":" + ext["m"] + ":" + ext["sec"])
    return pd.to_datetime(combined.where(ok), format="%Y-%m-%d %H:%M:%S", errors="coerce")


def _to_datetime(s: pd.Series) -> pd.Series:
    """날짜 파싱. 한글 오전/오후 형식이면 그쪽을 먼저 시도한다.

    컬럼 전체가 한글 형식이면(엑셀 문자열 저장, 실사례) pandas 표준 파서는 전부
    실패한 뒤에야 행별 dateutil 폴백으로 넘어가 149k행 기준 수 초~수십 초가 걸린다.
    먼저 감지해서 순서를 바꾸면 빠른 벡터화 경로만 탄다.
    """
    looks_korean = (s.dtype == object and s.notna().any()
                    and s.dropna().astype(str).str.contains("오전|오후").any())

    if looks_korean:
        out = _parse_korean_ampm(s)
        still_missing = out.isna() & s.notna()
        if still_missing.any():
            out = out.copy()
            out.loc[still_missing] = pd.to_datetime(s[still_missing], errors="coerce")
        return out

    primary = pd.to_datetime(s, errors="coerce")
    still_missing = primary.isna() & s.notna()
    if still_missing.any():
        primary = primary.copy()
        primary.loc[still_missing] = _parse_korean_ampm(s[still_missing])
    return primary


def build_cell_table(raw: pd.DataFrame) -> pd.DataFrame:
    """원본 wide 테이블 → 셀 단위 표준 테이블.

    반환 컬럼:
      식별  : tray_id, cell_no, cell_pos, grade, row, col, lot, product, ...
      전압  : v_<stage_key>   (원 단위 그대로 — 스케일은 qc.detect_unit_scale 로 결정)
      온도  : t_<stage_key>   (전압 단계 측정시점 온도), temp_<therm_key>_{min,mean,max,single}
      시각  : start_<stage_key>, end_<stage_key>, dur_<stage_key>(전압 단계)
              start_<therm_key>, end_<therm_key>, dur_<therm_key> (온도 스텝)
              start_<aging_key>, end_<aging_key>, box_<aging_key>
      기타  : docv7_raw (Delta OCV #07 컬럼), pol_<end_voltage_col> (있는 것만)
    """
    cols: dict[str, pd.Series] = {}

    for key, col in schema.ID_COLS.items():
        if col in raw.columns:
            cols[key] = raw[col]

    pos_col, no_col = schema.ID_COLS["cell_pos"], schema.ID_COLS["cell_no"]
    row_col = pd.Series(np.nan, index=raw.index, dtype="float")
    col_col = pd.Series(np.nan, index=raw.index, dtype="float")
    if pos_col in raw.columns:
        parsed = raw[pos_col].map(schema.parse_cell_pos)
        row_col = parsed.map(lambda t: t[0] if t else np.nan)
        col_col = parsed.map(lambda t: t[1] if t else np.nan)
    missing = row_col.isna()
    if missing.any() and no_col in raw.columns:
        parsed_no = raw.loc[missing, no_col].map(schema.parse_cell_no)
        row_col.loc[missing] = parsed_no.map(lambda t: t[0] if t else np.nan)
        col_col.loc[missing] = parsed_no.map(lambda t: t[1] if t else np.nan)
    cols["row"] = row_col
    cols["col"] = col_col

    for s in schema.STAGES:
        if s.value_col in raw.columns:
            cols[f"v_{s.key}"] = _to_numeric(raw[s.value_col])
        if s.temp_col and s.temp_col in raw.columns:
            cols[f"t_{s.key}"] = _to_numeric(raw[s.temp_col])
        if s.start_col and s.start_col in raw.columns:
            cols[f"start_{s.key}"] = _to_datetime(raw[s.start_col])
        if s.end_col and s.end_col in raw.columns:
            cols[f"end_{s.key}"] = _to_datetime(raw[s.end_col])
        if s.dur_col and s.dur_col in raw.columns:
            cols[f"dur_{s.key}"] = _to_numeric(raw[s.dur_col])

    for t in schema.THERM_STEPS:
        for stat, col in t.temp_cols.items():
            if col in raw.columns:
                cols[f"temp_{t.key}_{stat}"] = _to_numeric(raw[col])
        if t.start_col and t.start_col in raw.columns:
            cols[f"start_{t.key}"] = _to_datetime(raw[t.start_col])
        if t.end_col and t.end_col in raw.columns:
            cols[f"end_{t.key}"] = _to_datetime(raw[t.end_col])
        if t.dur_col and t.dur_col in raw.columns:
            cols[f"dur_{t.key}"] = _to_numeric(raw[t.dur_col])

    for a in schema.AGING:
        if a.start_col in raw.columns:
            cols[f"start_{a.key}"] = _to_datetime(raw[a.start_col])
        if a.end_col in raw.columns:
            cols[f"end_{a.key}"] = _to_datetime(raw[a.end_col])
        if a.box_col and a.box_col in raw.columns:
            cols[f"box_{a.key}"] = raw[a.box_col]

    if schema.DOCV7_COL in raw.columns:
        cols["docv7_raw"] = _to_numeric(raw[schema.DOCV7_COL])

    for end_col in schema.END_VOLTAGE_COLS:
        if end_col in raw.columns:
            safe = end_col.replace(" ", "_").replace("#", "")
            cols[f"endv_{safe}"] = _to_numeric(raw[end_col])

    return pd.concat(cols, axis=1)


def load(path: str) -> pd.DataFrame:
    """편의 함수: 읽기 + 표준화를 한 번에."""
    return build_cell_table(read_any(path))
