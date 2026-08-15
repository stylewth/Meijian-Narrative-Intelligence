from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import IO, Any

import pandas as pd

from src.data_validator import ValidatedDataset, validate_dataframe
from src.schemas import SampleType


FileSource = str | PathLike[str] | IO[str] | IO[bytes]


def load_tabular_file(source: FileSource, filename: str | None = None) -> pd.DataFrame:
    if filename is None:
        if not isinstance(source, (str, PathLike)):
            raise ValueError("文件对象必须提供 filename")
        filename = Path(source).name
    suffix = Path(filename).suffix.lower()
    if suffix == ".xls":
        raise ValueError("不支持旧版 .xls，请转换为 .xlsx 或 .csv")
    if suffix not in {".xlsx", ".csv"}:
        raise ValueError(f"不支持的文件格式: {suffix or '无扩展名'}")
    if hasattr(source, "seek"):
        source.seek(0)  # type: ignore[union-attr]
    try:
        frame = pd.read_excel(source, engine="openpyxl") if suffix == ".xlsx" else pd.read_csv(source)
    except (pd.errors.EmptyDataError, ValueError) as exc:
        raise ValueError(f"文件无法读取或内容为空: {filename}") from exc
    if frame.empty and len(frame.columns) == 0:
        raise ValueError(f"文件内容为空: {filename}")
    return frame


def load_validated_tabular_file(
    source: FileSource,
    filename: str | None = None,
    *,
    default_sample_type: SampleType | str | None = None,
    max_records: int = 500,
) -> ValidatedDataset:
    """Read a supported file and apply the same contract validator as Feishu imports."""

    return validate_dataframe(
        load_tabular_file(source, filename),
        default_sample_type=default_sample_type,
        max_records=max_records,
    )
