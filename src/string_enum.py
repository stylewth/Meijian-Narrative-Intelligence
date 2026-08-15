from __future__ import annotations

from enum import Enum


class StringEnum(str, Enum):
    """Python 3.10-compatible string enum with StrEnum-style text output."""

    def __str__(self) -> str:
        return self.value
