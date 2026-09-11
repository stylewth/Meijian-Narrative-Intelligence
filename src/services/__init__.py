from __future__ import annotations

import json
from typing import Any

from src.brand_profile import active_brand_prompt_block
from src.prompt_loader import load_prompt


def _build_system_prompt(task_name: str) -> str:
    return (
        f"{load_prompt('system')}\n\n"
        f"{active_brand_prompt_block()}\n\n"
        f"{load_prompt(task_name)}"
    )


def _serialize_prompt_input(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
