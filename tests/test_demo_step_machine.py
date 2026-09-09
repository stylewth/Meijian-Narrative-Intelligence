"""远端单步推进机测试：27 步走完全程、进度文本与边界。"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.demo_notification_control import (  # noqa: E402
    advance_demo_one_step,
    current_demo_progress_text,
)
from src.ui.workspace_shell import Workspace  # noqa: E402


def test_full_walk_27_steps() -> None:
    state: dict = {}
    feedback = [advance_demo_one_step(state) for _ in range(27)]

    assert feedback[0] == "数据预处理 · 「结构校验」完成（1/5）"
    assert feedback[4].startswith("数据预处理 · 「冻结」完成（5/5）")
    assert feedback[5] == "叙事压力测试 · 进入「基础压力检查」（2/5）"
    assert "团队确认" in feedback[9]
    assert feedback[10] == "实时决策看板 · 检查点 1：增量评论进入"
    assert feedback[12] == "实时决策看板 · 检查点 1：叙事文案更新"
    assert feedback[25].startswith("五个检查点全部释放完毕")
    assert feedback[26] == "终幕 · 第二幕已展开（三支柱、场景与边界）"

    assert state["evolution_show_finale"] is True
    assert state["evolution_finale_act"] == 2
    assert state["active_workspace"] == Workspace.REALTIME_DECISION.value
    assert len(state["completed_workspaces"]) == 2

    assert advance_demo_one_step(state) == "已到终幕结尾"


def test_progress_text_follows_state() -> None:
    state: dict = {}
    assert current_demo_progress_text(state) == "数据预处理 · 「结构校验」（0/5）"
    advance_demo_one_step(state)
    assert current_demo_progress_text(state) == "数据预处理 · 「清洗」（1/5）"
    for _ in range(4):
        advance_demo_one_step(state)
    assert current_demo_progress_text(state) == "叙事压力测试 · 「AI 原始五项」（1/5）"
    for _ in range(5):
        advance_demo_one_step(state)
    assert "团队确认" not in current_demo_progress_text(state)
    assert current_demo_progress_text(state) == "实时决策看板 · 检查点 0/5 · 分数与排名更新"
    for _ in range(3):
        advance_demo_one_step(state)
    assert current_demo_progress_text(state) == "实时决策看板 · 检查点 1/5 · 叙事文案更新"
    for _ in range(13):
        advance_demo_one_step(state)
    assert current_demo_progress_text(state) == "实时决策看板 · 终幕 第1/2幕"
    advance_demo_one_step(state)
    assert current_demo_progress_text(state) == "实时决策看板 · 终幕 第2/2幕"


def test_step_machine_is_idempotent_at_boundaries() -> None:
    state: dict = {}
    for _ in range(27):
        advance_demo_one_step(state)
    for _ in range(3):
        assert advance_demo_one_step(state) == "已到终幕结尾"
    assert state["evolution_finale_act"] == 2
