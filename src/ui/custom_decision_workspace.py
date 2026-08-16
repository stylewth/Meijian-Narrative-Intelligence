"""自定义使用入口的自由链路渲染：压力测试与实时决策两个工作区。

线上公开部署（梅见案例展示）只回放冻结产物，不经过本模块；本地配置了
``.env`` 模型密钥的自定义使用入口，由本模块驱动 DecisionRunner 在三个
既有工作区中跑真实推理：预处理（导入+在线标注+发布）→ 叙事压力测试
（基线+五维压力检查+挑战修订）→ 实时决策看板（选线→holdout→冻结→增量批）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import Settings
from src.llm_client import LLMClient
from src.schemas import (
    CommentRecord,
    DecisionEvolutionCheckpoint,
    DecisionFoundationState,
    EvidenceAtom,
)
from src.services.decision_runner import DecisionRunner
from src.services.prepared_corpus import list_prepared_corpora
from src.ui.workspace_shell import Workspace, activate_workspace, complete_workspace

_RUNNER_KEY = "custom_decision_runner"
_FOUNDATION_KEY = "custom_decision_foundation"
_CHECKPOINTS_KEY = "custom_decision_checkpoints"
_SOURCES_KEY = "custom_decision_sources"
_LOG_KEY = "custom_decision_log"


@dataclass
class CustomRunInputs:
    """本地自由运行的最小输入束；与官方链路的 DecisionInputBundle 同构。"""

    baseline_records: list[CommentRecord]
    baseline_atoms: list[EvidenceAtom]


def _log(message: str) -> None:
    st.session_state.setdefault(_LOG_KEY, []).append(message)


def _load_package(
    prepared_root: Path, package_id: str
) -> tuple[list[CommentRecord], list[EvidenceAtom]]:
    for published in list_prepared_corpora(prepared_root):
        if published.manifest.package_id == package_id:
            return list(published.package.records), list(published.package.evidence_atoms)
    raise ValueError(f"未找到已发布语料包: {package_id}")


def _render_ranking_table(foundation: DecisionFoundationState) -> None:
    rows = []
    for ranked in foundation.ranking.ranked_candidates:
        stress = next(
            (
                item
                for item in foundation.stress_results
                if item.candidate_id == ranked.candidate.candidate_id
            ),
            None,
        )
        passed = sum(
            1
            for check in (stress.checks if stress else [])
            if check.execution_status.value == "COMPLETED"
            and check.decision is not None
            and check.decision.value == "PASS"
        )
        rows.append(
            {
                "排名": ranked.rank,
                "候选": ranked.candidate.candidate_id,
                "标题": ranked.candidate.title,
                "加权分": ranked.weighted_score,
                "压力检查通过": f"{passed} / 5",
            }
        )
    st.dataframe(rows, use_container_width=True)


def _render_stress_details(foundation: DecisionFoundationState) -> None:
    for stress in foundation.stress_results:
        with st.expander(f"压力检查 · {stress.candidate_id}", expanded=False):
            st.dataframe(
                [
                    {
                        "检查": check.check_type.value,
                        "状态": check.execution_status.value,
                        "结论": check.decision.value if check.decision else "—",
                        "依据": check.rationale,
                    }
                    for check in stress.checks
                ],
                use_container_width=True,
            )


def _render_checkpoint(checkpoint: DecisionEvolutionCheckpoint) -> None:
    with st.expander(
        f"{checkpoint.checkpoint_id} · release-{checkpoint.release_index} · "
        f"主候选 {checkpoint.selected_candidate_id}",
        expanded=False,
    ):
        rows = [
            {
                "候选": snapshot.ranked_narrative.candidate.candidate_id,
                "排名": snapshot.ranked_narrative.rank,
                "加权分": snapshot.ranked_narrative.weighted_score,
                "业务状态": snapshot.business_status.value,
                "支持证据": snapshot.supporting_count,
                "反对证据": snapshot.counter_count,
            }
            for snapshot in checkpoint.candidates
        ]
        st.dataframe(rows, use_container_width=True)
        for patch in checkpoint.patches:
            st.markdown(
                f"**修订 {patch.candidate_id} · {patch.field_name}**：{patch.reason}"
            )
        if checkpoint.switch_suggestion is not None:
            st.warning(
                f"换线建议：{checkpoint.switch_suggestion.from_candidate_id} → "
                f"{checkpoint.switch_suggestion.to_candidate_id}；"
                f"{checkpoint.switch_suggestion.reason}"
            )


def _render_package_selection(prepared_root: Path) -> dict[str, Any] | None:
    packages = list_prepared_corpora(prepared_root)
    if not packages:
        st.warning(
            "尚未发布任何自定义语料包：请先完成『数据预处理』工作区的导入、在线标注与发布。"
        )
        return None
    options = {published.manifest.package_id: published for published in packages}
    label = lambda package_id: (  # noqa: E731
        f"{package_id} · {len(options[package_id].package.records)} 条评论 / "
        f"{len(options[package_id].package.evidence_atoms)} 条证据"
    )
    baseline_id = st.selectbox("基线语料包", list(options), format_func=label)
    other_ids = [package_id for package_id in options if package_id != baseline_id]
    if len(other_ids) < 2:
        st.warning(
            "完整自由链路要求挑战批与 holdout 批语料：请再发布至少两个语料包"
            "（可在预处理工作区导入不同批次数据后分别发布）。"
        )
        return None
    challenge_id = st.selectbox("挑战批语料包", other_ids)
    holdout_candidates = [pid for pid in other_ids if pid != challenge_id]
    holdout_id = st.selectbox("holdout 语料包", holdout_candidates)
    release_ids = st.multiselect(
        "增量批语料包（按顺序释放，可选）",
        [pid for pid in other_ids if pid not in {challenge_id, holdout_id}],
    )
    return {
        "baseline": baseline_id,
        "challenge": challenge_id,
        "holdout": holdout_id,
        "releases": list(release_ids),
    }


def _render_log() -> None:
    log = st.session_state.get(_LOG_KEY, [])
    if log:
        with st.expander("运行日志", expanded=False):
            for line in log:
                st.write(line)


def render_custom_pressure_test(
    *,
    prepared_root: Path,
    online_enabled: bool,
    dotenv_path: Path,
) -> None:
    st.subheader("叙事压力测试 · 自定义语料真实链路")
    if not online_enabled:
        st.info(
            "自由链路需要本地 .env 配置 LLM_API_KEY 与 LLM_MODEL；"
            "公开部署不使用团队模型密钥，因此此模块在线上不可运行。"
        )
        return

    foundation: DecisionFoundationState | None = st.session_state.get(_FOUNDATION_KEY)
    if foundation is None:
        sources = _render_package_selection(prepared_root)
        if sources is None:
            return
        if st.button("启动基线分析与压力测试", type="primary"):
            records, atoms = _load_package(prepared_root, sources["baseline"])
            client = LLMClient(Settings.from_sources(dotenv_path=dotenv_path))
            runner = DecisionRunner(client=client, brand_facts=[])
            with st.spinner("基线分析进行中：语料 → 候选 → 评分 → 五维压力检查…"):
                built = runner.build_baseline(CustomRunInputs(records, atoms))
            st.session_state[_RUNNER_KEY] = runner
            st.session_state[_FOUNDATION_KEY] = built
            st.session_state[_CHECKPOINTS_KEY] = []
            st.session_state[_SOURCES_KEY] = sources
            _log(f"基线完成：{len(records)} 条评论 → {len(built.candidates)} 个候选")
            st.rerun()
        return

    sources: dict[str, Any] = st.session_state.get(_SOURCES_KEY, {})
    runner: DecisionRunner | None = st.session_state.get(_RUNNER_KEY)
    if runner is None:
        st.error("运行状态缺失，请重新启动基线分析。")
        return
    st.caption(
        f"run_id={foundation.run_id} · 基线包 {sources.get('baseline')} · "
        f"可见证据 {len(foundation.visible_atoms)} 条"
    )
    _render_ranking_table(foundation)
    _render_stress_details(foundation)

    if not foundation.challenge_applied:
        if st.button(f"应用挑战批并完成压力测试（{sources['challenge']}）"):
            _, challenge_atoms = _load_package(prepared_root, sources["challenge"])
            with st.spinner("挑战批评估与候选修订进行中…"):
                challenged = runner.apply_challenge(foundation, challenge_atoms)
            st.session_state[_FOUNDATION_KEY] = challenged
            complete_workspace(st.session_state, Workspace.PRESSURE_TEST)
            _log(f"挑战批完成：{len(challenge_atoms)} 条证据进入评估")
            st.rerun()
        return

    st.success("挑战批已应用，候选修订如下；确认后进入实时决策看板。")
    for ranked in foundation.ranking.ranked_candidates:
        st.markdown(f"**{ranked.candidate.candidate_id}** · {ranked.candidate.draft_proposition}")
    if st.button("进入实时决策看板", type="primary", key="custom_enter_realtime"):
        activate_workspace(st.session_state, Workspace.REALTIME_DECISION)
        st.rerun()
    _render_log()


def _stage(foundation: DecisionFoundationState) -> str:
    if not foundation.selected_ids:
        return "select"
    if not foundation.holdout_validated:
        return "holdout"
    if not foundation.frozen:
        return "freeze"
    return "release"


def render_custom_evolution_dashboard(
    *,
    prepared_root: Path,
    online_enabled: bool,
    dotenv_path: Path,
) -> None:
    st.subheader("实时决策看板 · 自定义语料真实链路")
    if not online_enabled:
        st.info(
            "自由链路需要本地 .env 配置 LLM_API_KEY 与 LLM_MODEL；"
            "公开部署不使用团队模型密钥，因此此模块在线上不可运行。"
        )
        return

    foundation: DecisionFoundationState | None = st.session_state.get(_FOUNDATION_KEY)
    runner: DecisionRunner | None = st.session_state.get(_RUNNER_KEY)
    if foundation is None or runner is None:
        st.warning("尚未开始自由链路：请先在『叙事压力测试』工作区完成基线与挑战批。")
        return

    sources: dict[str, Any] = st.session_state.get(_SOURCES_KEY, {})
    st.caption(
        f"run_id={foundation.run_id} · 可见证据 {len(foundation.visible_atoms)} 条"
    )
    _render_ranking_table(foundation)

    stage = _stage(foundation)
    st.markdown(f"#### 决策链推进（当前阶段：{stage}）")

    if stage == "select":
        ranked = foundation.ranking.ranked_candidates
        selected = [item.candidate.candidate_id for item in ranked[:3]]
        st.write(f"自动入围（按当前排名前三）：{selected}")
        if st.button("确认选线"):
            st.session_state[_FOUNDATION_KEY] = runner.select_narratives(
                foundation, selected, primary_candidate_id=selected[0]
            )
            _log(f"选线完成：{selected}")
            st.rerun()
    elif stage == "holdout":
        if st.button(f"运行 holdout 验证（{sources['holdout']}）"):
            _, holdout_atoms = _load_package(prepared_root, sources["holdout"])
            with st.spinner("holdout 验证进行中…"):
                st.session_state[_FOUNDATION_KEY] = runner.validate_holdout(
                    foundation, holdout_atoms
                )
            _log(f"holdout 完成：{len(holdout_atoms)} 条留出证据")
            st.rerun()
    elif stage == "freeze":
        if st.button("冻结原始快照"):
            st.session_state[_FOUNDATION_KEY] = runner.freeze_original_snapshot(
                foundation
            )
            _log("原始快照已冻结")
            st.rerun()
    else:
        checkpoints: list[DecisionEvolutionCheckpoint] = st.session_state.get(
            _CHECKPOINTS_KEY, []
        )
        release_plan: list[str] = sources.get("releases", [])
        next_index = len(checkpoints) + 1
        if next_index <= len(release_plan):
            if st.button(f"释放增量批 {next_index}（{release_plan[next_index - 1]}）"):
                _, batch_atoms = _load_package(
                    prepared_root, release_plan[next_index - 1]
                )
                with st.spinner("增量批评估进行中…"):
                    checkpoint = runner.release_batch(
                        st.session_state[_FOUNDATION_KEY],
                        batch_index=next_index,
                        atoms=batch_atoms,
                    )
                checkpoints.append(checkpoint)
                st.session_state[_CHECKPOINTS_KEY] = checkpoints
                _log(f"增量批 {next_index} 完成，新增 {len(batch_atoms)} 条证据")
                st.rerun()
        else:
            st.success("全部增量批已释放，自由运行结束。")

    for checkpoint in st.session_state.get(_CHECKPOINTS_KEY, []):
        _render_checkpoint(checkpoint)
    _render_log()
