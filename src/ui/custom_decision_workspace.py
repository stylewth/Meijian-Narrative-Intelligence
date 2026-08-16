"""自由决策工作区：自定义语料在本地模型配置下实时运行完整决策链。

线上公开部署只回放冻结案例；本工作区面向本地配置了 ``.env`` 模型密钥的
使用者，从已发布的 PreparedCorpusPackage 出发，驱动 DecisionRunner 走完
基线 → 挑战 → 选线 → holdout → 冻结 → 增量批 的真实推理链路。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import Settings
from src.llm_client import LLMClient
from src.schemas import (
    BrandFact,
    CommentRecord,
    DecisionEvolutionCheckpoint,
    DecisionFoundationState,
    EvidenceAtom,
)
from src.services.decision_runner import DecisionRunner
from src.services.prepared_corpus import list_prepared_corpora
from src.ui import SystemEntry

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
                "压力检查通过": passed,
            }
        )
    st.dataframe(rows, use_container_width=True)


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


def _stage(foundation: DecisionFoundationState) -> str:
    if not foundation.challenge_applied:
        return "challenge"
    if not foundation.selected_ids:
        return "select"
    if not foundation.holdout_validated:
        return "holdout"
    if not foundation.frozen:
        return "freeze"
    return "release"


def render_custom_decision_workspace(
    *,
    prepared_root: Path,
    entry: Any,
    online_enabled: bool,
    dotenv_path: Path,
) -> None:
    st.subheader("自由决策实验")
    if entry is not SystemEntry.CUSTOM:
        st.info(
            "自由决策实验面向自定义语料：请从入口选择『进入 · 自定义使用』，"
            "并在本地配置模型密钥后运行完整推理链路。"
        )
        return
    if not online_enabled:
        st.info(
            "自由决策需要本地 .env 配置 LLM_API_KEY 与 LLM_MODEL；"
            "公开部署不使用团队模型密钥，因此此模块在线上不可运行。"
        )
        return

    packages = list_prepared_corpora(prepared_root)
    if not packages:
        st.warning(
            "尚未发布任何自定义语料包：请先在『数据预处理』工作区完成导入、"
            "标注导入与发布，再回到此处运行决策链。"
        )
        return
    options = {published.manifest.package_id: published for published in packages}
    label = lambda package_id: (  # noqa: E731
        f"{package_id} · {len(options[package_id].package.records)} 条评论 / "
        f"{len(options[package_id].package.evidence_atoms)} 条证据"
    )

    st.markdown("#### 第 1 步 · 装载语料")
    baseline_id = st.selectbox("基线语料包", list(options), format_func=label)
    other_ids = [package_id for package_id in options if package_id != baseline_id]
    if len(other_ids) < 2:
        st.warning(
            "完整决策链要求挑战批与 holdout 批语料：请再发布至少两个语料包"
            "（可在预处理工作区导入不同批次数据后分别发布）。"
        )
        return
    challenge_id = st.selectbox("挑战批语料包", other_ids)
    holdout_candidates = [pid for pid in other_ids if pid != challenge_id]
    holdout_id = st.selectbox("holdout 语料包", holdout_candidates)
    release_ids = st.multiselect(
        "增量批语料包（按顺序释放，可选）",
        [pid for pid in other_ids if pid not in {challenge_id, holdout_id}],
    )

    brand_facts_text = st.text_area(
        "品牌事实 BrandFact 列表（JSON 数组，可选；留空则不注入接地约束）",
        height=120,
    )

    if st.button("装载语料并启动基线分析", type="primary"):
        records, atoms = _load_package(prepared_root, baseline_id)
        brand_facts: list[BrandFact] = []
        if brand_facts_text.strip():
            brand_facts = [
                BrandFact.model_validate(item) for item in json.loads(brand_facts_text)
            ]
        client = LLMClient(Settings.from_sources(dotenv_path=dotenv_path))
        runner = DecisionRunner(client=client, brand_facts=brand_facts)
        foundation = runner.build_baseline(CustomRunInputs(records, atoms))
        st.session_state[_RUNNER_KEY] = runner
        st.session_state[_FOUNDATION_KEY] = foundation
        st.session_state[_CHECKPOINTS_KEY] = []
        st.session_state[_SOURCES_KEY] = {
            "baseline": baseline_id,
            "challenge": challenge_id,
            "holdout": holdout_id,
            "releases": list(release_ids),
        }
        _log(f"基线完成：{len(records)} 条评论 → {len(foundation.candidates)} 个候选")
        st.rerun()

    runner: DecisionRunner | None = st.session_state.get(_RUNNER_KEY)
    foundation: DecisionFoundationState | None = st.session_state.get(_FOUNDATION_KEY)
    if runner is None or foundation is None:
        return

    sources: dict[str, Any] = st.session_state.get(_SOURCES_KEY, {})
    st.caption(
        f"运行中 run_id={foundation.run_id} · 基线包 {sources.get('baseline')} · "
        f"可见证据 {len(foundation.visible_atoms)} 条"
    )
    _render_ranking_table(foundation)

    stage = _stage(foundation)
    st.markdown(f"#### 第 2 步 · 决策链推进（当前阶段：{stage}）")

    if stage == "challenge":
        if st.button(f"应用挑战批（{sources['challenge']}）"):
            _, challenge_atoms = _load_package(prepared_root, sources["challenge"])
            st.session_state[_FOUNDATION_KEY] = runner.apply_challenge(
                foundation, challenge_atoms
            )
            _log(f"挑战批完成：{len(challenge_atoms)} 条证据进入评估")
            st.rerun()
    elif stage == "select":
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

    log = st.session_state.get(_LOG_KEY, [])
    if log:
        with st.expander("运行日志", expanded=False):
            for line in log:
                st.write(line)
