<!-- prompt-version: v1 -->

# 增量证据影响判定

根据输入的新增 `evidence_atoms`，逐条输出对现有冲突或候选的影响。输出 JSON 对象：

```json
{"proposals":[{"evidence_id":"...","impact":"SUPPORT|CHALLENGE|NEW_SIGNAL|NEUTRAL","target_id":"... 或 null","reason":"..."}]}
```

规则：

- 每一个输入 `evidence_id` 必须且只能输出一次；不得遗漏、重复或虚构 ID。
- `SUPPORT` 和 `CHALLENGE` 必须绑定输入 `conflict_ids` 或 `candidate_ids` 中已存在的 `target_id`。
- `NEW_SIGNAL` 和 `NEUTRAL` 的 `target_id` 必须为 `null`。
- 只基于输入的事实和证据判断；不得生成候选、修订候选、评分或补写证据。
- `reason` 简洁解释关联，不得捏造输入外的事实。

HOLDOUT 模式会使用不同的输出结构：每条证据只能对已冻结候选产生 `SUPPORT` 或 `CHALLENGE`，不得生成、修订或重新评分候选。
