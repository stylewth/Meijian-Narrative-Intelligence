<!-- prompt-version: v1 -->

# 局部叙事修订

根据当前候选与当前新增批次的证据，判断是否只修订允许字段。输出 JSON 对象：

```json
{"revision":{"candidate_id":"...","draft_proposition":"...","main_scenes":["..."],"competitor_difference":"...","trigger_evidence_ids":["..."],"reason":"..."}}
```

规则：

- `candidate_id` 必须精确等于输入候选 ID，不得新建或切换候选。
- 只能输出 `draft_proposition`、`main_scenes`、`competitor_difference` 及所列审计字段；不得改动候选的任何其他字段。
- `trigger_evidence_ids` 必须全部来自当前新增批次，且至少引用一条真实证据。
- 不得输出 `before`、`after`、`added_phrases` 或 `removed_phrases`；程序会从旧、新文本计算逐词差异。
- 没有可靠的新增证据触发时，保持三个允许字段完全原样。
