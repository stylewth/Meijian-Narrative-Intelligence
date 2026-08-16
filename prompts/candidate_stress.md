<!-- prompt-version: v1 -->

# Candidate Stress 语义攻击约束

你只处理输入中给出的候选和 `attack_evidence`。返回一个 `StressCheckResult`：

- `check_type` 必须等于输入值；
- 每条攻击必须明确说明“哪条 evidence_id 如何攻击候选的哪项主张”；
- `reference_ids` 只能使用 `attack_evidence` 中实际存在的 evidence_id，不能补造来源、竞品表现、比例或分数；
- 证据不足时返回 `COMPLETED + REVISE`，说明最小修订边界；
- 不得把高 AI confidence 当作高 Evidence Grade；
- 不得在 BrandFact、EvidenceAtom 或输入候选之外补充品牌、产品或竞品事实；
- 不得输出任何伪精确的成功概率、竞品评分或预期提升。
