# 梅见品牌叙事智能决策系统

从真实市场意见出发，经过数据预处理、双 Agent 压力测试与增量证据演化，把品牌叙事候选变成可追溯、可质疑、可迭代的决策结果。

[打开 Streamlit 在线案例](https://meijian-narrative-intelligence.streamlit.app)

> 比赛提交版本：`competition-2026-08-15`。在线站点完整开放“梅见案例展示”；在线 AI 与飞书机器人仅在本地配置后启用，公开部署不使用团队密钥。

![系统入口](assets/gateway.png)

## 三个展示板块

| 板块 | 输入 | 系统动作 | 输出 |
|---|---|---|---|
| 数据预处理 | 多平台市场评论 | 校验、筛选、分层拆分、AI 标注、证据冻结 | 五条基础品牌叙事机会 |
| 叙事压力测试 | 五条候选与冻结证据 | 五维检查、Luna 审查、DeepSeek 修订、HOLDOUT 与真人盲评 | 三条进入演化的候选 |
| 实时决策看板 | 229 条基线与四批增量证据 | 逐批更新分数、风险、排名和叙事文本 | 三支柱核心叙事与分层场景 |

![数据预处理](assets/preprocessing.png)

![叙事压力测试](assets/pressure-test.png)

![实时叙事演化](assets/evolution.png)

## 数据与证据边界

- 原始采集 484 条，筛选后有效 419 条。
- 有效集冻结拆分为 `ANALYSIS 249 / GOLD 50 / CHALLENGE_POOL 60 / HOLDOUT 60`。
- 演化展示使用 `229` 条基线与来自 ANALYSIS 的 `20` 条策展式增量，不额外制造样本。
- 案例页面回放由正式链路生成并冻结的结果，不在前端伪装实时模型推理。
- 公开仓库不发布完整 419 条评论、原始平台链接、人工标注表或模型调用过程包；只保留页面实际需要的 40 条脱敏证据摘录与冻结结果。

## 本地运行

要求 Python 3.10。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

不配置密钥也可以完整查看“梅见案例展示”。如需本地使用在线 AI 或飞书机器人：

```powershell
Copy-Item .env.example .env
# 只在本机填写 .env，严禁提交
python tools/run_feishu_bot.py
```

公开 Streamlit 不配置 `LLM_API_KEY` 或飞书凭据。缺少本地配置时，相关操作会明确显示为不可用，不会降级到伪造结果。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `app.py` | Streamlit 入口与三板块编排 |
| `src/ui/` | 系统入口、预处理、压力测试和演化界面 |
| `src/services/` | 冻结产物校验、投影与本地协同服务 |
| `data/` | 最小冻结展示包与脱敏证据目录 |
| `prompts/` | 当前运行仍引用的 Prompt |
| `tools/run_feishu_bot.py` | 仅本地配置后启动的飞书长连接入口 |
| `tests/` | 提交范围、数据边界和无密钥行为验收 |

## 安全说明

- `.env`、`.streamlit/secrets.toml`、通知数据库、日志和上传文件均被 Git 排除。
- 仓库未附开源许可证，保留全部权利。
- 品牌叙事候选是待验证决策材料，不构成事实承诺；请理性饮酒，仅限法定饮酒年龄成年人。
