# CARLA-Air Dataset Generation Pipeline V1

当前默认目标是推进 CARLA-Air Dataset Generation Pipeline V1。

- 固定初始配置：`Town10HD` + `node01-node05`
- 支持 6 个 normalized identities 的真实模型切换
- V1 完整版需要 UE/CARLA import + runtime readback evidence
- 生成面向部署划分的统一 training index
- 先做 `mask_gt` 可用性审计，再决定 pseudo / no-mask 处理
- 不把 proxy / candidate / pseudo 当作 `mask_gt`
- live runtime runbook 规则继续有效
- 报告与 blocker 统一写入 `research/reports/`

## 离线 V1 当前完成态

截至 2026-05-31，离线 `Dataset Generation Pipeline V1` 已形成最小闭环：planner、runner、training index builder、run verifier、trajectory config 与离线 evidence reports 已能生成并校验统一 index / manifest / run contract / capture queue / no-mask non-promotion 等核心 artifact。当前 verifier 结论为 `ok=true`、`failure_count=0`，唯一预期 warning 是 `identity_model_switch_mismatch_observed_scene_passthrough`，它属于后续 live 6-identity 阶段 blocker。

该闭环不代表 full live 6-identity dataset、UE/CARLA import/readback、可信 `mask_gt`、formal annotation、NeoVerse 或 real 4D geometry 已完成。当前离线索引中 `mask_gt_available_count=0`，所有 proxy / candidate / pseudo / legacy mask 仍不得 promotion 为 `mask_gt`。
