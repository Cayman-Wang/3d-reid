# Repository Guidelines

## Project Structure

- `research/`: authoritative plans, handoffs, reviews, and retrospectives.
- `mvp-demo/`: runnable scripts, MuJoCo assets, and benchmark/runtime outputs.
- `文献/`: paper PDFs organized by topic (example: `文献/3D重识别/`).

If you add new notes or runbooks, prefer placing decision records in `research/` and keeping runtime artifacts outside `research/`.

## Build, Test, and Development Commands

This repository is documentation-first, with runnable scripts under `mvp-demo/`. There is no unified build system or test runner checked in today.

- View Markdown: use your editor preview (for example `research/plans/tri_camera_node_3d_aware_reid/master_plan_zh.md`).
- Run the node pipeline from `mvp-demo/` as needed, for example:

```powershell
python mvp-demo\scripts\mj_capture_3cam_node.py --help
```

## CARLA-Air / AirSim Runtime Self-Start Authorization

Codex may self-start the existing CARLA-Air / AirSim runtime only when live evidence is required.

- First check whether CARLA is already online at `127.0.0.1:2000` and AirSim is already online at `127.0.0.1:41451`.
- If both ports are already reachable, do not start another runtime.
- Use only the documented runbook launcher; do not guess paths or flags.
- Write runtime logs under `local/carla_air/runtime_logs/` and PID records under `local/carla_air/runtime_pids/`.
- Wait for both ports to be ready before any live probe.
- Only stop PIDs started by Codex in the current run.
- On failure, write the blocker to `research/reports/` and do not fabricate evidence.

## Coding Style & Naming Conventions

- Filenames: prefer `snake_case` and descriptive English names.
- Papers: keep PDFs under `文献/<topic>/`; prefer stable identifiers in names (e.g., arXiv id/version).
- Markdown: use clear section headings, short paragraphs, and fenced code blocks with language tags.

## Testing & Reproducibility

- No automated tests yet. When adding pipeline steps, include a minimal "smoke check" describing expected artifacts (e.g., output folder/file names) and what success looks like.
- If you add notebooks later, they should run top-to-bottom on a clean kernel and avoid absolute paths.

## Commit & Pull Request Guidelines

Git history may not be available in this folder; use a simple convention going forward:

- Commit messages: `docs: ...`, `notes: ...`, `papers: ...` (imperative, <72 chars).
- PRs: explain the change, link relevant papers/issues, and include screenshots for diagram changes; keep notebooks' outputs minimal to reduce diff noise.

## Subagent Delegation (Repository-Specific, Token-Aware)

**仓库级策略覆盖**：本策略在全局默认基础上，补充 `research/`、`mvp-demo/`、benchmark 结果的特定触发条件。默认采用 **token-first / aggressive delegation**：优先把可拆分、低上下文、边界清晰的工作下放给 subagent，以减少主 agent（尤其 `gpt-5.5`）的 token 消耗；主 agent 保留任务拆解、关键判断、diff review、验证和最终答复。

**核心原则**：主 agent 不默认亲自执行可委派的具体操作。只要任务能清晰切分、写入范围可独占、验收命令明确，就优先派 subagent。`explorer` 交付候选事实和证据；`worker` 交付候选补丁、文档更新或隔离验证结果；主 agent 负责最终方案拍板、跨模块取舍、冲突裁决、diff review、验证和用户最终答复。

**默认策略**：
- 文档阅读、跨文件检索、路径定位、来源归因、差异汇总、旁路验证、隔离测试、边界清晰的小实现，默认主动委派。
- 跨 2 个及以上文件的查证、状态汇总、路径定位、输出证据核对，默认派 `explorer`，除非主 agent 已经在同一上下文中完成该读取。
- 文档编辑、报告同步、runbook 更新、单文件局部脚本 guard、plan-only 输出补字段、局部验证命令修复，默认派 `worker`，并声明独占文件范围和验收命令。
- 即使主 agent 可以自行完成，只要预计会消耗明显上下文、输出 token、重复检索或机械编辑，也优先委派给 subagent。
- 主 agent 不重复实现 worker 的同一任务；等待返回后只做 diff review、必要修正、整合和最终验证。
- 派发 worker 时必须说明：它并非独占整个代码库，不得 revert 他人改动，必须适配已有工作区变更。

**多代理并行条件**：
- 允许并鼓励 2 个及以上 subagent，尤其是可拆分成多个互不依赖的信息流、写入流或验证流时。必须满足：
  - 独立流：任务可分割为两个或多个互不依赖的检索、核对、实现或验证流（例如 `research/` vs `mvp-demo/`、`research/` vs benchmark、`mvp-demo/` vs benchmark）
  - 低上下文：每个 subagent 只接收最小必要文件列表
  - 写入隔离：多个 worker 的写入范围必须 disjoint，避免同一文件或同一章节并发修改
  - 主 token 节省：优先减少主 agent 重复搜索、长文阅读、机械编辑和验证输出

**禁止场景**：
- 单点概念问答、单条命令查询、小重命名
- 写入范围无法独占，或会与已知并发修改直接冲突
- 需要主 agent 立即作出架构、安全、数据真实性或跨模块取舍的关键判断
- 用户明确要求主 agent 亲自执行，或明确禁止委派

**本仓库判例**：
- `research/` 内多计划文档检索或比对 → 1 个轻量 explorer
- `mvp-demo/` 内多脚本/多输出文件的参数追踪或来源归因 → 1 个轻量 explorer
- `research/` vs `mvp-demo/` 双侧核对 → 2 个轻量 explorer，由主 agent 汇总
- 单文档章节更新、报告同步、runbook 补充（边界清晰）→ 1 个轻量 worker，独占文件或章节
- 单脚本局部修改、guard 增强、plan-only 输出补字段（边界清晰）→ 1 个中等 worker，独占脚本文件
- 跨文件但边界清晰的实现 → 按 disjoint write set 拆给 1-3 个 worker，主 agent 只整合和验收
- 独立验证、smoke、`py_compile`、输出 artifact 核对 → 可派 explorer 或 worker 旁路验证
- 跨理论方案与脚本实现的复杂不一致排障 → `gpt-5.4` 高模 subagent 优先独立复核，必要时主 agent 深入

## Model Selection (Repository-Specific)

**主 agent**：保持高模，主要负责任务拆解、跨模块判断、冲突裁决、最终审核和最终答复；避免把 `gpt-5.5` 用在可拆分的长文阅读、机械编辑、局部实现和隔离验证上。

**角色与模型口径**：
- `explorer`：只读检索、文档阅读、路径定位、参数追踪、来源归因、差异汇总 → 默认显式指定 `gpt-5.4-mini`
- 文档/Markdown worker：报告同步、runbook 更新、单文件章节更新、结构化摘要 → 默认显式指定 `gpt-5.4-mini`
- 代码 worker：局部实现、脚本 guard、plan-only 输出补字段、隔离测试修复 → 默认显式指定 `gpt-5.4-mini`
- 跨文件但边界清晰的代码修改 → 优先 `gpt-5.4`；若任务足够小且可独立收敛，仍可用 `gpt-5.4-mini`
- 高风险例外：复杂排障、独立复核、跨模块验证、research 理论 vs demo 实现不一致 → `gpt-5.4`，必要时才由主 agent 深入

## Reasoning Effort (Repository-Optimized)

**基于任务类型的默认设置**：
- explorer 轻量检索、文档阅读、路径定位：`low`
- 文档/Markdown worker、普通局部实现、局部验证：`medium`
- 机械验证、smoke 输出核对、`py_compile`/`git diff --check` 旁路验证：`low` 或 `medium`
- 跨模块复杂问题（research 理论 vs demo 实现不一致）：`high`
- `xhigh` 仅用于确有必要的高风险复核，避免默认消耗过多 token

## Delegation Boundaries (Repository-Specific)

**文件范围约束**（避免冲突）：
- 只读任务：返回"结论 + 文件路径 + 1-3 条证据"，禁止长报告
- 写任务：必须声明独占文件范围
  - `research/` 文档更新：按章节或文件独占
  - `mvp-demo/` 脚本修改：按单个脚本文件独占
  - benchmark 结果分析：按输出文件独占

**上下文最小化**：
- 搜索 `research/` 时，只传递相关计划目录路径
- 分析 `mvp-demo/` 时，只传递具体脚本或输出文件路径
- 双侧核对时，分别传递各自需要的最小文件集合

**输出格式规范**：
- 只读核查：返回简洁结论，包含关键文件路径和 1-3 个关键发现
- 实现任务：返回修改摘要，包含变更文件、影响范围、验证命令和 1-3 个需要主 agent 复核的风险点
- 分析任务：返回对比表格或列表，避免冗长叙述

**主 agent 复核契约**：
- subagent 写入结果必须由主 agent review diff 后才能视为接受。
- 主 agent 必须运行与改动相称的验证，例如 Markdown 结构检查、`py_compile`、plan-only 命令或 `git diff --check`。
- subagent 不得把 proxy annotation / proxy points 判定为最终 synthetic annotation / real geometry；此类结论只能由主 agent 在验证后给出。

## Hugging Face Downloads

- In this repository, if Codex needs to download models, weights, or datasets from Hugging Face, configure the mirror first before downloading.
- In PowerShell sessions, set:
  - `$env:HF_ENDPOINT = "https://hf-mirror.com"`
- Then run commands that rely on `huggingface_hub`, `huggingface-cli`, `open_clip`, or similar Hugging Face download flows.
- If the required asset already exists locally or the tool supports an explicit file path, prefer the local file over re-downloading it.
