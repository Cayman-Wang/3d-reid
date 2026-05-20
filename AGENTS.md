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

**仓库级策略覆盖**：本策略在全局默认基础上，补充 `research/`、`mvp-demo/`、benchmark 结果的特定触发条件。全局默认偏积极，但本仓库仍强调文件边界和输出约束。

**核心原则**：主 agent 负责拆解、拍板、审核和最终答复；subagent 用于可拆分、低上下文、能减少主 agent 重复工作的任务。只要任务可清晰拆分、可并行推进、不会阻塞主路径，就可以优先考虑委派。

**默认策略**：
- 文档阅读、跨文件检索、路径定位、来源归因、差异汇总、旁路验证、隔离测试、边界清晰的小实现，都可以主动委派。
- 若主 agent 通过 1-2 次精确搜索/读取即可完成，或委派不会明显节省 token / 时间，则可直接由主 agent 完成。
- 主 agent 不把最终方案拍板、跨模块取舍、冲突裁决、用户最终答复交给 subagent。

**多代理并行条件**：
- 允许 2 个及以上 subagent，必须满足：
  - 独立流：任务可分割为两个或多个互不依赖的检索、核对、实现或验证流（例如 `research/` vs `mvp-demo/`、`research/` vs benchmark、`mvp-demo/` vs benchmark）
  - 低上下文：每个 subagent 只接收最小必要文件列表
  - 明确节省主 agent 工作量：避免重复搜索、核对和验证

**禁止场景**：
- 单点概念问答、一步检查、小重命名
- 主 agent 明显可以在极少量读取内完成的任务
- 需要主 agent 立即拿到结果才能继续的关键路径任务

**本仓库判例**：
- `research/` 内多计划文档检索或比对 → 1 个轻量 explorer
- `mvp-demo/` 内多脚本/多输出文件的参数追踪或来源归因 → 1 个轻量 explorer
- `research/` vs `mvp-demo/` 双侧核对 → 1-2 个轻量 explorer，由主 agent 汇总
- 单脚本修改、单文档章节更新（边界清晰）→ 1 个中等 worker（独占文件范围）
- 跨理论方案与脚本实现的复杂不一致排障 → 高模例外（必要时才用）

## Model Selection (Repository-Specific)

**主 agent**：保持高模，负责任务拆解、跨模块判断、最终审核和最终答复。

**角色与模型口径**：
- `explorer`：优先用于只读检索、文档阅读、路径定位、参数追踪、来源归因、差异汇总 → `gpt-5.4-mini`
- `worker`：优先用于局部实现、局部验证、独立文档更新、隔离验证 → `gpt-5.3-codex` 或 `gpt-5.2`
- 高风险例外：复杂排障、独立复核、跨模块验证 → `gpt-5.4`（必要时才用）

## Reasoning Effort (Repository-Optimized)

**基于任务类型的默认设置**：
- 轻量检索、文档阅读、路径定位：`low`
- 普通局部实现、局部验证：`medium`
- 跨模块复杂问题（research 理论 vs demo 实现不一致）：`high` 或 `xhigh`

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
- 实现任务：返回修改摘要，包含变更文件和影响范围
- 分析任务：返回对比表格或列表，避免冗长叙述

## Hugging Face Downloads

- In this repository, if Codex needs to download models, weights, or datasets from Hugging Face, configure the mirror first before downloading.
- In PowerShell sessions, set:
  - `$env:HF_ENDPOINT = "https://hf-mirror.com"`
- Then run commands that rely on `huggingface_hub`, `huggingface-cli`, `open_clip`, or similar Hugging Face download flows.
- If the required asset already exists locally or the tool supports an explicit file path, prefer the local file over re-downloading it.
