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

- No automated tests yet. When adding pipeline steps, include a minimal “smoke check” describing expected artifacts (e.g., output folder/file names) and what success looks like.
- If you add notebooks later, they should run top-to-bottom on a clean kernel and avoid absolute paths.

## Commit & Pull Request Guidelines

Git history may not be available in this folder; use a simple convention going forward:

- Commit messages: `docs: ...`, `notes: ...`, `papers: ...` (imperative, <72 chars).
- PRs: explain the change, link relevant papers/issues, and include screenshots for diagram changes; keep notebooks’ outputs minimal to reduce diff noise.

## Subagent Delegation

- In this repository, Codex should automatically decide when to use subagents; do not require the user to explicitly request them.
- Default posture is balanced: prefer local execution for simple tasks, and use subagents only when parallel delegation can materially reduce main-path latency, lower decision risk, or allow bounded sidecar work to proceed independently.
- Do not use subagents for simple requests such as brief explanations, single-file trivial edits, small renames, or one-step checks.
- For medium tasks, use one subagent when a clearly bounded analysis, implementation, or verification task can proceed in parallel without blocking the main agent's next action.
- For complex or high-risk tasks, proactively use multiple subagents when the work can be partitioned into independent streams such as repository exploration, implementation in separate modules, or parallel verification.
- Do not delegate the immediate blocking critical-path task if the main agent needs that result before it can continue.
- Delegate only tasks with clear ownership and independently completable outcomes.
- When multiple subagents write code or docs, assign disjoint ownership by file or module, and do not ask them to overwrite or revert each other's changes.

## Model Selection

- Choose subagent models by task complexity.
- Use `gpt-5.1-codex-mini` for lightweight exploration, code search, narrow read-only analysis, or very small isolated edits.
- Use `gpt-5.3-codex` or `gpt-5.2-codex` for standard implementation, ordinary debugging, localized test fixes, or bounded documentation work.
- Use `gpt-5.4` for cross-cutting design, ambiguous debugging, high-risk refactors, critical validation, or tasks whose failure would significantly impact correctness.

## Reasoning Effort

- Use `low` or `medium` for simple tasks.
- Use `medium` for normal engineering work.
- Use `high` or `xhigh` for ambiguous, cross-module, or high-risk tasks.

## Delegation Boundaries

- Prefer delegating sidecar work that can run in parallel with the current local step.
- Use explorer-style subagents for codebase search, fact gathering, and narrow read-only analysis.
- Use worker-style subagents for bounded implementation, bug fixes, or verification with explicit ownership.
- The main agent remains responsible for integration, conflict resolution, and final delivery.

## Hugging Face Downloads

- In this repository, if Codex needs to download models, weights, or datasets from Hugging Face, configure the mirror first before downloading.
- In PowerShell sessions, set:
  - `$env:HF_ENDPOINT = "https://hf-mirror.com"`
- Then run commands that rely on `huggingface_hub`, `huggingface-cli`, `open_clip`, or similar Hugging Face download flows.
- If the required asset already exists locally or the tool supports an explicit local file path, prefer the local file over re-downloading it.
