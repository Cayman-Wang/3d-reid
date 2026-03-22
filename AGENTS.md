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

- In this repository, Codex may automatically use subagents when parallel delegation can clearly speed up the task or reduce blocking on the main path.
- Do not force subagent use for simple tasks; prefer local execution when delegation adds little value.
