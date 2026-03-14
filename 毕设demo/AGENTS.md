# Repository Guidelines

## Project Structure

- `3D重建-3Dreid/`: 3D ReID pipeline notes and prototypes (`*.md`, `*.ipynb`).
- `文献/`: paper PDFs organized by topic (example: `文献/3D重识别/`).

If you add runnable code later, keep it separate from papers/notes (recommended: `src/`, `scripts/`, `tests/`, `assets/`).

## Build, Test, and Development Commands

This repository is primarily documentation; there is no build system or test runner checked in today.

- View Markdown: use your editor preview (e.g., open `3D重建-3Dreid/rgbd_3d_reid_pipeline_routes.md`).
- Run the notebook (if you have Jupyter installed):

```powershell
jupyter lab
```

- Optional: export a notebook for review in PRs:

```powershell
jupyter nbconvert --to markdown 3D重建-3Dreid/rgbd_3d_reid_pipeline_routes.ipynb
```

## Coding Style & Naming Conventions

- Filenames: prefer `snake_case` and descriptive English names (match existing patterns like `rgbd_3d_reid_pipeline_routes.md`).
- Papers: keep PDFs under `文献/<topic>/`; prefer stable identifiers in names (e.g., arXiv id/version).
- Markdown: use clear section headings, short paragraphs, and fenced code blocks with language tags.

## Testing & Reproducibility

- No automated tests yet. When adding pipeline steps, include a minimal “smoke check” describing expected artifacts (e.g., output folder/file names) and what success looks like.
- Notebooks should run top-to-bottom on a clean kernel and avoid absolute paths (use relative paths under the repo).

## Commit & Pull Request Guidelines

Git history may not be available in this folder; use a simple convention going forward:

- Commit messages: `docs: ...`, `notes: ...`, `papers: ...` (imperative, <72 chars).
- PRs: explain the change, link relevant papers/issues, and include screenshots for diagram changes; keep notebooks’ outputs minimal to reduce diff noise.

