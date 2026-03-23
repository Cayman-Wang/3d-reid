from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_iciscae_failure_modes import generate_failure_analysis


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _query_scene_name(item: dict) -> str:
    return Path(str(item.get("query_scene_dir", ""))).name


def _fmt_metric(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, (float, int)):
        return f"{float(value):.4f}"
    return str(value)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize branch-level ICISCAE eval results into JSON + Markdown.")
    ap.add_argument("--benchmark_id", default="iciscae_node01_uav_v3_clean", type=str)
    ap.add_argument(
        "--branches",
        nargs="+",
        default=["rgb_only", "rgb_predicted_depth_geometry", "rgb_fused_geometry", "gt_upper_bound"],
    )
    ap.add_argument("--baseline_branch", default="rgb_only", type=str)
    ap.add_argument("--skip_failure_analysis", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    eval_root = repo_root / "mvp-demo" / "output" / "evals" / str(args.benchmark_id)
    branch_payloads: dict[str, dict] = {}
    for branch in args.branches:
        payload_path = eval_root / str(branch) / "all_queries_vs_all_scenes.json"
        if not payload_path.exists():
            raise SystemExit(f"Missing branch summary: {payload_path}")
        branch_payloads[str(branch)] = _load_json(payload_path)

    summary_rows = []
    per_query: dict[str, dict[str, dict]] = {}
    for branch, payload in branch_payloads.items():
        summary = dict(payload.get("summary") or {})
        summary_rows.append(
            {
                "branch": branch,
                "mAP": summary.get("mAP"),
                "recall_at_1": summary.get("recall_at_1"),
                "recall_at_5": summary.get("recall_at_5"),
                "recall_at_10": summary.get("recall_at_10"),
                "num_queries": summary.get("num_queries"),
                "num_gallery": summary.get("num_gallery"),
                "metric_queries": summary.get("metric_queries"),
            }
        )
        for item in payload.get("results") or []:
            scene_name = _query_scene_name(item)
            top1 = (item.get("topk") or [None])[0] or {}
            per_query.setdefault(scene_name, {})[branch] = {
                "average_precision": item.get("average_precision"),
                "top1_relevant": top1.get("is_relevant"),
                "top1_track": top1.get("gallery_track_id"),
                "top1_score": top1.get("score"),
            }

    output_payload = {"benchmark_id": str(args.benchmark_id), "summary_rows": summary_rows, "per_query": per_query}
    (eval_root / "branch_comparison_summary.json").write_text(
        json.dumps(output_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    md_lines = [
        f"# {args.benchmark_id} branch comparison",
        "",
        "## Summary",
        "",
        "| branch | mAP | recall@1 | recall@5 | recall@10 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        md_lines.append(
            "| {branch} | {mAP} | {recall_at_1} | {recall_at_5} | {recall_at_10} |".format(
                branch=row["branch"],
                mAP=_fmt_metric(row["mAP"]),
                recall_at_1=_fmt_metric(row["recall_at_1"]),
                recall_at_5=_fmt_metric(row["recall_at_5"]),
                recall_at_10=_fmt_metric(row["recall_at_10"]),
            )
        )

    md_lines.extend(["", "## Per-query top1 comparison", "", "| query_scene | branch | AP | top1_relevant | top1_track | top1_score |", "| --- | --- | --- | --- | --- | --- |"])
    for scene_name in sorted(per_query.keys()):
        for branch in args.branches:
            row = per_query.get(scene_name, {}).get(branch, {})
            md_lines.append(
                "| {scene} | {branch} | {ap} | {rel} | {track} | {score} |".format(
                    scene=scene_name,
                    branch=branch,
                    ap=_fmt_metric(row.get("average_precision")),
                    rel=_fmt_metric(row.get("top1_relevant")),
                    track=row.get("top1_track") or "-",
                    score=_fmt_metric(row.get("top1_score")),
                )
            )

    (eval_root / "branch_comparison_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    if not bool(args.skip_failure_analysis):
        generate_failure_analysis(
            eval_root=eval_root,
            benchmark_id=str(args.benchmark_id),
            branches=[str(branch) for branch in args.branches],
            baseline_branch=str(args.baseline_branch),
        )


if __name__ == "__main__":
    main()
