from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _query_scene_name(item: dict[str, Any]) -> str:
    return Path(str(item.get("query_scene_dir", ""))).name


def _best_relevant(topk: list[dict[str, Any]]) -> dict[str, Any] | None:
    for candidate in topk:
        if candidate.get("is_relevant"):
            return candidate
    return None


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, (float, int)):
        return f"{float(value):.4f}"
    return str(value)


def _load_branch_results(eval_root: Path, branch: str) -> dict[str, Any]:
    payload_path = eval_root / branch / "all_queries_vs_all_scenes.json"
    if not payload_path.exists():
        raise SystemExit(f"Missing branch summary: {payload_path}")
    return _load_json(payload_path)


def _build_row(
    query_scene: str,
    branch: str,
    item: dict[str, Any],
    baseline_ap: float,
) -> dict[str, Any]:
    topk = list(item.get("topk") or [])
    top1 = topk[0] if topk else {}
    best_relevant = _best_relevant(topk)
    top1_score = top1.get("score")
    relevant_score = best_relevant.get("score") if best_relevant else None
    score_margin = None
    if top1_score is not None and relevant_score is not None:
        score_margin = float(top1_score) - float(relevant_score)
    top1_relevant = bool(top1.get("is_relevant"))
    return {
        "query_scene": query_scene,
        "query_identity_id": item.get("query_identity_id"),
        "branch": branch,
        "average_precision": item.get("average_precision"),
        "top1_track": top1.get("gallery_track_id"),
        "top1_relevant": top1_relevant,
        "delta_vs_rgb_only": float(item.get("average_precision", 0.0)) - float(baseline_ap),
        "score_margin": score_margin,
        "confusion_target": "-" if top1_relevant else (top1.get("gallery_identity_id") or "-"),
        "best_relevant_track": best_relevant.get("gallery_track_id") if best_relevant else None,
        "best_relevant_score": relevant_score,
        "top1_score": top1_score,
    }


def generate_failure_analysis(
    *,
    eval_root: Path,
    benchmark_id: str,
    branches: list[str],
    baseline_branch: str = "rgb_only",
) -> dict[str, Any]:
    branch_payloads = {branch: _load_branch_results(eval_root, branch) for branch in branches}
    baseline_results = {
        _query_scene_name(item): item for item in branch_payloads[str(baseline_branch)].get("results") or []
    }
    rows: list[dict[str, Any]] = []
    per_query_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for branch in branches:
        payload = branch_payloads[branch]
        branch_result_map = {_query_scene_name(item): item for item in payload.get("results") or []}
        for query_scene in sorted(baseline_results.keys()):
            item = branch_result_map.get(query_scene)
            if item is None:
                raise SystemExit(f"Missing query {query_scene} in branch {branch}")
            baseline_ap = float(baseline_results[query_scene].get("average_precision") or 0.0)
            row = _build_row(query_scene=query_scene, branch=branch, item=item, baseline_ap=baseline_ap)
            rows.append(row)
            per_query_rows.setdefault(query_scene, {})[branch] = row

    branch_stats: list[dict[str, Any]] = []
    baseline_by_scene = per_query_rows
    for branch in branches:
        branch_rows = [row for row in rows if row["branch"] == branch]
        improved = 0
        regressed = 0
        unchanged = 0
        fixed_top1 = 0
        lost_top1 = 0
        correct_top1 = 0
        for row in branch_rows:
            delta = float(row["delta_vs_rgb_only"])
            if delta > 1e-9:
                improved += 1
            elif delta < -1e-9:
                regressed += 1
            else:
                unchanged += 1
            baseline_row = baseline_by_scene[row["query_scene"]][str(baseline_branch)]
            if row["top1_relevant"]:
                correct_top1 += 1
            if (not baseline_row["top1_relevant"]) and row["top1_relevant"]:
                fixed_top1 += 1
            if baseline_row["top1_relevant"] and (not row["top1_relevant"]):
                lost_top1 += 1
        branch_stats.append(
            {
                "branch": branch,
                "correct_top1_queries": correct_top1,
                "ap_improved_queries": improved,
                "ap_regressed_queries": regressed,
                "ap_unchanged_queries": unchanged,
                "top1_fixed_vs_rgb_only": fixed_top1,
                "top1_lost_vs_rgb_only": lost_top1,
            }
        )

    gt_recovered_queries = []
    persistent_prediction_failures = []
    geometry_regressions = []
    for query_scene in sorted(per_query_rows.keys()):
        baseline_row = per_query_rows[query_scene][str(baseline_branch)]
        pred_row = per_query_rows[query_scene].get("rgb_predicted_depth_geometry")
        fused_row = per_query_rows[query_scene].get("rgb_fused_geometry")
        gt_row = per_query_rows[query_scene].get("gt_upper_bound")

        if gt_row and (not baseline_row["top1_relevant"]) and gt_row["top1_relevant"]:
            gt_recovered_queries.append(query_scene)

        if (
            pred_row
            and fused_row
            and gt_row
            and (not pred_row["top1_relevant"])
            and (not fused_row["top1_relevant"])
            and gt_row["top1_relevant"]
        ):
            persistent_prediction_failures.append(query_scene)

        if baseline_row["top1_relevant"]:
            regressed_branches = []
            if pred_row and (not pred_row["top1_relevant"]):
                regressed_branches.append("rgb_predicted_depth_geometry")
            if fused_row and (not fused_row["top1_relevant"]):
                regressed_branches.append("rgb_fused_geometry")
            if regressed_branches:
                geometry_regressions.append({"query_scene": query_scene, "branches": regressed_branches})

    output_payload = {
        "benchmark_id": str(benchmark_id),
        "baseline_branch": str(baseline_branch),
        "branch_stats": branch_stats,
        "rows": rows,
        "query_groups": {
            "gt_recovered_queries": gt_recovered_queries,
            "persistent_prediction_failures": persistent_prediction_failures,
            "geometry_regressions": geometry_regressions,
        },
    }
    (eval_root / "query_failure_analysis.json").write_text(
        json.dumps(output_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    md_lines = [
        f"# {benchmark_id} query failure analysis",
        "",
        f"- baseline: `{baseline_branch}`",
        "- `score_margin = top1_score - correct_match_score`，`0` 表示 top1 已命中正确匹配。",
        "",
        "## Branch Delta Summary",
        "",
        "| branch | correct_top1_queries | ap_improved | ap_regressed | ap_unchanged | top1_fixed | top1_lost |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in branch_stats:
        md_lines.append(
            "| {branch} | {correct_top1_queries} | {ap_improved_queries} | {ap_regressed_queries} | {ap_unchanged_queries} | {top1_fixed_vs_rgb_only} | {top1_lost_vs_rgb_only} |".format(
                **row
            )
        )

    md_lines.extend(
        [
            "",
            "## Key Query Groups",
            "",
            f"- `gt_recovered_queries`: {', '.join(gt_recovered_queries) if gt_recovered_queries else '-'}",
            "- `persistent_prediction_failures`: {items}".format(
                items=", ".join(persistent_prediction_failures) if persistent_prediction_failures else "-"
            ),
            "- `geometry_regressions`: {items}".format(
                items=", ".join(
                    f"{item['query_scene']} ({'/'.join(item['branches'])})" for item in geometry_regressions
                )
                if geometry_regressions
                else "-"
            ),
            "",
            "## Query Diff Rows",
            "",
            "| query_scene | branch | AP | top1_track | top1_relevant | delta_vs_rgb_only | score_margin | confusion_target |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        md_lines.append(
            "| {query_scene} | {branch} | {average_precision} | {top1_track} | {top1_relevant} | {delta_vs_rgb_only} | {score_margin} | {confusion_target} |".format(
                query_scene=row["query_scene"],
                branch=row["branch"],
                average_precision=_fmt_metric(row["average_precision"]),
                top1_track=row["top1_track"] or "-",
                top1_relevant=_fmt_metric(row["top1_relevant"]),
                delta_vs_rgb_only=_fmt_metric(row["delta_vs_rgb_only"]),
                score_margin=_fmt_metric(row["score_margin"]),
                confusion_target=row["confusion_target"],
            )
        )

    (eval_root / "query_failure_analysis.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return output_payload


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate query-level failure analysis artifacts for ICISCAE branch results."
    )
    ap.add_argument("--benchmark_id", default="iciscae_node01_uav_v3_clean", type=str)
    ap.add_argument("--baseline_branch", default="rgb_only", type=str)
    ap.add_argument(
        "--branches",
        nargs="+",
        default=["rgb_only", "rgb_predicted_depth_geometry", "rgb_fused_geometry", "gt_upper_bound"],
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    eval_root = repo_root / "mvp-demo" / "output" / "evals" / str(args.benchmark_id)
    generate_failure_analysis(
        eval_root=eval_root,
        benchmark_id=str(args.benchmark_id),
        branches=[str(branch) for branch in args.branches],
        baseline_branch=str(args.baseline_branch),
    )


if __name__ == "__main__":
    main()
