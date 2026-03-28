from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _query_scene_name(item: dict[str, Any]) -> str:
    return Path(str(item.get("query_scene_dir", ""))).name


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, (float, int)):
        return f"{float(value):.4f}"
    return str(value)


def _average_precision(relevant: list[bool]) -> float | None:
    if not relevant or not any(relevant):
        return None
    hits = 0
    precision_sum = 0.0
    for idx, is_rel in enumerate(relevant, start=1):
        if is_rel:
            hits += 1
            precision_sum += hits / float(idx)
    return precision_sum / float(hits)


def _load_branch_payload(eval_root: Path, branch: str) -> dict[str, Any]:
    path = eval_root / str(branch) / "all_queries_vs_all_scenes.json"
    if not path.exists():
        raise SystemExit(f"Missing branch summary: {path}")
    return _load_json(path)


def _build_gallery_map(item: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for hit in item.get("topk") or []:
        gallery_track_id = str(hit.get("gallery_track_id", ""))
        if not gallery_track_id:
            continue
        result[gallery_track_id] = {
            "score": float(hit.get("score", 0.0)),
            "gallery_scene_dir": str(hit.get("gallery_scene_dir", "")),
            "gallery_identity_id": hit.get("gallery_identity_id"),
            "is_relevant": bool(hit.get("is_relevant")),
        }
    return result


def _fuse_query_item(
    *,
    base_item: dict[str, Any],
    fusion_item: dict[str, Any],
    alpha: float,
    topk: int,
) -> dict[str, Any]:
    base_gallery = _build_gallery_map(base_item)
    fusion_gallery = _build_gallery_map(fusion_item)
    if set(base_gallery.keys()) != set(fusion_gallery.keys()):
        raise SystemExit(
            f"Gallery mismatch for query {_query_scene_name(base_item)}: "
            f"{sorted(base_gallery.keys())} vs {sorted(fusion_gallery.keys())}"
        )

    ranked = []
    for gallery_track_id, base_hit in base_gallery.items():
        fusion_hit = fusion_gallery[gallery_track_id]
        if bool(base_hit["is_relevant"]) != bool(fusion_hit["is_relevant"]):
            raise SystemExit(f"Relevance mismatch for query {_query_scene_name(base_item)} track {gallery_track_id}")
        score = (1.0 - alpha) * float(base_hit["score"]) + alpha * float(fusion_hit["score"])
        ranked.append(
            {
                "gallery_track_id": gallery_track_id,
                "gallery_scene_dir": base_hit["gallery_scene_dir"],
                "gallery_identity_id": base_hit["gallery_identity_id"],
                "is_relevant": bool(base_hit["is_relevant"]),
                "score": float(score),
            }
        )
    ranked.sort(key=lambda item: float(item["score"]), reverse=True)
    ranked_topk = []
    for rank, item in enumerate(ranked[:topk], start=1):
        ranked_topk.append(
            {
                "rank": rank,
                "score": float(item["score"]),
                "gallery_track_id": item["gallery_track_id"],
                "gallery_scene_dir": item["gallery_scene_dir"],
                "gallery_identity_id": item["gallery_identity_id"],
                "is_relevant": bool(item["is_relevant"]),
            }
        )

    return {
        "query_track_id": base_item.get("query_track_id"),
        "query_scene_dir": base_item.get("query_scene_dir"),
        "query_identity_id": base_item.get("query_identity_id"),
        "topk": ranked_topk,
        "average_precision": _average_precision([bool(item["is_relevant"]) for item in ranked]),
    }


def _summarize_results(results: list[dict[str, Any]], *, branch_name: str) -> dict[str, Any]:
    ap_values = [float(item["average_precision"]) for item in results if item.get("average_precision") is not None]
    metric_queries = len(ap_values)
    recall_hits = {1: 0, 5: 0, 10: 0}
    for item in results:
        if item.get("average_precision") is None:
            continue
        relevant_flags = [bool(hit.get("is_relevant")) for hit in item.get("topk") or []]
        for k in recall_hits:
            recall_hits[k] += int(any(relevant_flags[:k]))
    return {
        "branch": branch_name,
        "num_queries": len(results),
        "num_gallery": len((results[0].get("topk") or [])) if results else 0,
        "metric_queries": metric_queries,
        "mAP": (sum(ap_values) / float(metric_queries)) if metric_queries else None,
        "recall_at_1": (recall_hits[1] / float(metric_queries)) if metric_queries else None,
        "recall_at_5": (recall_hits[5] / float(metric_queries)) if metric_queries else None,
        "recall_at_10": (recall_hits[10] / float(metric_queries)) if metric_queries else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze score-level late fusion between frozen ICISCAE branches.")
    ap.add_argument("--benchmark_id", default="iciscae_node01_uav_v3_clean", type=str)
    ap.add_argument("--base_branch", default="rgb_only", type=str)
    ap.add_argument("--fusion_branch", default="rgb_fused_geometry", type=str)
    ap.add_argument("--alphas", nargs="+", default=["0.25", "0.5", "0.75"], type=str)
    ap.add_argument("--topk", default=5, type=int)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    eval_root = repo_root / "mvp-demo" / "output" / "evals" / str(args.benchmark_id)
    base_payload = _load_branch_payload(eval_root, str(args.base_branch))
    fusion_payload = _load_branch_payload(eval_root, str(args.fusion_branch))

    base_results = {_query_scene_name(item): item for item in base_payload.get("results") or []}
    fusion_results = {_query_scene_name(item): item for item in fusion_payload.get("results") or []}
    if set(base_results.keys()) != set(fusion_results.keys()):
        raise SystemExit(f"Query mismatch: {sorted(base_results.keys())} vs {sorted(fusion_results.keys())}")

    out_dir = eval_root / f"late_fusion_{args.base_branch}_{args.fusion_branch}"
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows = [
        {
            "branch": str(args.base_branch),
            "alpha": None,
            **dict(base_payload.get("summary") or {}),
        },
        {
            "branch": str(args.fusion_branch),
            "alpha": None,
            **dict(fusion_payload.get("summary") or {}),
        },
    ]

    for alpha_str in args.alphas:
        alpha = float(alpha_str)
        branch_name = f"late_fusion_a{alpha_str}"
        fused_results = []
        for query_scene in sorted(base_results.keys()):
            fused_results.append(
                _fuse_query_item(
                    base_item=base_results[query_scene],
                    fusion_item=fusion_results[query_scene],
                    alpha=alpha,
                    topk=max(1, int(args.topk)),
                )
            )
        summary = _summarize_results(fused_results, branch_name=branch_name)
        payload = {
            "config": {
                "benchmark_id": str(args.benchmark_id),
                "base_branch": str(args.base_branch),
                "fusion_branch": str(args.fusion_branch),
                "alpha": alpha,
                "score_formula": "score = (1 - alpha) * base_score + alpha * fusion_score",
            },
            "summary": summary,
            "results": fused_results,
        }
        out_path = out_dir / f"{branch_name}_all_queries_vs_all_scenes.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        comparison_rows.append({"branch": branch_name, "alpha": alpha, **summary})

    comparison_payload = {
        "benchmark_id": str(args.benchmark_id),
        "base_branch": str(args.base_branch),
        "fusion_branch": str(args.fusion_branch),
        "rows": comparison_rows,
    }
    (out_dir / "late_fusion_summary.json").write_text(
        json.dumps(comparison_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    md_lines = [
        f"# {args.benchmark_id} late fusion summary",
        "",
        f"- base branch: `{args.base_branch}`",
        f"- fusion branch: `{args.fusion_branch}`",
        "- score formula: `score = (1 - alpha) * base_score + alpha * fusion_score`",
        "",
        "## Summary",
        "",
        "| branch | alpha | mAP | recall@1 | recall@5 | recall@10 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in comparison_rows:
        md_lines.append(
            "| {branch} | {alpha} | {mAP} | {recall_at_1} | {recall_at_5} | {recall_at_10} |".format(
                branch=row["branch"],
                alpha=_fmt_metric(row.get("alpha")),
                mAP=_fmt_metric(row.get("mAP")),
                recall_at_1=_fmt_metric(row.get("recall_at_1")),
                recall_at_5=_fmt_metric(row.get("recall_at_5")),
                recall_at_10=_fmt_metric(row.get("recall_at_10")),
            )
        )
    (out_dir / "late_fusion_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
