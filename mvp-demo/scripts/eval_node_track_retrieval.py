from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _l2norm_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.size == 0:
        return x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


def _load_scene_embeddings(scene_dir: Path) -> tuple[np.ndarray, list[dict]]:
    emb_path = scene_dir / "embeddings" / "tracks.npy"
    meta_path = scene_dir / "embeddings" / "tracks_meta.json"
    if not emb_path.exists() or not meta_path.exists():
        raise SystemExit(f"Missing embeddings in scene: {scene_dir}")
    embs = np.load(str(emb_path)).astype(np.float32)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, list):
        raise SystemExit(f"Invalid track metadata: {meta_path}")
    for item in meta:
        if isinstance(item, dict):
            item.setdefault("scene_dir", str(scene_dir))
    return embs, meta


def _average_precision(relevant: np.ndarray) -> float:
    if relevant.size == 0 or not np.any(relevant):
        return float("nan")
    precision_sum = 0.0
    hits = 0
    for idx, is_rel in enumerate(relevant.tolist(), start=1):
        if is_rel:
            hits += 1
            precision_sum += hits / float(idx)
    return float(precision_sum / hits)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate cosine retrieval over node-level track embeddings.")
    ap.add_argument("--query_scene_dir", action="append", default=[], type=str, help="Repeatable")
    ap.add_argument("--gallery_scene_dir", action="append", default=[], type=str, help="Repeatable")
    ap.add_argument("--topk", default=5, type=int)
    ap.add_argument("--exclude_same_track_id", action="store_true")
    ap.add_argument("--exclude_same_scene", action="store_true")
    ap.add_argument("--out", default="", type=str, help="Optional results JSON path")
    args = ap.parse_args()

    q_scenes = [Path(s).resolve() for s in (args.query_scene_dir or [])]
    g_scenes = [Path(s).resolve() for s in (args.gallery_scene_dir or [])]
    if not q_scenes:
        raise SystemExit("Provide at least one --query_scene_dir")
    if not g_scenes:
        raise SystemExit("Provide at least one --gallery_scene_dir")

    q_embs_all: list[np.ndarray] = []
    q_meta_all: list[dict] = []
    for scene_dir in q_scenes:
        embs, meta = _load_scene_embeddings(scene_dir)
        q_embs_all.append(_l2norm_rows(embs))
        q_meta_all.extend(meta)

    g_embs_all: list[np.ndarray] = []
    g_meta_all: list[dict] = []
    for scene_dir in g_scenes:
        embs, meta = _load_scene_embeddings(scene_dir)
        g_embs_all.append(_l2norm_rows(embs))
        g_meta_all.extend(meta)

    q_embs = np.concatenate(q_embs_all, axis=0) if q_embs_all else np.zeros((0, 0), dtype=np.float32)
    g_embs = np.concatenate(g_embs_all, axis=0) if g_embs_all else np.zeros((0, 0), dtype=np.float32)
    if q_embs.size == 0 or g_embs.size == 0:
        raise SystemExit("Empty embeddings (no tracks).")
    if q_embs.shape[1] != g_embs.shape[1]:
        raise SystemExit(f"Dim mismatch: query D={q_embs.shape[1]} vs gallery D={g_embs.shape[1]}")

    scores = q_embs @ g_embs.T
    topk = max(1, int(args.topk))

    all_results: list[dict] = []
    ap_values: list[float] = []
    recall_hits = {1: 0, 5: 0, 10: 0}
    metric_queries = 0

    for qi, q_meta in enumerate(q_meta_all):
        q_track_id = str(q_meta.get("track_id", f"q{qi}"))
        q_scene_dir = str(q_meta.get("scene_dir", ""))
        q_identity_id = q_meta.get("identity_id")
        row = scores[qi]
        order = np.argsort(-row)

        filtered_idx: list[int] = []
        relevant_flags: list[bool] = []
        for gi in order.tolist():
            g_meta = g_meta_all[int(gi)]
            g_track_id = str(g_meta.get("track_id", f"g{gi}"))
            g_scene_dir = str(g_meta.get("scene_dir", ""))
            if args.exclude_same_track_id and g_track_id == q_track_id:
                continue
            if args.exclude_same_scene and g_scene_dir == q_scene_dir:
                continue
            filtered_idx.append(int(gi))
            relevant_flags.append(bool(q_identity_id is not None and g_meta.get("identity_id") == q_identity_id))

        top_hits = []
        for rank, gi in enumerate(filtered_idx[:topk], start=1):
            g_meta = g_meta_all[gi]
            top_hits.append(
                {
                    "rank": rank,
                    "score": float(row[gi]),
                    "gallery_track_id": str(g_meta.get("track_id", f"g{gi}")),
                    "gallery_scene_dir": str(g_meta.get("scene_dir", "")),
                    "gallery_identity_id": g_meta.get("identity_id"),
                    "is_relevant": bool(relevant_flags[rank - 1]),
                }
            )

        result = {
            "query_track_id": q_track_id,
            "query_scene_dir": q_scene_dir,
            "query_identity_id": q_identity_id,
            "topk": top_hits,
        }

        relevant = np.asarray(relevant_flags, dtype=bool)
        if q_identity_id is not None and np.any(relevant):
            metric_queries += 1
            ap = _average_precision(relevant)
            ap_values.append(ap)
            for k in recall_hits:
                recall_hits[k] += int(np.any(relevant[:k]))
            result["average_precision"] = ap
        else:
            result["average_precision"] = None

        all_results.append(result)

    summary = {
        "num_queries": len(q_meta_all),
        "num_gallery": len(g_meta_all),
        "metric_queries": metric_queries,
        "mAP": float(np.nanmean(ap_values)) if ap_values else None,
        "recall_at_1": (recall_hits[1] / metric_queries) if metric_queries else None,
        "recall_at_5": (recall_hits[5] / metric_queries) if metric_queries else None,
        "recall_at_10": (recall_hits[10] / metric_queries) if metric_queries else None,
    }

    payload = {"summary": summary, "results": all_results}
    if args.out:
        out_path = Path(args.out).resolve()
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote: {out_path}")

    print(
        f"[summary] num_queries={summary['num_queries']} num_gallery={summary['num_gallery']} "
        f"metric_queries={summary['metric_queries']} mAP={summary['mAP']}"
    )
    for key in ("recall_at_1", "recall_at_5", "recall_at_10"):
        print(f"[summary] {key}={summary[key]}")

    for result in all_results[: min(10, len(all_results))]:
        print(f"[query] {result['query_track_id']} identity={result['query_identity_id']}")
        for hit in result["topk"]:
            print(
                f"  {hit['rank']:>2d} score={hit['score']:.4f} relevant={hit['is_relevant']} "
                f"{hit['gallery_track_id']} ({hit['gallery_scene_dir']})"
            )


if __name__ == "__main__":
    main()
