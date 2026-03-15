from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _l2norm_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.size == 0:
        return x
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def _load_scene_embeddings(scene_dir: Path) -> tuple[np.ndarray, list[dict]]:
    emb_path = scene_dir / "embeddings" / "tracks.npy"
    meta_path = scene_dir / "embeddings" / "tracks_meta.json"
    if not emb_path.exists() or not meta_path.exists():
        raise SystemExit(f"Missing embeddings in scene: {scene_dir} (need embeddings/tracks.npy + tracks_meta.json)")
    embs = np.load(str(emb_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, list):
        raise SystemExit(f"Invalid meta json: {meta_path}")
    return embs.astype(np.float32), meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Cosine retrieval over track embeddings (numpy baseline; FAISS optional later).")
    ap.add_argument("--query_scene_dir", required=True, type=str)
    ap.add_argument("--gallery_scene_dir", action="append", default=[], type=str, help="Repeatable")
    ap.add_argument("--topk", default=5, type=int)
    ap.add_argument("--exclude_same_track_id", action="store_true", help="Skip identical track_id matches")
    ap.add_argument("--out", default="", type=str, help="Optional: write results json")
    args = ap.parse_args()

    q_scene = Path(args.query_scene_dir).resolve()
    g_scenes = [Path(s).resolve() for s in (args.gallery_scene_dir or [])]
    if not g_scenes:
        raise SystemExit("Provide at least one --gallery_scene_dir")

    q_embs, q_meta = _load_scene_embeddings(q_scene)
    q_embs = _l2norm_rows(q_embs)

    gallery_embs_all: list[np.ndarray] = []
    gallery_meta_all: list[dict] = []
    for gs in g_scenes:
        g_embs, g_meta = _load_scene_embeddings(gs)
        g_embs = _l2norm_rows(g_embs)
        gallery_embs_all.append(g_embs)
        for m in g_meta:
            m = dict(m)
            m["scene_dir"] = str(gs)
            gallery_meta_all.append(m)

    g_embs_all = np.concatenate(gallery_embs_all, axis=0) if gallery_embs_all else np.zeros((0, 0), dtype=np.float32)
    if q_embs.size == 0 or g_embs_all.size == 0:
        raise SystemExit("Empty embeddings (no tracks).")
    if q_embs.shape[1] != g_embs_all.shape[1]:
        raise SystemExit(f"Dim mismatch: query D={q_embs.shape[1]} vs gallery D={g_embs_all.shape[1]}")

    scores = q_embs @ g_embs_all.T  # cosine if both normalized
    topk = int(args.topk)
    results: list[dict] = []
    for qi, qm in enumerate(q_meta):
        qid = str(qm.get("track_id", f"q{qi}"))
        row = scores[qi]
        idx = np.argsort(-row)[: max(topk * 5, topk)]  # take a few extra for filtering

        hits: list[dict] = []
        for gi in idx.tolist():
            gm = gallery_meta_all[int(gi)]
            gid = str(gm.get("track_id", f"g{gi}"))
            if args.exclude_same_track_id and gid == qid:
                continue
            hits.append(
                {
                    "rank": len(hits) + 1,
                    "score": float(row[int(gi)]),
                    "gallery_track_id": gid,
                    "gallery_scene_dir": str(gm.get("scene_dir", "")),
                }
            )
            if len(hits) >= topk:
                break

        results.append({"query_track_id": qid, "query_scene_dir": str(q_scene), "topk": hits})

    if args.out:
        out_path = Path(args.out).resolve()
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote: {out_path}")

    # Also print a quick human-readable preview.
    for r in results[: min(10, len(results))]:
        print(f"[query] {r['query_track_id']}")
        for hit in r["topk"]:
            print(f"  {hit['rank']:>2d}  score={hit['score']:.4f}  {hit['gallery_track_id']}  ({hit['gallery_scene_dir']})")


if __name__ == "__main__":
    main()

