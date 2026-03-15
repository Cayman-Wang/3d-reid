#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Face:
    verts: List[Tuple[int, Optional[int]]]


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip())
    cleaned = cleaned.strip("_")
    if not cleaned:
        cleaned = "material"
    if not cleaned[0].isalpha():
        cleaned = f"m_{cleaned}"
    return cleaned


def _parse_obj(obj_path: Path) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float]], OrderedDict[str, List[Face]]]:
    vertices: List[Tuple[float, float, float]] = []
    texcoords: List[Tuple[float, float]] = []
    faces_by_mtl: "OrderedDict[str, List[Face]]" = OrderedDict()
    current_mtl = "__default__"
    faces_by_mtl[current_mtl] = []

    with obj_path.open("r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                continue
            if line.startswith("vt "):
                parts = line.split()
                if len(parts) >= 3:
                    texcoords.append((float(parts[1]), float(parts[2])))
                continue
            if line.startswith("usemtl "):
                current_mtl = line.split(maxsplit=1)[1].strip() or "__default__"
                faces_by_mtl.setdefault(current_mtl, [])
                continue
            if line.startswith("f "):
                parts = line.split()[1:]
                verts: List[Tuple[int, Optional[int]]] = []
                for part in parts:
                    segs = part.split("/")
                    v_idx = int(segs[0]) if segs[0] else 0
                    vt_idx: Optional[int] = None
                    if len(segs) >= 2 and segs[1]:
                        vt_idx = int(segs[1])
                    verts.append((v_idx, vt_idx))
                faces_by_mtl[current_mtl].append(Face(verts=verts))

    return vertices, texcoords, faces_by_mtl


def _write_obj(
    out_path: Path,
    vertices: List[Tuple[float, float, float]],
    texcoords: List[Tuple[float, float]],
    faces: List[Face],
) -> None:
    v_map: Dict[int, int] = {}
    vt_map: Dict[int, int] = {}
    out_v: List[Tuple[float, float, float]] = []
    out_vt: List[Tuple[float, float]] = []

    remapped_faces: List[List[Tuple[int, Optional[int]]]] = []
    for face in faces:
        remapped: List[Tuple[int, Optional[int]]] = []
        for v_idx, vt_idx in face.verts:
            if v_idx not in v_map:
                v_map[v_idx] = len(out_v) + 1
                out_v.append(vertices[v_idx - 1])
            new_v = v_map[v_idx]
            new_vt: Optional[int] = None
            if vt_idx is not None:
                if vt_idx not in vt_map:
                    vt_map[vt_idx] = len(out_vt) + 1
                    out_vt.append(texcoords[vt_idx - 1])
                new_vt = vt_map[vt_idx]
            remapped.append((new_v, new_vt))
        remapped_faces.append(remapped)

    with out_path.open("w", encoding="utf-8") as f:
        for v in out_v:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for vt in out_vt:
            f.write(f"vt {vt[0]} {vt[1]}\n")
        for face in remapped_faces:
            parts: List[str] = []
            for v_idx, vt_idx in face:
                if vt_idx is None:
                    parts.append(str(v_idx))
                else:
                    parts.append(f"{v_idx}/{vt_idx}")
            f.write("f " + " ".join(parts) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Split an OBJ by usemtl into per-material OBJ files.")
    ap.add_argument("--obj", required=True, type=str, help="Input OBJ file.")
    ap.add_argument("--out-dir", required=True, type=str, help="Output directory for per-material OBJ files.")
    ap.add_argument(
        "--prefix",
        default="j10_mtl_",
        type=str,
        help="Prefix for output file names (sanitized material name is appended).",
    )
    args = ap.parse_args()

    obj_path = Path(args.obj).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    vertices, texcoords, faces_by_mtl = _parse_obj(obj_path)
    if not vertices:
        raise SystemExit("No vertices found in OBJ.")
    if not faces_by_mtl:
        raise SystemExit("No faces found in OBJ.")

    for mtl, faces in faces_by_mtl.items():
        if not faces:
            continue
        safe_name = _sanitize_name(mtl)
        out_name = f"{args.prefix}{safe_name}.obj"
        _write_obj(out_dir / out_name, vertices, texcoords, faces)

    print(f"Wrote {len([f for f in faces_by_mtl.values() if f])} OBJ files to {out_dir}")


if __name__ == "__main__":
    main()
