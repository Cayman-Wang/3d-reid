from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def lexical_abspath(path: Path) -> Path:
    out = Path(path)
    if not out.is_absolute():
        out = Path.cwd() / out
    return out.absolute()


def _iter_xml_refs(xml_path: Path, *, _visited: set[str] | None = None) -> list[Path]:
    visited = _visited if _visited is not None else set()
    xml_key = str(xml_path)
    if xml_key in visited:
        return []
    visited.add(xml_key)

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    refs: list[Path] = []
    for elem in root.iter():
        file_attr = elem.attrib.get("file")
        if not file_attr:
            continue
        ref_path = lexical_abspath(xml_path.parent / file_attr)
        refs.append(ref_path)
        if elem.tag == "include" and ref_path.exists():
            refs.extend(_iter_xml_refs(ref_path, _visited=visited))
    return refs


def preflight_mjcf_assets(mjcf_path: Path) -> dict[str, Any]:
    mjcf = lexical_abspath(mjcf_path)
    report: dict[str, Any] = {
        "mjcf": str(mjcf),
        "exists": bool(mjcf.exists()),
        "ascii_path": str(mjcf).isascii(),
        "platform_requires_ascii": bool(os.name == "nt"),
        "parse_error": None,
        "external_refs": [],
        "missing_refs": [],
        "ok": False,
    }

    if not mjcf.exists():
        return report

    try:
        refs = _iter_xml_refs(mjcf)
    except Exception as e:
        report["parse_error"] = repr(e)
        return report

    deduped = sorted({str(p) for p in refs})
    missing = sorted(path_str for path_str in deduped if not Path(path_str).exists())
    report["external_refs"] = deduped
    report["missing_refs"] = missing
    report["ok"] = (
        report["exists"]
        and (not report["platform_requires_ascii"] or report["ascii_path"])
        and report["parse_error"] is None
        and not missing
    )
    return report


def format_mjcf_preflight_error(report: dict[str, Any]) -> str:
    lines = [f'MJCF preflight failed: {report.get("mjcf", "<unknown>")}']
    if not bool(report.get("exists")):
        lines.append("- MJCF file does not exist.")
        return "\n".join(lines)

    if bool(report.get("platform_requires_ascii")) and not bool(report.get("ascii_path")):
        lines.append("- On Windows, the MJCF path contains non-ASCII characters.")
        lines.append("- Run MuJoCo from an ASCII-safe workspace path before capturing spin scenes.")

    parse_error = report.get("parse_error")
    if parse_error:
        lines.append(f"- Failed to parse MJCF/XML references: {parse_error}")

    missing_refs = list(report.get("missing_refs") or [])
    if missing_refs:
        lines.append(f"- Missing external assets: {len(missing_refs)}")
        for path_str in missing_refs[:12]:
            lines.append(f"  - {path_str}")
        if len(missing_refs) > 12:
            lines.append(f"  - ... and {len(missing_refs) - 12} more")

    return "\n".join(lines)
