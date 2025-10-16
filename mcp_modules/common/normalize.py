from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from solcx import (
    install_solc,
    install_solc_pragma,
    set_solc_version,
    get_installed_solc_versions,
)


def extract_pragma(input_code: str) -> str | None:
    """Extract pragma directive from Solidity code."""
    pragma_match = re.search(r'pragma\s+solidity\s+([^;]+);', input_code, re.IGNORECASE)
    if pragma_match:
        return pragma_match.group(1).strip()
    return None


def detect_pragma(src: str) -> str | None:
    """Detect pragma directive from Solidity source code."""
    m = re.search(r"pragma\s+solidity\s+([^;]+);", src)
    return m.group(1).strip() if m else None


def ensure_solc_auto_by_pragma(input_code: str, fallback_version: str | None = "0.8.26") -> str:
    """Check we have solc version installed by pragma"""
    pragma = extract_pragma(input_code)
    
    try:
        ver = install_solc_pragma(pragma)  
        if ver:
            set_solc_version(str(ver))
            return ver
    except Exception:
        pass

    if fallback_version:
        ensure_solc(fallback_version)
        return fallback_version

    raise RuntimeError("Unable to determine or install a suitable solc version.")


def ensure_solc(solc_version: str) -> None:
    """Check we have solc version installed by solc_version"""
    installed = {str(v) for v in get_installed_solc_versions()}
    if solc_version not in installed:
        install_solc(solc_version)
    set_solc_version(solc_version)


def string_or_none(v: Any) -> Optional[str]:
    """Convert value to string or None if not a string."""
    return v if isinstance(v, str) else None


def src_span(node: Dict[str, Any]) -> Dict[str, int | None]:
    """Extract source span information from AST node."""
    s = node.get("src")
    if not isinstance(s, str) or ":" not in s:
        return {"offsetStart": None, "offsetEnd": None}
    try:
        start_str, length_str, _ = s.split(":")
        start = int(start_str)
        length = int(length_str)
        return {"offsetStart": start, "offsetEnd": start + length}
    except Exception:
        return {"offsetStart": None, "offsetEnd": None}


def line_from_offset(source: str, offset: int | None) -> int | None:
    """Calculate line number from character offset in source code."""
    if offset is None or offset < 0:
        return None
    try:
        return source[:offset].count("\n") + 1
    except Exception:
        return None


def node_kind(node: Dict[str, Any]) -> str:
    """Get the node type/kind from AST node."""
    return node.get("nodeType") or node.get("name") or ""


def iter_nodes(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get children of the node (uniformly for legacyAST)."""
    if "nodes" in node and isinstance(node["nodes"], list):
        return node["nodes"]

    if "children" in node and isinstance(node["children"], list):
        return node["children"]
    return []