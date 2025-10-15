# mcp_modules/parser_solidity.py
from __future__ import annotations

import json, hashlib, time, os, re
from typing import Any, Dict, List, Literal, Optional, TypedDict
from pathlib import Path

try:
    from platformdirs import user_cache_dir
    PLATFORMDIRS_AVAILABLE = True
except ImportError:
    PLATFORMDIRS_AVAILABLE = False

from solcx import (
    compile_standard,
    install_solc,
    set_solc_version,
    get_installed_solc_versions,
    install_solc_pragma         
)

Engine = Literal["treesitter", "solc"]

class ParserSolidityResult(TypedDict, total=False):
    status: Literal["ok", "error"]
    module: Literal["parser_solidity"]
    warnings: List[str]
    errors: List[str]
    meta: Dict[str, Any]
    ast_uri: str  # URI to cached AST
    functions: List[Dict[str, Any]]
    contracts: List[Dict[str, Any]]

def _extract_pragma(input_code: str) -> str | None:
    """Extract pragma directive from Solidity code."""
    pragma_match = re.search(r'pragma\s+solidity\s+([^;]+);', input_code, re.IGNORECASE)
    if pragma_match:
        return pragma_match.group(1).strip()
    return None

def _get_ast_cache_dir(custom_dir: str | None = None) -> Path:
    """Get AST cache directory using platformdirs or custom path."""
    if custom_dir:
        return Path(custom_dir)
    
    if PLATFORMDIRS_AVAILABLE:
        cache_dir = Path(user_cache_dir("mcp-auditor")) / "ast"
    else:
        cache_dir = Path("/tmp/ast_cache")
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def _ensure_solc_auto_by_pragma(input_code: str, fallback_version: str | None = "0.8.26") -> str:
    """Check we have solc version installed by pragma"""

    # Extract pragma to see what version we're looking for
    pragma = _extract_pragma(input_code)
    
    try:
        ver = install_solc_pragma(pragma)  
        if ver:
            set_solc_version(str(ver))
            return ver
    except Exception:
        pass

    if fallback_version:
        _ensure_solc(fallback_version)
        return fallback_version
    raise RuntimeError("SolcNotInstalled")

def _ensure_solc(solc_version: str) -> None:
    """Check we have solc version installed by solc_version"""
    
    installed = {str(v) for v in get_installed_solc_versions()}
    if solc_version not in installed:
        install_solc(solc_version)
    set_solc_version(solc_version)


def _compile_to_ast_with_solc(input_code: str, solc_version: str) -> Dict[str, Any]:
    """Compile to AST with solc"""
    
    std_input = {
        "language": "Solidity",
        "sources": {"Contract.sol": {"content": input_code}},
        "settings": {
            "outputSelection": {"*": {"": ["ast", "legacyAST"]}},
        },
    }
    out = compile_standard(std_input, allow_paths=".")
    src = out.get("sources", {}).get("Contract.sol", {})

    #Get AST safely
    ast = src.get("ast") or src.get("legacyAST")
    if not ast:
        ast = out.get("ast") or out

    return ast


def _node_kind(node: Dict[str, Any]) -> str:
    return node.get("nodeType") or node.get("name") or ""


def _iter_nodes(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Children of the node (uniformly for legacyAST)."""
    if "nodes" in node and isinstance(node["nodes"], list):
        return node["nodes"]

    if "children" in node and isinstance(node["children"], list):
        return node["children"]
    return []


def _string_or_none(v: Any) -> Optional[str]:
    return v if isinstance(v, str) else None


def _src_span(node: Dict[str, Any]) -> Dict[str, int | None]:
    s = node.get("src")
    if not isinstance(s, str) or ":" not in s:
        return {"offsetStart": None, "offsetEnd": None}
    try:
        start_str, length_str, _ = s.split(":")
        start = int(start_str); length = int(length_str)
        return {"offsetStart": start, "offsetEnd": start + length}
    except Exception:
        return {"offsetStart": None, "offsetEnd": None}

def _line_from_offset(source: str, offset: int | None) -> int | None:
    if offset is None or offset < 0:
        return None
    try:
        return source[:offset].count("\n") + 1
    except Exception:
        return None


def _extract_contracts_with_members(ast: Dict[str, Any], source_text: str) -> List[Dict[str, Any]]:
    contracts: List[Dict[str, Any]] = []

    def _extract_contract_header(node: Dict[str, Any]) -> Dict[str, Any]:
        kind = _string_or_none(node.get("contractKind")) or "contract"
        name = _string_or_none(node.get("name")) or ""
        bases: List[str] = []
        for b in node.get("baseContracts", []) or []:
            bn = b.get("baseName", {}).get("name")
            if bn:
                bases.append(bn)
        return {"name": name, "kind": kind, "bases": bases}

    def walk(node: Dict[str, Any]) -> None:
        if _node_kind(node) == "ContractDefinition":
            header = _extract_contract_header(node)
            members = _collect_from_contract(node, source_text)
            contracts.append(
                {
                    **header,
                    "functions": members["functions"],
                    "modifiers": members["modifiers"],
                    "events": members["events"],
                    "structs": members["structs"],
                    "enums": members["enums"],
                }
            )
            # Не идём внутрь этого узла дальше, так как _collect_from_contract уже прошёл его детей
            return

        for ch in _iter_nodes(node):
            walk(ch)

    walk(ast)

    # Убираем возможные дубликаты по (name, kind) сохраняя последний вариант
    uniq: Dict[tuple, Dict[str, Any]] = {}
    for c in contracts:
        uniq[(c["name"], c["kind"])] = c
    return list(uniq.values())


def _params_list(param_list_node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Returns normalized parameters (inputs/returns)."""
    res: List[Dict[str, Any]] = []
    if not param_list_node:
        return res

    params = param_list_node.get("parameters")
    if isinstance(params, list):
        for p in params:
            if not isinstance(p, dict):
                continue
            typ = p.get("typeDescriptions", {}).get("typeString") or p.get("typeName", {}).get("name") or "unknown"
            res.append(
                {
                    "name": _string_or_none(p.get("name")) or "",
                    "type": typ,
                }
            )
        return res

    for ch in _iter_nodes(param_list_node):
        if _node_kind(ch) == "VariableDeclaration":
            typ = (
                ch.get("typeDescriptions", {}).get("typeString")
                or ch.get("attributes", {}).get("type")
                or ch.get("typeName", {}).get("name")
                or "unknown"
            )
            res.append({"name": _string_or_none(ch.get("name")) or "", "type": typ})
    return res


def _func_signature(name: str, inputs: List[Dict[str, Any]], returns: List[Dict[str, Any]]) -> str:
    ins = ",".join(p.get("type", "unknown") for p in inputs)
    if returns:
        outs = ",".join(p.get("type", "unknown") for p in returns)
        return f"{name}({ins}) returns ({outs})"
    return f"{name}({ins})"


def _collect_from_contract(contract_node: Dict[str, Any], source_text: str) -> Dict[str, List[Dict[str, Any]]]:
    functions: List[Dict[str, Any]] = []
    modifiers_decl: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    structs: List[Dict[str, Any]] = []
    enums: List[Dict[str, Any]] = []

    for ch in _iter_nodes(contract_node):
        kind = _node_kind(ch)

        if kind == "FunctionDefinition":
            fn_src = ch.get("src")
            fn_name = _string_or_none(ch.get("name")) or ""
            fn_kind = _string_or_none(ch.get("kind")) or "function"
            visibility = _string_or_none(ch.get("visibility")) or "public"
            state_mut = _string_or_none(ch.get("stateMutability")) or "nonpayable"

            inputs = _params_list(ch.get("parameters") or ch.get("parametersList") or {})
            returns = _params_list(ch.get("returnParameters") or ch.get("returnParametersList") or {})

            mods: List[str] = []
            for m in ch.get("modifiers", []) or []:
                name = (
                    m.get("modifierName", {}).get("name")
                    or m.get("name")
                    or m.get("modifierName", {}).get("identifier", {}).get("name")
                )
                if name:
                    mods.append(name)

            is_constructor = fn_kind == "constructor"
            is_fallback = fn_kind == "fallback"
            is_receive = fn_kind == "receive"

            display_name = fn_name or fn_kind

            pos = _src_span(ch)
            start_line = _line_from_offset(source_text, pos["offsetStart"])

            functions.append(
                {
                    "name": display_name,
                    "kind": fn_kind,
                    "visibility": visibility,
                    "stateMutability": state_mut,
                    "modifiers": mods,
                    "inputs": inputs,
                    "returns": returns,
                    "signature": _func_signature(display_name, inputs, returns),
                    "isConstructor": is_constructor,
                    "isFallback": is_fallback,
                    "isReceive": is_receive,
                    "position": {
                        "offsetStart": pos["offsetStart"],
                        "offsetEnd": pos["offsetEnd"],
                        "line": start_line
                    },
                    "src": fn_src,
                }
            )

        elif kind == "ModifierDefinition":
            modifiers_decl.append(
                {
                    "name": _string_or_none(ch.get("name")) or "",
                    "parameters": _params_list(ch.get("parameters") or {}),
                }
            )

        elif kind == "EventDefinition":
            params = _params_list(ch.get("parameters") or {})
            events.append(
                {
                    "name": _string_or_none(ch.get("name")) or "",
                    "parameters": params,
                }
            )

        elif kind == "StructDefinition":
            members: List[Dict[str, Any]] = []
            for mem in _iter_nodes(ch):
                if _node_kind(mem) == "VariableDeclaration":
                    typ = (
                        mem.get("typeDescriptions", {}).get("typeString")
                        or mem.get("typeName", {}).get("name")
                        or "unknown"
                    )
                    members.append({"name": _string_or_none(mem.get("name")) or "", "type": typ})
            structs.append({"name": _string_or_none(ch.get("name")) or "", "members": members})

        elif kind == "EnumDefinition":
            vals: List[str] = []
            for v in _iter_nodes(ch):
                nm = _string_or_none(v.get("name"))
                if nm:
                    vals.append(nm)
            enums.append({"name": _string_or_none(ch.get("name")) or "", "values": vals})

    return {
        "functions": functions,
        "modifiers": modifiers_decl,
        "events": events,
        "structs": structs,
        "enums": enums,
    }


def _save_ast(raw_ast_json: str, cache_dir: Path) -> tuple[str | None, str | None, int]:
    """Save AST to cache and return URI, hash, and size."""
    try:
        h = hashlib.sha1(raw_ast_json.encode("utf-8")).hexdigest()[:8]
        path = cache_dir / f"{h}.json"
        path.write_text(raw_ast_json, encoding="utf-8")
        return f"ast://{h}", h, len(raw_ast_json.encode("utf-8"))
    except Exception:
        return None, None, 0

def run(
    input_code: str,
    *,
    engine: Engine = "solc",
    auto_version: bool = True,          
    solc_version: str | None = None,
    persist_ast: bool = True,
    ast_cache_dir: str | None = None,
) -> ParserSolidityResult:
    t0 = time.time()
    if not input_code or not input_code.strip():
        return {
            "status": "error",
            "module": "parser_solidity",
            "errors": ["EmptyInput"],
            "warnings": [],
            "meta": {"duration_ms": int((time.time() - t0) * 1000), "engine": engine, "solc_version": solc_version},
        }

    if engine != "solc":
        return {
            "status": "error",
            "module": "parser_solidity",
            "errors": [f"EngineNotImplemented:{engine}"],
            "warnings": [],
            "meta": {"duration_ms": int((time.time() - t0) * 1000)},
        }

    try:
        steps = []
        
        if auto_version:
            chosen = _ensure_solc_auto_by_pragma(input_code, fallback_version=solc_version or "0.8.26")
        else:
            chosen = solc_version or "0.8.26"
            _ensure_solc(chosen)
        
        steps.append({"step": "picked_solc", "value": str(chosen)})

        ast = _compile_to_ast_with_solc(input_code, solc_version=chosen)
        steps.append({"step": "compiled", "ok": True})
        
        payload =  {
            "contracts":  _extract_contracts_with_members(ast, input_code)
        }

        # Extract pragma for metadata
        pragma = _extract_pragma(input_code)

        result: ParserSolidityResult = {
            "status": "ok",
            "module": "parser_solidity",
            "warnings": [],
            "errors": [],
            "meta": {
                "duration_ms": int((time.time() - t0) * 1000),
                "engine": engine,
                "solc_version": str(chosen),
                "pragma": pragma,
                "log": steps,
            },
            **payload,
        }

        if persist_ast:
            cache_dir = _get_ast_cache_dir(ast_cache_dir)
            raw_ast_json = json.dumps(ast, ensure_ascii=False, indent=2)
            ast_uri, ast_hash, ast_size_bytes = _save_ast(raw_ast_json, cache_dir)
            if ast_uri:
                result["ast_uri"] = ast_uri
                result["meta"]["ast_hash"] = ast_hash
                result["meta"]["ast_size_bytes"] = ast_size_bytes
                result["meta"]["cache_dir"] = str(cache_dir)
                steps.append({"step": "ast_saved", "uri": ast_uri})
                result["meta"]["log"] = steps
       
        return result

    except RuntimeError as e:
        msg = str(e)
        if msg.startswith("SolcInstallFailed"):
            code = "SolcInstallFailed"
        elif "SolcNotInstalled" in msg:
            code = "SolcNotInstalled"
        else:
            code = "RuntimeError"
        return {
            "status": "error",
            "module": "parser_solidity",
            "errors": [code, msg],
            "warnings": [],
            "meta": {"duration_ms": int((time.time() - t0) * 1000), "engine": engine, "solc_version": solc_version},
        }
    except Exception as e:
        return {
            "status": "error",
            "module": "parser_solidity",
            "errors": [type(e).__name__, str(e)],
            "warnings": [],
            "meta": {"duration_ms": int((time.time() - t0) * 1000), "engine": engine, "solc_version": solc_version},
        }