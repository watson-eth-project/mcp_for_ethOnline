# mcp_modules/parser_solidity.py
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Literal, Optional, TypedDict

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
    ast: Dict[str, Any]
    functions: List[Dict[str, Any]]
    contracts: List[Dict[str, Any]]


def _ensure_solc_auto_by_pragma(input_code: str, fallback_version: str | None = "0.8.26") -> str:
    """Check we have solc version installed by pragma"""

    try:
        ver = install_solc_pragma(input_code)  
        if ver:
            set_solc_version(ver)             
            return ver
    except Exception:
        pass

    if fallback_version:
        _ensure_solc(fallback_version)
        return fallback_version

    raise RuntimeError("Unable to determine or install a suitable solc version.")

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


def _extract_contracts_with_members(ast: Dict[str, Any]) -> List[Dict[str, Any]]:
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
            members = _collect_from_contract(node)
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


def _collect_from_contract(contract_node: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    functions: List[Dict[str, Any]] = []
    modifiers_decl: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    structs: List[Dict[str, Any]] = []
    enums: List[Dict[str, Any]] = []

    for ch in _iter_nodes(contract_node):
        kind = _node_kind(ch)

        if kind == "FunctionDefinition":
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

def run(
    input_code: str,
    *,
    engine: Engine = "solc",
    return_raw_ast: bool = False,
    auto_version: bool = True,          
    solc_version: str | None = None,
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
        if auto_version:
            chosen = _ensure_solc_auto_by_pragma(input_code, fallback_version=solc_version or "0.8.26")
        else:
            chosen = solc_version or "0.8.26"
            _ensure_solc(chosen)

        ast = _compile_to_ast_with_solc(input_code, solc_version=chosen)
        payload =  {
            "contracts":  _extract_contracts_with_members(ast)
        }

        result: ParserSolidityResult = {
            "status": "ok",
            "module": "parser_solidity",
            "warnings": [],
            "errors": [],
            "meta": {
                "duration_ms": int((time.time() - t0) * 1000),
                "engine": engine,
                "solc_version": chosen,
            },
            **payload,
        }
        if return_raw_ast:
            result["ast"] = ast
        return result

    except Exception as e:
        return {
            "status": "error",
            "module": "parser_solidity",
            "errors": [type(e).__name__, str(e)],
            "warnings": [],
            "meta": {"duration_ms": int((time.time() - t0) * 1000), "engine": engine, "solc_version": solc_version},
        }