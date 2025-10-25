# mcp_modules/parser_solidity.py
from __future__ import annotations

import json, hashlib, time, os, re
from typing import Any, Dict, List, Literal, Optional
from pathlib import Path

try:
    from platformdirs import user_cache_dir
    PLATFORMDIRS_AVAILABLE = True
except ImportError:
    PLATFORMDIRS_AVAILABLE = False

from solcx import compile_standard
from .common.normalize import (
    extract_pragma,
    ensure_solc_auto_by_pragma,
    ensure_solc,
    string_or_none,
    src_span,
    line_from_offset,
    node_kind,
    iter_nodes,
)

from .common.errors import (
    ErrorHandler,
    ErrorCode,
    empty_input_error,
    engine_not_implemented_error,
    solc_install_failed_error,
    solc_not_installed_error
)

from .common.types import (
    ParserSolidityResult,
    FunctionDefinition,
    ContractDefinition,
    FunctionParameter,
    FunctionPosition,
    ModifierDefinition,
    EventDefinition,
    StructDefinition,
    StructMember,
    EnumDefinition,
    SourceMapping,
    ParserSolidityMeta
)

Engine = Literal["treesitter", "solc"]


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




def _compile_to_ast_with_solc(input_code: str | Dict[str, str], solc_version: str) -> Dict[str, Any]:
    """Compile to AST with solc, supporting both single file and multiple files."""
    
    if isinstance(input_code, str):
        sources = {"Contract.sol": {"content": input_code}}
        source_list = ["Contract.sol"]
    else:
        sources = {filename: {"content": content} for filename, content in input_code.items()}
        source_list = list(input_code.keys())
    
    # Create output selection for each source file
    output_selection = {}
    for filename in source_list:
        output_selection[filename] = {"": ["ast", "legacyAST"]}
    
    std_input = {
        "language": "Solidity",
        "sources": sources,
        "settings": {
            "outputSelection": output_selection,
        },
    }
    out = compile_standard(std_input, allow_paths=".")
    
    if len(source_list) == 1:
        src = out.get("sources", {}).get(source_list[0], {})
        ast = src.get("ast") or src.get("legacyAST")
        if not ast:
            ast = out.get("ast") or out
    else:
        ast = out
    
    return ast




def _create_src_mapping(compilation_result: Dict[str, Any], source_list: List[str]) -> Dict[str, SourceMapping]:
    """
    Create mapping from src strings to (filename, start, length) for accurate positioning.
    Returns dict with src as key and {"filename": str, "start": int, "length": int} as value.
    """
    src_mapping: Dict[str, SourceMapping] = {}
    
    if len(source_list) == 1:
        filename = source_list[0]
        sources = compilation_result.get("sources", {})
        if filename in sources:
            src_ast = sources[filename].get("ast") or sources[filename].get("legacyAST")
            if src_ast:
                _extract_src_from_ast(src_ast, filename, src_mapping)
    else:
        sources = compilation_result.get("sources", {})
        for filename in source_list:
            if filename in sources:
                src_ast = sources[filename].get("ast") or sources[filename].get("legacyAST")
                if src_ast:
                    _extract_src_from_ast(src_ast, filename, src_mapping)
    
    return src_mapping

def _extract_src_from_ast(node: Dict[str, Any], filename: str, src_mapping: Dict[str, SourceMapping]) -> None:
    """Recursively extract src mappings from AST nodes."""
    if isinstance(node, dict):
        src = node.get("src")
        if src and isinstance(src, str):
            try:
                parts = src.split(":")
                if len(parts) >= 2:
                    start = int(parts[0])
                    length = int(parts[1])
                    src_mapping[src] = {
                        "filename": filename,
                        "start": start,
                        "length": length
                    }
            except (ValueError, IndexError):
                pass
        
        for value in node.values():
            if isinstance(value, (dict, list)):
                _extract_src_from_ast(value, filename, src_mapping)
    
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, (dict, list)):
                _extract_src_from_ast(item, filename, src_mapping)

def _extract_contracts_with_members(ast: Dict[str, Any], source_text: str | Dict[str, str], src_mapping: Dict[str, SourceMapping] = None) -> List[ContractDefinition]:
    contracts: List[ContractDefinition] = []

    def _extract_contract_header(node: Dict[str, Any]) -> Dict[str, Any]:
        kind = string_or_none(node.get("contractKind")) or "contract"
        name = string_or_none(node.get("name")) or ""
        bases: List[str] = []
        for b in node.get("baseContracts", []) or []:
            bn = b.get("baseName", {}).get("name")
            if bn:
                bases.append(bn)
        return {"name": name, "kind": kind, "bases": bases}

    def walk(node: Dict[str, Any]) -> None:
        if node_kind(node) == "ContractDefinition":
            header = _extract_contract_header(node)
            members = _collect_from_contract(node, source_text, src_mapping)
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
            return

        for ch in iter_nodes(node):
            walk(ch)

    walk(ast)

    uniq: Dict[tuple, Dict[str, Any]] = {}
    for c in contracts:
        uniq[(c["name"], c["kind"])] = c
    return list(uniq.values())


def _params_list(param_list_node: Dict[str, Any]) -> List[FunctionParameter]:
    """Returns normalized parameters (inputs/returns)."""
    res: List[FunctionParameter] = []
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
                    "name": string_or_none(p.get("name")) or "",
                    "type": typ,
                }
            )
        return res

    for ch in iter_nodes(param_list_node):
        if node_kind(ch) == "VariableDeclaration":
            typ = (
                ch.get("typeDescriptions", {}).get("typeString")
                or ch.get("attributes", {}).get("type")
                or ch.get("typeName", {}).get("name")
                or "unknown"
            )
            res.append({"name": string_or_none(ch.get("name")) or "", "type": typ})
    return res


def _func_signature(name: str, inputs: List[FunctionParameter], returns: List[FunctionParameter]) -> str:
    ins = ",".join(p.get("type", "unknown") for p in inputs)
    if returns:
        outs = ",".join(p.get("type", "unknown") for p in returns)
        return f"{name}({ins}) returns ({outs})"
    return f"{name}({ins})"


def _collect_from_contract(contract_node: Dict[str, Any], source_text: str | Dict[str, str], src_mapping: Dict[str, SourceMapping] = None) -> Dict[str, List[Any]]:
    functions: List[FunctionDefinition] = []
    modifiers_decl: List[ModifierDefinition] = []
    events: List[EventDefinition] = []
    structs: List[StructDefinition] = []
    enums: List[EnumDefinition] = []

    for ch in iter_nodes(contract_node):
        kind = node_kind(ch)

        if kind == "FunctionDefinition":
            fn_src = ch.get("src")
            fn_name = string_or_none(ch.get("name")) or ""
            fn_kind = string_or_none(ch.get("kind")) or "function"
            visibility = string_or_none(ch.get("visibility")) or "public"
            state_mut = string_or_none(ch.get("stateMutability")) or "nonpayable"

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

            pos = src_span(ch)
            
            # Handle multi-file case by determining which file this function belongs to
            if isinstance(source_text, dict) and src_mapping and fn_src:
                # Get the filename from src_mapping
                src_info = src_mapping.get(fn_src)
                if src_info and src_info["filename"] in source_text:
                    file_content = source_text[src_info["filename"]]
                    start_line = line_from_offset(file_content, pos["offsetStart"])
                else:
                    start_line = None
            else:
                # Single file case
                start_line = line_from_offset(source_text, pos["offsetStart"])

            position: FunctionPosition = {
                "offset_start": pos["offsetStart"],
                "offset_end": pos["offsetEnd"],
                "line": start_line
            }

            function_def: FunctionDefinition = {
                "name": display_name,
                "kind": fn_kind,
                "visibility": visibility,
                "state_mutability": state_mut,
                "modifiers": mods,
                "inputs": inputs,
                "returns": returns,
                "signature": _func_signature(display_name, inputs, returns),
                "is_constructor": is_constructor,
                "is_fallback": is_fallback,
                "is_receive": is_receive,
                "position": position,
                "src": fn_src,
                "stateMutability": state_mut,
            }
            functions.append(function_def)

        elif kind == "ModifierDefinition":
            modifier_def: ModifierDefinition = {
                "name": string_or_none(ch.get("name")) or "",
                "parameters": _params_list(ch.get("parameters") or {}),
            }
            modifiers_decl.append(modifier_def)

        elif kind == "EventDefinition":
            params = _params_list(ch.get("parameters") or {})
            event_def: EventDefinition = {
                "name": string_or_none(ch.get("name")) or "",
                "parameters": params,
            }
            events.append(event_def)

        elif kind == "StructDefinition":
            members: List[StructMember] = []
            for mem in iter_nodes(ch):
                if node_kind(mem) == "VariableDeclaration":
                    typ = (
                        mem.get("typeDescriptions", {}).get("typeString")
                        or mem.get("typeName", {}).get("name")
                        or "unknown"
                    )
                    member: StructMember = {
                        "name": string_or_none(mem.get("name")) or "", 
                        "type": typ
                    }
                    members.append(member)
            struct_def: StructDefinition = {
                "name": string_or_none(ch.get("name")) or "", 
                "members": members
            }
            structs.append(struct_def)

        elif kind == "EnumDefinition":
            vals: List[str] = []
            for v in iter_nodes(ch):
                nm = string_or_none(v.get("name"))
                if nm:
                    vals.append(nm)
            enum_def: EnumDefinition = {
                "name": string_or_none(ch.get("name")) or "", 
                "values": vals
            }
            enums.append(enum_def)

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
    input_code: str | Dict[str, str],
    *,
    engine: Engine = "solc",
    auto_version: bool = True,          
    solc_version: str | None = None,
    persist_ast: bool = True,
    ast_cache_dir: str | None = None,
) -> ParserSolidityResult:
    t0 = time.time()
    
    if isinstance(input_code, str):
        if not input_code or not input_code.strip():
            return empty_input_error(
                duration_ms=int((time.time() - t0) * 1000),
                engine=engine,
                solc_version=solc_version
            )
        source_list = ["Contract.sol"]
    else:
        if not input_code or not any(content.strip() for content in input_code.values()):
            return empty_input_error(
                duration_ms=int((time.time() - t0) * 1000),
                engine=engine,
                solc_version=solc_version
            )
        source_list = list(input_code.keys())

    if engine != "solc":
        return engine_not_implemented_error(
            engine=engine,
            duration_ms=int((time.time() - t0) * 1000)
        )

    try:
        steps = []
        
        if auto_version:
            # Handle multi-file case for pragma extraction
            if isinstance(input_code, str):
                pragma_input = input_code
            else:
                # Use the first file for pragma detection
                pragma_input = next(iter(input_code.values()))
            chosen = ensure_solc_auto_by_pragma(pragma_input, fallback_version=solc_version or "0.8.26")
        else:
            chosen = solc_version or "0.8.26"
            ensure_solc(chosen)
        
        steps.append({"step": "picked_solc", "value": str(chosen)})

        ast = _compile_to_ast_with_solc(input_code, solc_version=chosen)
        steps.append({"step": "compiled", "ok": True})
        
        src_mapping = _create_src_mapping(ast, source_list)
        
        # Extract contracts from each source file
        contracts = []
        if isinstance(input_code, str):
            # Single file case
            contracts = _extract_contracts_with_members(ast, input_code, src_mapping)
        else:
            # Multi-file case - extract from each source AST
            sources = ast.get("sources", {})
            for filename in source_list:
                if filename in sources and "ast" in sources[filename]:
                    source_ast = sources[filename]["ast"]
                    source_contracts = _extract_contracts_with_members(source_ast, input_code[filename], src_mapping)
                    contracts.extend(source_contracts)

        if isinstance(input_code, str):
            pragma = extract_pragma(input_code)
        else:
            first_file_content = next(iter(input_code.values()))
            pragma = extract_pragma(first_file_content)

        elapsed_ms = int((time.time() - t0) * 1000)
        
        all_functions: List[FunctionDefinition] = []
        all_events: List[EventDefinition] = []
        all_modifiers: List[ModifierDefinition] = []
        all_structs: List[StructDefinition] = []
        all_enums: List[EnumDefinition] = []
        
        for contract in contracts:
            all_functions.extend(contract.get("functions", []))
            all_events.extend(contract.get("events", []))
            all_modifiers.extend(contract.get("modifiers", []))
            all_structs.extend(contract.get("structs", []))
            all_enums.extend(contract.get("enums", []))
        
        result: ParserSolidityResult = {
            "status": "ok",
            "module": "parser_solidity",
            "warnings": [],
            "errors": [],
            "meta": {
                "duration_ms": elapsed_ms,
                "engine": engine,
                "solc_version": str(chosen),
                "pragma": pragma,
                "source_list": source_list,
                "src_mapping": src_mapping,
                "log": steps,
                "module_version": "0.1.0",
            },
            "contracts": contracts,
            "functions": [],
            "events": [],
            "modifiers": [],
            "structs":[],
            "enums": []
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
        error_code = ErrorHandler.parse_runtime_error(str(e))
        if error_code.value == "SolcInstallFailed":
            return solc_install_failed_error(
                version="unknown",
                error=str(e),
                duration_ms=int((time.time() - t0) * 1000),
                engine=engine
            )
        elif error_code.value == "SolcNotInstalled":
            return solc_not_installed_error(
                duration_ms=int((time.time() - t0) * 1000),
                engine=engine
            )
        else:
            return ErrorHandler.create_parser_error(
                error_code,
                message=str(e),
                duration_ms=int((time.time() - t0) * 1000),
                engine=engine,
                solc_version=solc_version
            )
    except Exception as e:
        return ErrorHandler.create_parser_error(
            ErrorCode.RUNTIME_ERROR,
            message=f"{type(e).__name__}: {str(e)}",
            duration_ms=int((time.time() - t0) * 1000),
            engine=engine,
            solc_version=solc_version
        )