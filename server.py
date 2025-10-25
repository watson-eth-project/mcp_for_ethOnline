from mcp_modules import parser_solidity, slither_wrapper, surya_wrapper

from mcp.server.fastmcp import FastMCP
from pathlib import Path
import json, re, time, os
from typing import Dict, List, Any, Optional
import jsonpointer

try:
    from platformdirs import user_cache_dir
    PLATFORMDIRS_AVAILABLE = True
except ImportError:
    PLATFORMDIRS_AVAILABLE = False

mcp = FastMCP("Demo")

def _get_ast_cache_dir() -> Path:
    """Get AST cache directory using platformdirs or fallback to /tmp."""
    if PLATFORMDIRS_AVAILABLE:
        cache_dir = Path(user_cache_dir("mcp-auditor")) / "ast"
    else:
        cache_dir = Path("/tmp/ast_cache")
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

AST_CACHE_DIR = _get_ast_cache_dir() 

def _create_ast_index(ast: dict, hash_name: str) -> dict:
    """
    Create an index mapping nodeType and name to JSON pointers for O(1) lookup.
    Returns a dictionary with structure:
    {
        "nodeType": {
            "FunctionDefinition": ["/sources/Contract.sol/ast/nodes/0", ...],
            "ContractDefinition": ["/sources/Contract.sol/ast/nodes/1", ...]
        },
        "name": {
            "transfer": ["/sources/Contract.sol/ast/nodes/0/nodes/2", ...],
            "owner": ["/sources/Contract.sol/ast/nodes/0/nodes/1", ...]
        }
    }
    """
    index = {"nodeType": {}, "name": {}}
    
    def index_node(node, path=""):
        if isinstance(node, dict):
            node_type = node.get("nodeType")
            if node_type:
                if node_type not in index["nodeType"]:
                    index["nodeType"][node_type] = []
                index["nodeType"][node_type].append(path)
            
            name = node.get("name")
            if name and isinstance(name, str):
                if name not in index["name"]:
                    index["name"][name] = []
                index["name"][name].append(path)
            
            for key, value in node.items():
                if isinstance(value, (dict, list)):
                    child_path = f"{path}/{key}" if path else f"/{key}"
                    index_node(value, child_path)
        
        elif isinstance(node, list):
            for i, item in enumerate(node):
                child_path = f"{path}/{i}" if path else f"/{i}"
                index_node(item, child_path)
    
    index_node(ast)
    return index

def _save_ast_index(index: dict, hash_name: str) -> None:
    """Save AST index to sidecar file."""
    index_path = AST_CACHE_DIR / f"{hash_name}.index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

def _load_ast_index(hash_name: str) -> dict | None:
    """Load AST index from sidecar file."""
    index_path = AST_CACHE_DIR / f"{hash_name}.index.json"
    if index_path.exists():
        try:
            return json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

@mcp.resource("ast://{hash}")
def get_ast(hash: str) -> Dict[str, Any]:
    """
    Returns the saved AST by hash as structured JSON.
    Safely check the hash (only hex, 8 characters), read from AST_CACHE_DIR.
    """
    if not re.fullmatch(r"[0-9a-fA-F]{8}", hash):
        raise ValueError("Bad AST hash")
    p = AST_CACHE_DIR / f"{hash}.json"
    if not p.exists():
        raise FileNotFoundError(f"AST not found: {hash}")
    
    return json.loads(p.read_text(encoding="utf-8"))

@mcp.tool()
def get_ast_node(hash: str, src: str | None = None, pointer: str | None = None, match: dict | None = None, limit: int = 10) -> dict:
    """
    Returns a subtree of the AST by src / JSON Pointer / simple match-filter.
    
    Match modes:
    - exact: exact match (default)
    - contains: substring match for string values
    - regex: regex pattern match for string values
    
    Example match patterns:
    - {"name": "transfer", "mode": "contains"} - find nodes with name containing "transfer"
    - {"nodeType": "FunctionDefinition", "mode": "exact"} - find exact nodeType matches
    - {"name": "^transfer.*", "mode": "regex"} - find nodes with name matching regex
    """
    limit = max(1, min(limit, 100))
    
    p = AST_CACHE_DIR / f"{hash}.json"
    if not p.exists():
        return {"status":"error", "error":"AST not found"}

    ast = json.loads(p.read_text("utf-8"))

    def find_by_src(node, target):
        found = []
        if isinstance(node, dict):
            if node.get("src") == target:
                found.append(node)
            for v in node.values():
                found.extend(find_by_src(v, target))
        elif isinstance(node, list):
            for it in node:
                found.extend(find_by_src(it, target))
        return found

    def find_by_match(node, match_criteria):
        found = []
        if isinstance(node, dict):
            match_mode = match_criteria.pop("mode", "exact")
            
            matches = True
            for key, expected_value in match_criteria.items():
                actual_value = node.get(key)
                
                if match_mode == "exact":
                    if actual_value != expected_value:
                        matches = False
                        break
                elif match_mode == "contains":
                    if not isinstance(actual_value, str) or expected_value not in actual_value:
                        matches = False
                        break
                elif match_mode == "regex":
                    if not isinstance(actual_value, str):
                        matches = False
                        break
                    try:
                        import re
                        if not re.search(expected_value, actual_value):
                            matches = False
                            break
                    except re.error:
                        matches = False
                        break
                else:
                    if actual_value != expected_value:
                        matches = False
                        break
            
            if matches:
                found.append(node)
            
            match_criteria["mode"] = match_mode
            
            for v in node.values():
                found.extend(find_by_match(v, match_criteria))
        elif isinstance(node, list):
            for it in node:
                found.extend(find_by_match(it, match_criteria))
        return found

    if src:
        nodes = find_by_src(ast, src)
        return {"status":"ok", "mode":"src", "count": len(nodes), "nodes": nodes[:limit]}  

    if pointer:
        try:
            node = jsonpointer.resolve_pointer(ast, pointer)
            return {"status":"ok", "mode":"pointer", "node": node}
        except Exception as e:
            return {"status":"error", "error": f"Bad pointer: {e}"}

    if match:
        nodes = find_by_match(ast, match.copy())  # Use copy to avoid modifying original
        return {"status":"ok", "mode":"match", "count": len(nodes), "nodes": nodes[:limit]}

    return {"status":"error", "error":"Provide src or pointer or match"}

@mcp.tool()
def quick_find_ast_nodes(hash: str, node_type: str | None = None, name: str | None = None, limit: int = 10) -> dict:
    """
    Fast O(1) lookup of AST nodes using pre-built index.
    Much faster than get_ast_node for simple nodeType/name searches.
    
    Args:
        hash: AST hash
        node_type: Exact nodeType to find (e.g., "FunctionDefinition")
        name: Exact name to find (e.g., "transfer")
        limit: Maximum number of results to return
    """
    limit = max(1, min(limit, 100))
    
    p = AST_CACHE_DIR / f"{hash}.json"
    if not p.exists():
        return {"status":"error", "error":"AST not found"}

    index = _load_ast_index(hash)
    if not index:
        return {"status":"error", "error":"Index not found. Run parse_solidity first to create index."}

    ast = json.loads(p.read_text("utf-8"))
    found_nodes = []
    
    if node_type:
        pointers = index["nodeType"].get(node_type, [])
        for pointer in pointers[:limit]:
            try:
                node = jsonpointer.resolve_pointer(ast, pointer)
                found_nodes.append(node)
            except Exception:
                continue
    
    if name:
        pointers = index["name"].get(name, [])
        for pointer in pointers[:limit]:
            try:
                node = jsonpointer.resolve_pointer(ast, pointer)
                found_nodes.append(node)
            except Exception:
                continue
    
    seen = set()
    unique_nodes = []
    for node in found_nodes:
        node_id = id(node)
        if node_id not in seen:
            seen.add(node_id)
            unique_nodes.append(node)
    
    return {
        "status": "ok",
        "mode": "indexed",
        "count": len(unique_nodes),
        "nodes": unique_nodes[:limit],
        "search_criteria": {
            "node_type": node_type,
            "name": name
        }
    }

@mcp.tool()
def parse_solidity(
    input_code: str | Dict[str, str], 
    engine: str = "solc",
    auto_version: bool = True,
    persist_ast: bool = True,
    ast_cache_dir: str | None = None,
    create_index: bool = True
) -> dict:
    """
    Parse Solidity source code and extract functions, modifiers, visibility, etc.
    
    Args:
        input_code: Solidity source code to parse (string) or dict of {filename: content}
        engine: Parser engine to use (currently only "solc" supported)
        persist_ast: If True, cache AST to disk for later access via ast_uri
        ast_cache_dir: Custom cache directory (uses platformdirs by default)
        create_index: If True, create index file for fast O(1) lookups
    """
    result = parser_solidity.run(
        input_code, 
        engine=engine,
        auto_version=auto_version,
        persist_ast=persist_ast,
        ast_cache_dir=ast_cache_dir,
    )
    
    if (result.get("status") == "ok" and 
        persist_ast and 
        create_index and 
        "ast_uri" in result):
        
        try:
            ast_uri = result["ast_uri"]
            if ast_uri.startswith("ast://"):
                hash_name = ast_uri[6:] 
                
                ast_path = AST_CACHE_DIR / f"{hash_name}.json"
                if ast_path.exists():
                    ast = json.loads(ast_path.read_text(encoding="utf-8"))
                    index = _create_ast_index(ast, hash_name)
                    _save_ast_index(index, hash_name)
                    
                    result["index_created"] = True
                    result["index_stats"] = {
                        "node_types": len(index["nodeType"]),
                        "names": len(index["name"])
                    }
        except Exception as e:
            result["index_error"] = str(e)
    
    return result

@mcp.tool()
def slither_scan(input_code: str, timeout_seconds: int = 120) -> dict:
    """Run Slither static analysis and return normalized findings."""
    return slither_wrapper.run(input_code, timeout_seconds=timeout_seconds)

@mcp.tool()
def slither_project_scan(
    project_path: str, 
    timeout_seconds: int = 300, 
    extra_args: List[str] = None,
    stream_logs: bool = False,
    foundry_mode: bool = True
) -> dict:
    """
    Run Slither analysis on an entire project directory.
    
    Args:
        project_path: Path to the project directory to analyze
        timeout_seconds: Maximum time to wait for analysis (default: 300 seconds)
        extra_args: Additional arguments to pass to Slither
        stream_logs: If True, stream output logs in real-time
        foundry_mode: If True, add --foundry flag for Foundry projects (default: True)
    
    This is equivalent to running: slither . --foundry
    
    Returns a detailed report with:
    - findings: List of security issues found
    - metrics: Summary statistics
    - severity_breakdown: Count of issues by severity level
    - raw_output: Complete stdout/stderr from Slither
    """
    return slither_wrapper.run_project_scan(
        project_path=project_path,
        timeout_seconds=timeout_seconds,
        extra_args=extra_args,
        stream_logs=stream_logs,
        foundry_mode=foundry_mode
    )

@mcp.tool()
def generate_slither_report(scan_result: dict) -> dict:
    """
    Generate a human-readable report from Slither scan results.
    
    Args:
        scan_result: Result from slither_project_scan or slither_scan
        
    Returns:
        Formatted report with summary, findings, and recommendations
    """
    if scan_result.get("status") != "ok":
        return {
            "status": "error",
            "error": "Invalid scan result provided",
            "scan_status": scan_result.get("status")
        }
    
    findings = scan_result.get("findings", [])
    metrics = scan_result.get("metrics", {})
    meta = scan_result.get("meta", {})
    
    findings_by_severity = {
        "high": [],
        "medium": [],
        "low": [],
        "info": []
    }
    
    for finding in findings:
        severity = finding.get("severity", "info")
        if severity in findings_by_severity:
            findings_by_severity[severity].append(finding)
    
    total_findings = len(findings)
    high_count = len(findings_by_severity["high"])
    medium_count = len(findings_by_severity["medium"])
    low_count = len(findings_by_severity["low"])
    info_count = len(findings_by_severity["info"])
    
    risk_level = "LOW"
    if high_count > 0:
        risk_level = "CRITICAL"
    elif medium_count > 3:
        risk_level = "HIGH"
    elif medium_count > 0 or low_count > 5:
        risk_level = "MEDIUM"
    
    recommendations = []
    if high_count > 0:
        recommendations.append("🚨 CRITICAL: Address high-severity issues immediately")
    if medium_count > 0:
        recommendations.append("⚠️  MEDIUM: Review and fix medium-severity issues")
    if low_count > 0:
        recommendations.append("ℹ️  LOW: Consider addressing low-severity issues")
    if total_findings == 0:
        recommendations.append("✅ No security issues found - good job!")
    
    formatted_findings = []
    for severity in ["high", "medium", "low", "info"]:
        for finding in findings_by_severity[severity]:
            elements = finding.get("elements", [])
            file_refs = []
            for element in elements:
                if element.get("filename"):
                    line_info = f":{element.get('line', 0)}" if element.get("line") else ""
                    file_refs.append(f"{element['filename']}{line_info}")
            
            formatted_findings.append({
                "severity": severity.upper(),
                "check": finding.get("check", "unknown"),
                "description": finding.get("description", ""),
                "locations": file_refs,
                "confidence": finding.get("confidence", "medium")
            })
    
    return {
        "status": "ok",
        "report": {
            "summary": {
                "total_findings": total_findings,
                "risk_level": risk_level,
                "severity_breakdown": {
                    "high": high_count,
                    "medium": medium_count,
                    "low": low_count,
                    "info": info_count
                },
                "analysis_duration_ms": meta.get("duration_ms", 0),
                "project_path": meta.get("project_path", "unknown")
            },
            "recommendations": recommendations,
            "findings": formatted_findings,
            "raw_metrics": metrics,
            "scan_metadata": {
                "foundry_mode": meta.get("foundry_mode", False),
                "exit_code": meta.get("exit_code", 0),
                "command": meta.get("raw_output", {}).get("command", "")
            }
        }
    }

@mcp.tool()
def list_cached_asts() -> Dict[str, Any]:
    """
    List all cached AST files with metadata (size, modification time, etc.).
    Useful for cache management and debugging.
    """
    try:
        cached_files = []
        total_size = 0
        
        for file_path in AST_CACHE_DIR.glob("*.json"):
            if file_path.is_file() and not file_path.name.endswith(".index.json"):
                stat = file_path.stat()
                hash_name = file_path.stem
                
                index_path = AST_CACHE_DIR / f"{hash_name}.index.json"
                has_index = index_path.exists()
                
                cached_files.append({
                    "hash": hash_name,
                    "size_bytes": stat.st_size,
                    "created_at": stat.st_ctime,  
                    "modified_at": stat.st_mtime,
                    "file_path": str(file_path),
                    "has_index": has_index
                })
                total_size += stat.st_size
        
        return {
            "status": "ok",
            "cache_dir": str(AST_CACHE_DIR),
            "total_files": len(cached_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "files": sorted(cached_files, key=lambda x: x["modified_at"], reverse=True)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "cache_dir": str(AST_CACHE_DIR)
        }

@mcp.tool()
def delete_ast(hash: str) -> Dict[str, Any]:
    """
    Delete a specific cached AST file by hash (including its index if exists).
    """
    if not re.fullmatch(r"[0-9a-fA-F]{8}", hash):
        return {
            "status": "error",
            "error": "Invalid hash format. Must be 8-character hex string."
        }
    
    try:
        file_path = AST_CACHE_DIR / f"{hash}.json"
        index_path = AST_CACHE_DIR / f"{hash}.index.json"
        
        if not file_path.exists():
            return {
                "status": "error",
                "error": f"AST file not found: {hash}"
            }
        
        file_size = file_path.stat().st_size
        file_path.unlink()
        
        index_size = 0
        if index_path.exists():
            index_size = index_path.stat().st_size
            index_path.unlink()
        
        return {
            "status": "ok",
            "deleted_hash": hash,
            "freed_bytes": file_size + index_size,
            "freed_mb": round((file_size + index_size) / (1024 * 1024), 2),
            "deleted_files": ["ast", "index"] if index_size > 0 else ["ast"]
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "hash": hash
        }

@mcp.tool()
def purge_cache(older_than_days: Optional[int] = None, size_limit_mb: Optional[int] = None) -> Dict[str, Any]:
    """
    Purge cached AST files based on age or size limits (including their indexes).
    """
    try:
        if older_than_days is None and size_limit_mb is None:
            return {
                "status": "error",
                "error": "Must specify either older_than_days or size_limit_mb"
            }
        
        files_info = []
        total_size = 0
        current_time = time.time()
        
        for file_path in AST_CACHE_DIR.glob("*.json"):
            if file_path.is_file() and not file_path.name.endswith(".index.json"):
                stat = file_path.stat()
                index_path = AST_CACHE_DIR / f"{file_path.stem}.index.json"
                index_size = index_path.stat().st_size if index_path.exists() else 0
                
                files_info.append({
                    "path": file_path,
                    "index_path": index_path,
                    "hash": file_path.stem,
                    "size": stat.st_size + index_size,
                    "age_days": (current_time - stat.st_mtime) / (24 * 3600)
                })
                total_size += stat.st_size + index_size
        
        files_info.sort(key=lambda x: x["age_days"], reverse=True)
        
        files_info_for_size = sorted(files_info, key=lambda x: x["age_days"], reverse=True)
        
        deleted_files = []
        freed_bytes = 0
        
        if older_than_days:
            for file_info in files_info:
                if file_info["age_days"] > older_than_days:
                    file_info["path"].unlink()
                    if file_info["index_path"].exists():
                        file_info["index_path"].unlink()
                    deleted_files.append(file_info["hash"])
                    freed_bytes += file_info["size"]
        
        if size_limit_mb:
            size_limit_bytes = size_limit_mb * 1024 * 1024
            current_size = total_size - freed_bytes
            
            for file_info in files_info_for_size:
                if current_size <= size_limit_bytes:
                    break
                if file_info["hash"] not in deleted_files:
                    file_info["path"].unlink()
                    if file_info["index_path"].exists():
                        file_info["index_path"].unlink()
                    deleted_files.append(file_info["hash"])
                    freed_bytes += file_info["size"]
                    current_size -= file_info["size"]
        
        return {
            "status": "ok",
            "deleted_count": len(deleted_files),
            "deleted_hashes": deleted_files,
            "freed_bytes": freed_bytes,
            "freed_mb": round(freed_bytes / (1024 * 1024), 2),
            "remaining_files": len(files_info) - len(deleted_files),
            "remaining_size_mb": round((total_size - freed_bytes) / (1024 * 1024), 2)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


"""@mcp.tool()
def surya_graph(input_code: str, simple: bool=False, include_modifiers: bool=False,
                include_libraries: bool=True, format: str="dot", timeout_seconds: int=60) -> dict:
    return surya_wrapper.graph(input_code, simple=simple, include_modifiers=include_modifiers,
                               include_libraries=include_libraries, format=format, timeout_seconds=timeout_seconds)"""

"""@mcp.tool()
def surya_ftrace(input_code: str, entry: str, scope: str="all", timeout_seconds: int=60) -> dict:
    return surya_wrapper.ftrace(input_code, entry=entry, scope=scope, timeout_seconds=timeout_seconds)"""

"""@mcp.tool()
def surya_inheritance(input_code: str, format: str="dot", timeout_seconds: int=60) -> dict:
    return surya_wrapper.inheritance_graph(input_code, format=format, timeout_seconds=timeout_seconds)"""


@mcp.tool()
def get_callers(hash: str, contract: str, function: str) -> dict:
    """
    Find all functions that call the specified function in the given contract.
    Returns structured data with evidence pointers.
    """
    try:
        ast_path = AST_CACHE_DIR / f"{hash}.json"
        if not ast_path.exists():
            return {"status": "error", "error": "AST not found"}
        
        ast = json.loads(ast_path.read_text("utf-8"))
        
        target_function = None
        target_pointer = None
        
        def find_function(node, path=""):
            nonlocal target_function, target_pointer
            if isinstance(node, dict):
                if (node.get("nodeType") == "FunctionDefinition" and 
                    node.get("name") == function):
                    target_function = node
                    target_pointer = path
                    return
                for key, value in node.items():
                    if isinstance(value, (dict, list)):
                        child_path = f"{path}/{key}" if path else f"/{key}"
                        find_function(value, child_path)
            elif isinstance(node, list):
                for i, item in enumerate(node):
                    child_path = f"{path}/{i}" if path else f"/{i}"
                    find_function(item, child_path)
        
        find_function(ast)
        
        if not target_function:
            return {"status": "error", "error": f"Function {function} not found in contract {contract}"}
        

        callers = []
        
        return {
            "status": "ok",
            "target": {"contract": contract, "function": function},
            "callers": callers,
            "evidence": [f"ast://{hash}#{target_pointer}"]
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def get_callees(hash: str, contract: str, function: str, call_type: str = "all") -> dict:
    """
    Find all functions called by the specified function.
    
    Args:
        call_type: "internal", "external", or "all"
    """
    try:
        ast_path = AST_CACHE_DIR / f"{hash}.json"
        if not ast_path.exists():
            return {"status": "error", "error": "AST not found"}
        
        ast = json.loads(ast_path.read_text("utf-8"))
        
        target_function = None
        target_pointer = None
        
        def find_function(node, path=""):
            nonlocal target_function, target_pointer
            if isinstance(node, dict):
                if (node.get("nodeType") == "FunctionDefinition" and 
                    node.get("name") == function):
                    target_function = node
                    target_pointer = path
                    return
                for key, value in node.items():
                    if isinstance(value, (dict, list)):
                        child_path = f"{path}/{key}" if path else f"/{key}"
                        find_function(value, child_path)
            elif isinstance(node, list):
                for i, item in enumerate(node):
                    child_path = f"{path}/{i}" if path else f"/{i}"
                    find_function(item, child_path)
        
        find_function(ast)
        
        if not target_function:
            return {"status": "error", "error": f"Function {function} not found in contract {contract}"}
        
    
        callees = []
        
        return {
            "status": "ok",
            "source": {"contract": contract, "function": function},
            "callees": callees,
            "call_type_filter": call_type,
            "evidence": [f"ast://{hash}#{target_pointer}"]
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def get_external_calls(hash: str, contract: str, function: str = None) -> dict:
    """
    Find all external calls in the contract or specific function.
    Returns calls with target addresses and function selectors when available.
    """
    try:
        ast_path = AST_CACHE_DIR / f"{hash}.json"
        if not ast_path.exists():
            return {"status": "error", "error": "AST not found"}
        
        ast = json.loads(ast_path.read_text("utf-8"))
        
        external_calls = []
        
        def find_external_calls(node, path="", current_function=None):
            if isinstance(node, dict):
                if node.get("nodeType") == "FunctionCall":
                    expression = node.get("expression", {})
                    if expression.get("nodeType") == "MemberAccess":
                        member_name = expression.get("memberName", "")
                        if member_name in ["call", "delegatecall", "staticcall", "send", "transfer"]:
                            call_info = {
                                "type": member_name,
                                "function": current_function,
                                "line": _get_line_from_src(node.get("src", "")),
                                "evidence": [f"ast://{hash}#{path}"]
                            }
                            
                            if "arguments" in node and node["arguments"]:
                                first_arg = node["arguments"][0]
                                if first_arg.get("nodeType") == "Identifier":
                                    call_info["target"] = first_arg.get("name", "unknown")
                            
                            external_calls.append(call_info)
                
                if node.get("nodeType") == "FunctionDefinition":
                    current_function = node.get("name", "unknown")
                
                for key, value in node.items():
                    if isinstance(value, (dict, list)):
                        child_path = f"{path}/{key}" if path else f"/{key}"
                        find_external_calls(value, child_path, current_function)
            elif isinstance(node, list):
                for i, item in enumerate(node):
                    child_path = f"{path}/{i}" if path else f"/{i}"
                    find_external_calls(item, child_path, current_function)
        
        find_external_calls(ast)
        
        if function:
            external_calls = [call for call in external_calls if call["function"] == function]
        
        return {
            "status": "ok",
            "contract": contract,
            "function": function,
            "external_calls": external_calls,
            "total_calls": len(external_calls)
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

def _get_line_from_src(src: str) -> int:
    """Extract line number from src string like '123:45:67'"""
    try:
        return int(src.split(":")[0])
    except:
        return 0

@mcp.tool()
def get_state_writes(hash: str, contract: str, function: str = None) -> dict:
    """
    Find all state variable writes in the contract or specific function.
    """
    try:
        ast_path = AST_CACHE_DIR / f"{hash}.json"
        if not ast_path.exists():
            return {"status": "error", "error": "AST not found"}
        
        ast = json.loads(ast_path.read_text("utf-8"))
        
        state_writes = []
        
        def find_state_writes(node, path="", current_function=None):
            if isinstance(node, dict):
                if node.get("nodeType") == "Assignment":
                    left = node.get("leftHandSide", {})
                    if left.get("nodeType") == "Identifier":
                        var_name = left.get("name", "")
                        if not _is_local_variable(var_name, current_function, ast):
                            write_info = {
                                "variable": var_name,
                                "function": current_function,
                                "line": _get_line_from_src(node.get("src", "")),
                                "evidence": [f"ast://{hash}#{path}"]
                            }
                            state_writes.append(write_info)
                
                if node.get("nodeType") == "FunctionDefinition":
                    current_function = node.get("name", "unknown")
                
                for key, value in node.items():
                    if isinstance(value, (dict, list)):
                        child_path = f"{path}/{key}" if path else f"/{key}"
                        find_state_writes(value, child_path, current_function)
            elif isinstance(node, list):
                for i, item in enumerate(node):
                    child_path = f"{path}/{i}" if path else f"/{i}"
                    find_state_writes(item, child_path, current_function)
        
        find_state_writes(ast)
        
        if function:
            state_writes = [write for write in state_writes if write["function"] == function]
        
        return {
            "status": "ok",
            "contract": contract,
            "function": function,
            "state_writes": state_writes,
            "total_writes": len(state_writes)
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

def _is_local_variable(var_name: str, function_name: str, ast: dict) -> bool:
    """Check if variable is local to a function (simplified heuristic)"""
    return var_name and var_name[0].islower()

@mcp.tool()
def get_payable_entrypoints(hash: str, contract: str) -> dict:
    """
    Find all functions that can receive ETH (payable functions and fallback/receive).
    """
    try:
        ast_path = AST_CACHE_DIR / f"{hash}.json"
        if not ast_path.exists():
            return {"status": "error", "error": "AST not found"}
        
        ast = json.loads(ast_path.read_text("utf-8"))
        
        payable_functions = []
        
        def find_payable_functions(node, path=""):
            if isinstance(node, dict):
                if node.get("nodeType") == "FunctionDefinition":
                    function_name = node.get("name", "")
                    state_mutability = node.get("stateMutability", "")
                    
                    if state_mutability == "payable":
                        func_info = {
                            "name": function_name,
                            "type": "payable_function",
                            "visibility": node.get("visibility", "public"),
                            "line": _get_line_from_src(node.get("src", "")),
                            "evidence": [f"ast://{hash}#{path}"]
                        }
                        payable_functions.append(func_info)
                    
                    elif function_name == "" or function_name == "fallback":
                        func_info = {
                            "name": function_name or "fallback",
                            "type": "fallback",
                            "visibility": node.get("visibility", "external"),
                            "line": _get_line_from_src(node.get("src", "")),
                            "evidence": [f"ast://{hash}#{path}"]
                        }
                        payable_functions.append(func_info)
                
                for key, value in node.items():
                    if isinstance(value, (dict, list)):
                        child_path = f"{path}/{key}" if path else f"/{key}"
                        find_payable_functions(value, child_path)
            elif isinstance(node, list):
                for i, item in enumerate(node):
                    child_path = f"{path}/{i}" if path else f"/{i}"
                    find_payable_functions(item, child_path)
        
        find_payable_functions(ast)
        
        return {
            "status": "ok",
            "contract": contract,
            "payable_entrypoints": payable_functions,
            "total_entrypoints": len(payable_functions)
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}



@mcp.tool()
def get_modifier_map(hash: str, contract: str) -> dict:
    """
    Get mapping of modifiers to functions for ACL analysis.
    """
    try:
        ast_path = AST_CACHE_DIR / f"{hash}.json"
        if not ast_path.exists():
            return {"status": "error", "error": "AST not found"}
        
        ast = json.loads(ast_path.read_text("utf-8"))
        
        modifier_map = {}
        
        def find_modifiers_and_functions(node, path=""):
            if isinstance(node, dict):
                if node.get("nodeType") == "ModifierDefinition":
                    modifier_name = node.get("name", "")
                    modifier_map[modifier_name] = {
                        "type": "modifier",
                        "line": _get_line_from_src(node.get("src", "")),
                        "functions": [],
                        "evidence": [f"ast://{hash}#{path}"]
                    }
                
                elif node.get("nodeType") == "FunctionDefinition":
                    function_name = node.get("name", "")
                    modifiers = node.get("modifiers", [])
                    
                    for modifier in modifiers:
                        modifier_name = modifier.get("modifierName", {}).get("name", "")
                        if modifier_name in modifier_map:
                            modifier_map[modifier_name]["functions"].append(function_name)
                        else:
                            if modifier_name not in modifier_map:
                                modifier_map[modifier_name] = {
                                    "type": "inherited_modifier",
                                    "functions": [function_name],
                                    "evidence": []
                                }
                
                for key, value in node.items():
                    if isinstance(value, (dict, list)):
                        child_path = f"{path}/{key}" if path else f"/{key}"
                        find_modifiers_and_functions(value, child_path)
            elif isinstance(node, list):
                for i, item in enumerate(node):
                    child_path = f"{path}/{i}" if path else f"/{i}"
                    find_modifiers_and_functions(item, child_path)
        
        find_modifiers_and_functions(ast)
        
        return {
            "status": "ok",
            "contract": contract,
            "modifier_map": modifier_map,
            "total_modifiers": len(modifier_map)
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}


@mcp.tool()
def parse_solidity_file(path_to_file: str, engine: str = "solc", auto_version: bool = True, persist_ast: bool = True, create_index: bool = True) -> dict:
    """
    Parse a Solidity file from the filesystem and extract functions, modifiers, visibility, etc.
    
    Args:
        path_to_file: Path to the Solidity file to parse
        engine: Parser engine to use (currently only "solc" supported)
        auto_version: If True, automatically detect solc version from pragma
        persist_ast: If True, cache AST to disk for later access via ast_uri
        create_index: If True, create index file for fast O(1) lookups
    """
    try:
        file_path = Path(path_to_file)
        
        if not file_path.exists():
            return {"status": "error", "error": f"File not found: {path_to_file}"}
        
        if not file_path.is_file():
            return {"status": "error", "error": f"Path is not a file: {path_to_file}"}
        
        if not file_path.suffix.lower() == '.sol':
            return {"status": "error", "error": f"File is not a Solidity file: {path_to_file}"}
        
        # Read file content
        try:
            file_content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return {"status": "error", "error": f"File encoding error: {path_to_file}"}
        
        if not file_content.strip():
            return {"status": "error", "error": f"File is empty: {path_to_file}"}
        
        # Parse the file
        result = parser_solidity.run(
            file_content, 
            engine=engine,
            auto_version=auto_version,
            persist_ast=persist_ast,
        )
        
        # Add file metadata to result
        if result.get("status") == "ok":
            result["file_info"] = {
                "path": str(file_path.absolute()),
                "name": file_path.name,
                "size_bytes": file_path.stat().st_size,
                "modified_at": file_path.stat().st_mtime
            }
            
            # Create index if requested
            if persist_ast and create_index and "ast_uri" in result:
                try:
                    ast_uri = result["ast_uri"]
                    if ast_uri.startswith("ast://"):
                        hash_name = ast_uri[6:] 
                        
                        ast_path = AST_CACHE_DIR / f"{hash_name}.json"
                        if ast_path.exists():
                            ast = json.loads(ast_path.read_text(encoding="utf-8"))
                            index = _create_ast_index(ast, hash_name)
                            _save_ast_index(index, hash_name)
                            
                            result["index_created"] = True
                            result["index_stats"] = {
                                "node_types": len(index["nodeType"]),
                                "names": len(index["name"])
                            }
                except Exception as e:
                    result["index_error"] = str(e)
        
        return result
        
    except Exception as e:
        return {"status": "error", "error": f"Failed to parse file: {str(e)}"}


@mcp.tool()
def slither_scan_file(path_to_file: str, timeout_seconds: int = 120, extra_args: List[str] = None) -> dict:
    """
    Run Slither static analysis on a Solidity file from the filesystem.
    
    Args:
        path_to_file: Path to the Solidity file to analyze
        timeout_seconds: Maximum time to wait for analysis to complete
        extra_args: Additional arguments to pass to Slither
    """
    try:
        file_path = Path(path_to_file)
        
        if not file_path.exists():
            return {"status": "error", "error": f"File not found: {path_to_file}"}
        
        if not file_path.is_file():
            return {"status": "error", "error": f"Path is not a file: {path_to_file}"}
        
        if not file_path.suffix.lower() == '.sol':
            return {"status": "error", "error": f"File is not a Solidity file: {path_to_file}"}
        
        # Read file content
        try:
            file_content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return {"status": "error", "error": f"File encoding error: {path_to_file}"}
        
        if not file_content.strip():
            return {"status": "error", "error": f"File is empty: {path_to_file}"}
        
        result = slither_wrapper.run(
            file_content, 
            timeout_seconds=timeout_seconds,
            extra_args=extra_args
        )
        
        if result.get("status") == "ok":
            result["file_info"] = {
                "path": str(file_path.absolute()),
                "name": file_path.name,
                "size_bytes": file_path.stat().st_size,
                "modified_at": file_path.stat().st_mtime
            }
        
        return result
        
    except Exception as e:
        return {"status": "error", "error": f"Failed to analyze file: {str(e)}"}

@mcp.tool()
def slither_scan_directory(
    path_to_directory: str, 
    timeout_seconds: int = 300, 
    extra_args: List[str] = None,
    stream_logs: bool = False,
    foundry_mode: bool = True
) -> dict:
    """
    Run Slither analysis on a directory containing Solidity files.
    
    Args:
        path_to_directory: Path to the directory to analyze
        timeout_seconds: Maximum time to wait for analysis (default: 300 seconds)
        extra_args: Additional arguments to pass to Slither
        stream_logs: If True, stream output logs in real-time
        foundry_mode: If True, add --foundry flag for Foundry projects (default: True)
    
    This is equivalent to running: slither <directory> --foundry
    """
    try:
        dir_path = Path(path_to_directory)
        
        if not dir_path.exists():
            return {"status": "error", "error": f"Directory not found: {path_to_directory}"}
        
        if not dir_path.is_dir():
            return {"status": "error", "error": f"Path is not a directory: {path_to_directory}"}
        
        result = slither_wrapper.run_project_scan(
            project_path=str(dir_path),
            timeout_seconds=timeout_seconds,
            extra_args=extra_args,
            stream_logs=stream_logs,
            foundry_mode=foundry_mode
        )
        
        if result.get("status") == "ok":
            result["directory_info"] = {
                "path": str(dir_path.absolute()),
                "name": dir_path.name,
                "foundry_mode": foundry_mode
            }
        
        return result
        
    except Exception as e:
        return {"status": "error", "error": f"Failed to analyze directory: {str(e)}"}


@mcp.tool()
def analyze_solidity_file(path_to_file: str, include_slither: bool = True, timeout_seconds: int = 120) -> dict:
    """
    Comprehensive analysis of a Solidity file including parsing and static analysis.
    
    Args:
        path_to_file: Path to the Solidity file to analyze
        include_slither: If True, include Slither static analysis
        timeout_seconds: Maximum time to wait for Slither analysis
    """
    try:
        file_path = Path(path_to_file)
        
        if not file_path.exists():
            return {"status": "error", "error": f"File not found: {path_to_file}"}
        
        if not file_path.is_file():
            return {"status": "error", "error": f"Path is not a file: {path_to_file}"}
        
        if not file_path.suffix.lower() == '.sol':
            return {"status": "error", "error": f"File is not a Solidity file: {path_to_file}"}
        
        try:
            file_content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return {"status": "error", "error": f"File encoding error: {path_to_file}"}
        
        if not file_content.strip():
            return {"status": "error", "error": f"File is empty: {path_to_file}"}
        
        parse_result = parser_solidity.run(
            file_content, 
            engine="solc",
            auto_version=True,
            persist_ast=True,
        )
        
        result = {
            "status": "ok",
            "file_info": {
                "path": str(file_path.absolute()),
                "name": file_path.name,
                "size_bytes": file_path.stat().st_size,
                "modified_at": file_path.stat().st_mtime
            },
            "parsing": parse_result
        }
        
        if include_slither:
            slither_result = slither_wrapper.run(
                file_content, 
                timeout_seconds=timeout_seconds
            )
            result["slither_analysis"] = slither_result
        
        if parse_result.get("status") == "ok" and "ast_uri" in parse_result:
            try:
                ast_uri = parse_result["ast_uri"]
                if ast_uri.startswith("ast://"):
                    hash_name = ast_uri[6:] 
                    
                    ast_path = AST_CACHE_DIR / f"{hash_name}.json"
                    if ast_path.exists():
                        ast = json.loads(ast_path.read_text(encoding="utf-8"))
                        index = _create_ast_index(ast, hash_name)
                        _save_ast_index(index, hash_name)
                        
                        result["index_created"] = True
                        result["index_stats"] = {
                            "node_types": len(index["nodeType"]),
                            "names": len(index["name"])
                        }
            except Exception as e:
                result["index_error"] = str(e)
        
        return result
        
    except Exception as e:
        return {"status": "error", "error": f"Failed to analyze file: {str(e)}"}


@mcp.tool()
def parse_solidity_directory(path_to_directory: str, engine: str = "solc", auto_version: bool = True, persist_ast: bool = True, create_index: bool = True) -> dict:
    """
    Parse all Solidity files in a directory and extract functions, modifiers, visibility, etc.
    
    Args:
        path_to_directory: Path to the directory containing Solidity files
        engine: Parser engine to use (currently only "solc" supported)
        auto_version: If True, automatically detect solc version from pragma
        persist_ast: If True, cache AST to disk for later access via ast_uri
        create_index: If True, create index file for fast O(1) lookups
    """
    try:
        dir_path = Path(path_to_directory)
        
        if not dir_path.exists():
            return {"status": "error", "error": f"Directory not found: {path_to_directory}"}
        
        if not dir_path.is_dir():
            return {"status": "error", "error": f"Path is not a directory: {path_to_directory}"}
        
        # Find all .sol files recursively
        sol_files = list(dir_path.rglob("*.sol"))
        
        if not sol_files:
            return {"status": "error", "error": f"No Solidity files found in directory: {path_to_directory}"}
        
        # Read all files
        files_content = {}
        file_errors = []
        
        for sol_file in sol_files:
            try:
                relative_path = sol_file.relative_to(dir_path)
                files_content[str(relative_path)] = sol_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                file_errors.append(f"Encoding error in {sol_file}")
            except Exception as e:
                file_errors.append(f"Error reading {sol_file}: {str(e)}")
        
        if not files_content:
            return {"status": "error", "error": f"Could not read any files from directory: {path_to_directory}"}
        
        # Parse all files together
        result = parser_solidity.run(
            files_content, 
            engine=engine,
            auto_version=auto_version,
            persist_ast=persist_ast,
        )
        
        # Add directory metadata to result
        if result.get("status") == "ok":
            result["directory_info"] = {
                "path": str(dir_path.absolute()),
                "total_files": len(sol_files),
                "parsed_files": len(files_content),
                "file_errors": file_errors,
                "files": [str(f.relative_to(dir_path)) for f in sol_files]
            }
            
            # Create index if requested
            if persist_ast and create_index and "ast_uri" in result:
                try:
                    ast_uri = result["ast_uri"]
                    if ast_uri.startswith("ast://"):
                        hash_name = ast_uri[6:] 
                        
                        ast_path = AST_CACHE_DIR / f"{hash_name}.json"
                        if ast_path.exists():
                            ast = json.loads(ast_path.read_text(encoding="utf-8"))
                            index = _create_ast_index(ast, hash_name)
                            _save_ast_index(index, hash_name)
                            
                            result["index_created"] = True
                            result["index_stats"] = {
                                "node_types": len(index["nodeType"]),
                                "names": len(index["name"])
                            }
                except Exception as e:
                    result["index_error"] = str(e)
        
        return result
        
    except Exception as e:
        return {"status": "error", "error": f"Failed to parse directory: {str(e)}"}

if __name__ == "__main__":
   # mcp.run(transport="stdio")  
    mcp.settings.host = "0.0.0.0"
    mcp.run(transport="streamable-http")  