from mcp_modules import parser_solidity, slither_wrapper   
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
    
    # Remove duplicates while preserving order
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

if __name__ == "__main__":
    mcp.run(transport="streamable-http")  