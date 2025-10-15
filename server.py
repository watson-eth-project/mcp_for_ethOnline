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

    def find_by_match(node, m):
        found = []
        if isinstance(node, dict):
            ok = all(node.get(k) == v for k,v in m.items())
            if ok:
                found.append(node)
            for v in node.values():
                found.extend(find_by_match(v, m))
        elif isinstance(node, list):
            for it in node:
                found.extend(find_by_match(it, m))
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
        nodes = find_by_match(ast, match)
        return {"status":"ok", "mode":"match", "count": len(nodes), "nodes": nodes[:limit]}

    return {"status":"error", "error":"Provide src or pointer or match"}

@mcp.tool()
def parse_solidity(
    input_code: str, 
    engine: str = "solc",
    auto_version: bool = True,
    persist_ast: bool = True,
    ast_cache_dir: str | None = None
) -> dict:
    """
    Parse Solidity source code and extract functions, modifiers, visibility, etc.
    
    Args:
        input_code: Solidity source code to parse
        engine: Parser engine to use (currently only "solc" supported)
        persist_ast: If True, cache AST to disk for later access via ast_uri
        ast_cache_dir: Custom cache directory (uses platformdirs by default)
    """
    return parser_solidity.run(
        input_code, 
        engine=engine,
        auto_version=auto_version,
        persist_ast=persist_ast,
        ast_cache_dir=ast_cache_dir,
    )

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
            if file_path.is_file():
                stat = file_path.stat()
                hash_name = file_path.stem
                
                cached_files.append({
                    "hash": hash_name,
                    "size_bytes": stat.st_size,
                    "created_at": stat.st_ctime,  
                    "modified_at": stat.st_mtime,
                    "file_path": str(file_path)
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
    Delete a specific cached AST file by hash.
    """
    if not re.fullmatch(r"[0-9a-fA-F]{8}", hash):
        return {
            "status": "error",
            "error": "Invalid hash format. Must be 8-character hex string."
        }
    
    try:
        file_path = AST_CACHE_DIR / f"{hash}.json"
        if not file_path.exists():
            return {
                "status": "error",
                "error": f"AST file not found: {hash}"
            }
        
        file_size = file_path.stat().st_size
        file_path.unlink()
        
        return {
            "status": "ok",
            "deleted_hash": hash,
            "freed_bytes": file_size,
            "freed_mb": round(file_size / (1024 * 1024), 2)
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
    Purge cached AST files based on age or size limits.
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
            if file_path.is_file():
                stat = file_path.stat()
                files_info.append({
                    "path": file_path,
                    "hash": file_path.stem,
                    "size": stat.st_size,
                    "age_days": (current_time - stat.st_mtime) / (24 * 3600)
                })
                total_size += stat.st_size
        
        files_info.sort(key=lambda x: x["age_days"], reverse=True)
        
        files_info_for_size = sorted(files_info, key=lambda x: x["age_days"], reverse=True)
        
        deleted_files = []
        freed_bytes = 0
        
        if older_than_days:
            for file_info in files_info:
                if file_info["age_days"] > older_than_days:
                    file_info["path"].unlink()
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