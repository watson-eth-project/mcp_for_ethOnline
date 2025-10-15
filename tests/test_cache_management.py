import pytest
import tempfile
import shutil
import os
from pathlib import Path
from mcp_modules.parser_solidity import run

import sys
sys.path.append(str(Path(__file__).parent.parent))
from server import list_cached_asts, delete_ast, purge_cache, _get_ast_cache_dir

SIMPLE_CODE = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

contract TestContract {
    function test() public pure returns (uint256) {
        return 42;
    }
}
"""

def test_list_cached_asts_empty():
    """Test listing cached ASTs when cache is empty."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cache_dir = _get_ast_cache_dir()
        
        cache_dir = Path(temp_dir) / "ast_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        import server
        server.AST_CACHE_DIR = cache_dir
        
        try:
            result = list_cached_asts()
            assert result["status"] == "ok"
            assert result["total_files"] == 0
            assert result["total_size_bytes"] == 0
            assert result["files"] == []
        finally:
            server.AST_CACHE_DIR = original_cache_dir


def test_list_cached_asts_with_files():
    """Test listing cached ASTs when files exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir) / "ast_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        test_files = [
            ("abc12345.json", '{"test": "data1"}'),
            ("def67890.json", '{"test": "data2"}'),
        ]
        
        for filename, content in test_files:
            file_path = cache_dir / filename
            file_path.write_text(content)
        
        import server
        original_cache_dir = server.AST_CACHE_DIR
        server.AST_CACHE_DIR = cache_dir
        
        try:
            result = list_cached_asts()
            assert result["status"] == "ok"
            assert result["total_files"] == 2
            assert result["total_size_bytes"] > 0
            
            files = result["files"]
            assert len(files) == 2
            
            for file_info in files:
                assert "hash" in file_info
                assert "size_bytes" in file_info
                assert "created_at" in file_info
                assert "file_path" in file_info
        finally:
            server.AST_CACHE_DIR = original_cache_dir


def test_delete_ast_valid():
    """Test deleting a valid AST file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir) / "ast_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        test_hash = "abc12345"
        test_file = cache_dir / f"{test_hash}.json"
        test_content = '{"test": "data"}'
        test_file.write_text(test_content)
        
        import server
        original_cache_dir = server.AST_CACHE_DIR
        server.AST_CACHE_DIR = cache_dir
        
        try:
            result = delete_ast(test_hash)
            assert result["status"] == "ok"
            assert result["deleted_hash"] == test_hash
            assert result["freed_bytes"] == len(test_content)
            assert not test_file.exists()
        finally:
            server.AST_CACHE_DIR = original_cache_dir


def test_delete_ast_invalid_hash():
    """Test deleting with invalid hash format."""
    result = delete_ast("invalid")
    assert result["status"] == "error"
    assert "Invalid hash format" in result["error"]


def test_delete_ast_not_found():
    """Test deleting non-existent AST file."""
    result = delete_ast("12345678")
    assert result["status"] == "error"
    assert "AST file not found" in result["error"]


def test_purge_cache_by_age():
    """Test purging cache by age."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir) / "ast_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        test_files = [
            ("old12345.json", '{"old": "data"}'),
            ("new67890.json", '{"new": "data"}'),
        ]
        
        import time
        old_time = time.time() - (2 * 24 * 3600)  
        
        for filename, content in test_files:
            file_path = cache_dir / filename
            file_path.write_text(content)
            os.utime(file_path, (old_time, old_time))
        
        import server
        original_cache_dir = server.AST_CACHE_DIR
        server.AST_CACHE_DIR = cache_dir
        
        try:
            result = purge_cache(older_than_days=1)
            assert result["status"] == "ok"
            assert result["deleted_count"] == 2
            assert len(result["deleted_hashes"]) == 2
        finally:
            server.AST_CACHE_DIR = original_cache_dir


def test_purge_cache_by_size():
    """Test purging cache by size limit."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir) / "ast_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        test_files = [
            ("file1.json", '{"data": "' + "x" * 1000 + '"}'),  
            ("file2.json", '{"data": "' + "y" * 1000 + '"}'),  
        ]
        
        for filename, content in test_files:
            file_path = cache_dir / filename
            file_path.write_text(content)
        
        import server
        original_cache_dir = server.AST_CACHE_DIR
        server.AST_CACHE_DIR = cache_dir
        
        try:
            result = purge_cache(size_limit_mb=0.001) 
            print(f"Purge by size result: {result}")  
            assert result["status"] == "ok"
            assert result["deleted_count"] > 0
        finally:
            server.AST_CACHE_DIR = original_cache_dir


def test_purge_cache_no_params():
    """Test purging cache without parameters."""
    result = purge_cache()
    assert result["status"] == "error"
    assert "Must specify either" in result["error"]
