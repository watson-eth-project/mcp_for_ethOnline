# mcp_modules/common/errors.py
"""
Centralized error handling and error codes for MCP Solidity Analysis Tools.
"""

from .normalize import detect_pragma
from typing import Dict, Any, List
from enum import Enum


class ErrorCode(Enum):
    """Standardized error codes for the MCP Solidity Analysis Tools."""
    
    # Parser errors
    EMPTY_INPUT = "EmptyInput"
    ENGINE_NOT_IMPLEMENTED = "EngineNotImplemented"
    SOLC_INSTALL_FAILED = "SolcInstallFailed"
    SOLC_NOT_INSTALLED = "SolcNotInstalled"
    RUNTIME_ERROR = "RuntimeError"
    
    # Slither errors
    SLITHER_NOT_FOUND = "SlitherNotFound"
    SLITHER_FAILED = "SlitherFailed"
    NO_JSON_PRODUCED = "NoJSONProduced"
    INVALID_JSON = "InvalidJSON"
    TIMEOUT = "Timeout"
    NON_ZERO_EXIT = "NonZeroExit"


class ErrorHandler:
    """Centralized error handling utilities."""
    
    @staticmethod
    def create_parser_error(
        error_code: ErrorCode,
        message: str = "",
        duration_ms: int = 0,
        engine: str = "solc",
        solc_version: str | None = None,
        **extra_meta: Any
    ) -> Dict[str, Any]:
        """Create a standardized parser error response."""
        return {
            "status": "error",
            "module": "parser_solidity",
            "errors": [error_code.value, message] if message else [error_code.value],
            "warnings": [],
            "meta": {
                "duration_ms": duration_ms,
                "engine": engine,
                "solc_version": solc_version,
                **extra_meta
            }
        }
    
    @staticmethod
    def create_slither_error(
        error_code: ErrorCode,
        message: str = "",
        duration_ms: int = 0,
        input_code: str | None = None,
        warnings: List[str] | None = None,
        solc_version: str | None = None,
        solc_bin: str | None = None,
        **extra_meta: Any
    ) -> Dict[str, Any]:
        """Create a standardized Slither error response."""
        errors = [error_code.value]
        if message:
            errors.append(message)
        
        return {
            "status": "error",
            "module": "slither_wrapper",
            "errors": errors,
            "warnings": warnings or [],
            "findings": [],
            "metrics": {
                "count": 0,
                "solc_version": str(solc_version) if solc_version else None,
                "pragma": detect_pragma(input_code) if input_code else None,
            },
            "meta": {
                "duration_ms": duration_ms,
                "solc_version": str(solc_version) if solc_version else None,
                "solc_bin": solc_bin,
                **extra_meta
            }
        }
    
    @staticmethod
    def parse_runtime_error(error_msg: str) -> ErrorCode:
        """Parse RuntimeError message to determine specific error code."""
        if error_msg.startswith("SolcInstallFailed"):
            return ErrorCode.SOLC_INSTALL_FAILED
        elif "SolcNotInstalled" in error_msg:
            return ErrorCode.SOLC_NOT_INSTALLED
        else:
            return ErrorCode.RUNTIME_ERROR
    


# Convenience functions for common error patterns
def empty_input_error(duration_ms: int = 0, engine: str = "solc", solc_version: str | None = None) -> Dict[str, Any]:
    """Create empty input error."""
    return ErrorHandler.create_parser_error(
        ErrorCode.EMPTY_INPUT,
        duration_ms=duration_ms,
        engine=engine,
        solc_version=solc_version
    )


def engine_not_implemented_error(engine: str, duration_ms: int = 0) -> Dict[str, Any]:
    """Create engine not implemented error."""
    return ErrorHandler.create_parser_error(
        ErrorCode.ENGINE_NOT_IMPLEMENTED,
        message=f"{engine}",
        duration_ms=duration_ms,
        engine=engine
    )


def solc_install_failed_error(version: str, error: str, duration_ms: int = 0, engine: str = "solc") -> Dict[str, Any]:
    """Create Solc installation failed error."""
    return ErrorHandler.create_parser_error(
        ErrorCode.SOLC_INSTALL_FAILED,
        message=f"{version}:{error}",
        duration_ms=duration_ms,
        engine=engine
    )


def solc_not_installed_error(duration_ms: int = 0, engine: str = "solc") -> Dict[str, Any]:
    """Create Solc not installed error."""
    return ErrorHandler.create_parser_error(
        ErrorCode.SOLC_NOT_INSTALLED,
        duration_ms=duration_ms,
        engine=engine
    )


def slither_not_found_error(error: str, duration_ms: int = 0, input_code: str | None = None, 
                          solc_version: str | None = None) -> Dict[str, Any]:
    """Create Slither not found error."""
    return ErrorHandler.create_slither_error(
        ErrorCode.SLITHER_NOT_FOUND,
        message=error,
        duration_ms=duration_ms,
        input_code=input_code,
        solc_version=solc_version
    )


def slither_failed_error(exit_code: int, stderr: str, duration_ms: int = 0, 
                        input_code: str | None = None, warnings: List[str] | None = None,
                        solc_version: str | None = None, solc_bin: str | None = None) -> Dict[str, Any]:
    """Create Slither failed error."""
    return ErrorHandler.create_slither_error(
        ErrorCode.SLITHER_FAILED,
        message=f"code={exit_code}",
        duration_ms=duration_ms,
        input_code=input_code,
        warnings=warnings,
        solc_version=solc_version,
        solc_bin=solc_bin
    )


def timeout_error(duration_ms: int = 0, input_code: str | None = None, 
                 solc_version: str | None = None) -> Dict[str, Any]:
    """Create timeout error."""
    return ErrorHandler.create_slither_error(
        ErrorCode.TIMEOUT,
        duration_ms=duration_ms,
        input_code=input_code,
        solc_version=solc_version
    )


def no_json_produced_error(stderr: str, duration_ms: int = 0, input_code: str | None = None,
                          warnings: List[str] | None = None, solc_version: str | None = None,
                          solc_bin: str | None = None) -> Dict[str, Any]:
    """Create no JSON produced error."""
    return ErrorHandler.create_slither_error(
        ErrorCode.NO_JSON_PRODUCED,
        message=stderr,
        duration_ms=duration_ms,
        input_code=input_code,
        warnings=warnings,
        solc_version=solc_version,
        solc_bin=solc_bin
    )


def invalid_json_error(error: str, duration_ms: int = 0, input_code: str | None = None,
                     solc_version: str | None = None) -> Dict[str, Any]:
    """Create invalid JSON error."""
    return ErrorHandler.create_slither_error(
        ErrorCode.INVALID_JSON,
        message=error,
        duration_ms=duration_ms,
        input_code=input_code,
        solc_version=solc_version
    )
