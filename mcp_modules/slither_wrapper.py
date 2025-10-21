from __future__ import annotations

import json
import re
import subprocess
import tempfile
import textwrap
import time
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Literal
import shutil

from .common.normalize import (
    extract_pragma,
    detect_pragma,
    ensure_solc_auto_by_pragma,
    ensure_solc,
)

from .common.errors import (
    ErrorHandler,
    ErrorCode,
    slither_not_found_error,
    slither_failed_error,
    timeout_error,
    no_json_produced_error,
    invalid_json_error
)

from .common.types import (
    SlitherResult,
    SlitherFinding,
    SlitherElement,
    SlitherMetrics,
    SlitherMeta
)




def _resolve_solc_bin(version: str) -> str:
    """
    Return the path to the solc binary for the specified version.
    """
    home_bin = Path.home() / ".solcx" / f"solc-v{version}"

    if home_bin.exists():
        return str(home_bin)

    if (home_bin.with_suffix(".exe")).exists():
        return str(home_bin.with_suffix(".exe"))

    solc_in_path = shutil.which("solc")
    return solc_in_path or "solc"
    

def _normalize_detectors(detectors: List[Dict[str, Any]]) -> List[SlitherFinding]:
    """Normalize the Slither output to a compact list of findings."""
    out: List[SlitherFinding] = []
    for d in detectors or []:
        check = d.get("check") or d.get("id") or ""
        impact = d.get("impact") or d.get("severity") or ""
        confidence = d.get("confidence") or ""
        desc = (d.get("description") or "").strip()
        elements_norm: List[SlitherElement] = []
        for e in d.get("elements", []) or []:
            el: SlitherElement = {
                "type": e.get("type"),
                "name": e.get("name") or e.get("function") or "",
            }
            sm = e.get("source_mapping") or {}
            el["line"] = sm.get("lines", [sm.get("start")])[0] if isinstance(sm.get("lines"), list) else sm.get("start")
            el["filename"] = sm.get("filename_absolute") or sm.get("filename_relative") or sm.get("filename")
            elements_norm.append(el)
        finding: SlitherFinding = {
            "check": check,
            "severity": impact,
            "confidence": confidence,
            "description": desc,
            "elements": elements_norm,
        }
        out.append(finding)
    return out

def _stream_pipe(pipe, prefix: str, collect: list[str]) -> None:
    """Reads pipe line by line"""
    for line in iter(pipe.readline, ''):
        ln = line.rstrip('\n')
        if ln:
            print(f"[slither/{prefix}] {ln}", flush=True)
            collect.append(ln)
    pipe.close()


def run(
    input_code: str,
    *,
    timeout_seconds: int = 120,
    extra_args: List[str] | None = None,
    auto_solc: bool = True,
    fallback_solc: str = "0.8.26",
    return_raw: bool = False,
    stream_logs: bool = False
) -> SlitherResult:

    t0 = time.time()
    warnings: List[str] = []
    if not input_code or not input_code.strip():
        return ErrorHandler.create_slither_error(
            ErrorCode.EMPTY_INPUT,
            duration_ms=int((time.time() - t0) * 1000)
        )
      
    chosen_solc: str | None = None

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")  

    try:
        if auto_solc:
            chosen_solc = ensure_solc_auto_by_pragma(input_code, fallback_version=fallback_solc)
        else:
            chosen_solc = fallback_solc
            ensure_solc(chosen_solc)

        solc_bin = _resolve_solc_bin(chosen_solc)

        with tempfile.TemporaryDirectory(prefix="slither_") as td:
            tmp = Path(td)
            src_path = tmp / "Contract.sol"
            src_path.write_text(input_code, encoding="utf-8")

            out_json_path = tmp / "slither.json"

            cmd = [
                "slither",
                str(tmp),
                "--solc",
                solc_bin,
                "--json",
                str(out_json_path)
            ]

            if extra_args:
                cmd.extend(extra_args)

            stdout  = ''
            stderr = ''

            if stream_logs:
                cp = subprocess.Popen(
                    cmd,
                    cwd=str(tmp),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,  
                    env=env,
                )
                stdout_lines: list[str] = []
                stderr_lines: list[str] = []

                t_out = threading.Thread(target=_stream_pipe, args=(cp.stdout, "stdout", stdout_lines))
                t_err = threading.Thread(target=_stream_pipe, args=(cp.stderr, "stderr", stderr_lines))
                t_out.start(); t_err.start()

                try:
                    rc = cp.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    try:
                        cp.kill()
                    except Exception:
                        pass
                    t_out.join(timeout=1)
                    t_err.join(timeout=1)
                    return timeout_error(
                        duration_ms=int((time.time() - t0) * 1000),
                        input_code=input_code,
                        solc_version=chosen_solc
                    )

                t_out.join()
                t_err.join()
                stdout = "\n".join(stdout_lines)
                stderr = "\n".join(stderr_lines)
                cp_returncode = rc
            else:
                cp = subprocess.run(
                    cmd,
                    cwd=str(tmp),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout_seconds,
                    env=env,
                )
                stdout = (cp.stdout or "").strip()
                stderr = (cp.stderr or "").strip()
                cp_returncode = cp.returncode
    
            if stderr:
                for line in stderr.splitlines():
                    ln = line.strip()
                    if ln and not ln.lower().startswith(("slither", "loading")):
                        warnings.append(ln)
        
            raw = {}
            if out_json_path.exists():
                try:
                    raw = json.loads(out_json_path.read_text(encoding="utf-8") or "{}")
                except json.JSONDecodeError as e:
                    warnings.append(f"InvalidJSON: {e}")

            results = (raw or {}).get("results", {})
            detectors = results.get("detectors", []) or (raw or {}).get("detectors", [])

            if detectors:
                findings = _normalize_detectors(detectors)
                metrics: SlitherMetrics = {
                    "count": len(findings),
                    "solc_version": str(chosen_solc) if chosen_solc else None,
                    "pragma": detect_pragma(input_code),
                }
                if cp_returncode != 0:
                    warnings.append(f"NonZeroExit:{cp_returncode}")
                
                result: SlitherResult = {
                    "status": "ok",
                    "module": "slither_wrapper",
                    "warnings": warnings,
                    "errors": [],
                    "findings": findings,
                    "metrics": metrics,
                    "meta": {
                        "duration_ms": int((time.time() - t0) * 1000),
                        "solc_version": str(chosen_solc) if chosen_solc else None,
                        "solc_bin": solc_bin,
                        "exit_code": cp_returncode,
                        "module_version": "0.1.0",
                    },
                }
                return result
            
            if cp_returncode not in (0,):
                return slither_failed_error(
                    exit_code=cp_returncode,
                    stderr=stderr or stdout,
                    duration_ms=int((time.time() - t0) * 1000),
                    input_code=input_code,
                    warnings=warnings,
                    solc_version=str(chosen_solc) if chosen_solc else None,
                    solc_bin=solc_bin
                )
            if not out_json_path.exists():
                return no_json_produced_error(
                    stderr=stderr or stdout,
                    duration_ms=int((time.time() - t0) * 1000),
                    input_code=input_code,
                    warnings=warnings,
                    solc_version=str(chosen_solc) if chosen_solc else None,
                    solc_bin=solc_bin
                )
            findings: List[SlitherFinding] = []
            metrics: SlitherMetrics = {
                "count": 0,
                "solc_version": str(chosen_solc) if chosen_solc else None,
                "pragma": detect_pragma(input_code),
            }
    
            out: SlitherResult = {
                "status": "ok",
                "module": "slither_wrapper",
                "warnings": warnings,
                "errors": [],
                "findings": findings,
                "metrics": metrics,
                "meta": {
                    "duration_ms": int((time.time() - t0) * 1000),
                    "solc_version": str(chosen_solc) if chosen_solc else None,
                    "solc_bin": solc_bin,
                    "exit_code": cp_returncode,
                    "module_version": "0.1.0",
                },
            }
            if return_raw:
                out["raw"] = raw
            return out
    
    except subprocess.TimeoutExpired:
        return timeout_error(
            duration_ms=int((time.time() - t0) * 1000),
            input_code=input_code,
            solc_version=chosen_solc
        )
    except FileNotFoundError as e:
        return slither_not_found_error(
            error=str(e),
            duration_ms=int((time.time() - t0) * 1000),
            input_code=input_code,
            solc_version=chosen_solc
        )
    except json.JSONDecodeError as e:
        return invalid_json_error(
            error=str(e),
            duration_ms=int((time.time() - t0) * 1000),
            input_code=input_code,
            solc_version=chosen_solc
        )
    except Exception as e:
        return ErrorHandler.create_slither_error(
            ErrorCode.RUNTIME_ERROR,
            message=f"{type(e).__name__}: {str(e)}",
            duration_ms=int((time.time() - t0) * 1000),
            input_code=input_code,
            warnings=warnings,
            solc_version=chosen_solc
        )