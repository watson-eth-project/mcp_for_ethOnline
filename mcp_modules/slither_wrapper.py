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
from typing import Any, Dict, List, Literal, TypedDict

from solcx import (
    install_solc,
    install_solc_pragma,
    set_solc_version,
    get_installed_solc_versions,
)

Status = Literal["ok", "error"]


class SlitherResult(TypedDict, total=False):
    status: Status
    module: Literal["slither_wrapper"]
    warnings: List[str]
    errors: List[str]
    meta: Dict[str, Any]
    findings: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    raw: Dict[str, Any]


def _err(
    errors: List[str],
    *,
    t0: float,
    input_code: str | None = None,
    warnings_list: List[str] | None = None,
    solc_ver: str | None = None,
    solc_bin: str | None = None
) -> SlitherResult:
    return {
        "status": "error",
        "module": "slither_wrapper",
        "errors": errors,
        "warnings": (warnings_list or []),
        "findings": [],
        "metrics": {
            "count": 0,
            "solc_version": str(solc_ver) if solc_ver else None,
            "pragma": _detect_pragma(input_code) if input_code else None,
        },
        "meta": {
            "duration_ms": int((time.time() - t0) * 1000),
            "solc_version": str(solc_ver) if solc_ver else None,
            "solc_bin": solc_bin,
        },
    }

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
    
def _detect_pragma(src: str) -> str | None:
    m = re.search(r"pragma\s+solidity\s+([^;]+);", src)
    return m.group(1).strip() if m else None

def _normalize_detectors(detectors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize the Slither output to a compact list of findings."""
    out: List[Dict[str, Any]] = []
    for d in detectors or []:
        check = d.get("check") or d.get("id") or ""
        impact = d.get("impact") or d.get("severity") or ""
        confidence = d.get("confidence") or ""
        desc = (d.get("description") or "").strip()
        elements_norm: List[Dict[str, Any]] = []
        for e in d.get("elements", []) or []:
            el = {
                "type": e.get("type"),
                "name": e.get("name") or e.get("function") or "",
            }
            sm = e.get("source_mapping") or {}
            el["line"] = sm.get("lines", [sm.get("start")])[0] if isinstance(sm.get("lines"), list) else sm.get("start")
            el["filename"] = sm.get("filename_absolute") or sm.get("filename_relative") or sm.get("filename")
            elements_norm.append(el)
        out.append(
            {
                "check": check,
                "severity": impact,
                "confidence": confidence,
                "description": desc,
                "elements": elements_norm,
            }
        )
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
        return _err(["EmptyInput"], t0=t0, warnings_list=warnings)
      
    chosen_solc: str | None = None

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")  

    try:
        if auto_solc:
            chosen_solc = _ensure_solc_auto_by_pragma(input_code, fallback_version=fallback_solc)
        else:
            chosen_solc = fallback_solc
            _ensure_solc(chosen_solc)

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
                    timeout=timeout_seconds,
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
                    return _err(["Timeout"], t0=t0, input_code=input_code, warnings_list=warnings, solc_ver=chosen_solc)

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
                metrics = {
                    "count": len(findings),
                    "solc_version": str(chosen_solc) if chosen_solc else None,
                    "pragma": _detect_pragma(input_code),
                }
                if cp_returncode != 0:
                    warnings.append(f"NonZeroExit:{cp_returncode}")
                return {
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
                    },
                }
            
            if cp_returncode not in (0,):
                return _err(
                    ["SlitherFailed", f"code={cp_returncode}", stderr or stdout],
                    t0=t0,
                    input_code=input_code,
                    warnings_list=warnings,
                    solc_ver=str(chosen_solc) if chosen_solc else None,
                    solc_bin=solc_bin
                )
            if not out_json_path.exists():
                return _err(
                    ["NoJSONProduced", stderr or stdout],
                    t0=t0,
                    input_code=input_code,
                    warnings_list=warnings,
                    solc_ver=str(chosen_solc) if chosen_solc else None,
                    solc_bin=solc_bin
                )
            findings = []
            metrics = {
                "count": 0,
                "solc_version": str(chosen_solc) if chosen_solc else None,
                "pragma": _detect_pragma(input_code),
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
                },
            }
            if return_raw:
                out["raw"] = raw
            return out
    
    except subprocess.TimeoutExpired:
        return _err(["Timeout"], t0=t0, input_code=input_code, warnings_list=warnings, solc_ver=chosen_solc)
    except FileNotFoundError as e:
        return _err(["SlitherNotFound", str(e)], t0=t0, input_code=input_code, warnings_list=warnings, solc_ver=chosen_solc)
    except json.JSONDecodeError as e:
        return _err(["InvalidJSON", str(e)], t0=t0, input_code=input_code, warnings_list=warnings, solc_ver=chosen_solc)
    except Exception as e:
        return _err([type(e).__name__, str(e)], t0=t0, input_code=input_code, warnings_list=warnings, solc_ver=chosen_solc)