"""
Document creation utilities for converting various data types to universal Document format.
"""

from typing import Dict, Any, List, Optional
from .types import Document, DocumentMeta, FunctionDefinition, ContractDefinition, SlitherFinding

def create_code_document(
    id: str,
    text: str,
    project_id: str,
    ast_hash: str,
    file: Optional[str] = None,
    contract: Optional[str] = None,
    function: Optional[str] = None,
    signature: Optional[str] = None,
    stateMutability: Optional[str] = None,
    visibility: Optional[str] = None,
    src: Optional[str] = None,
    lines: Optional[tuple[int, int]] = None
) -> Document:
    """Create a document for code fragments."""
    meta: DocumentMeta = {
        "type": "code",
        "project_id": project_id,
        "ast_hash": ast_hash,
        "file": file,
        "contract": contract,
        "function": function,
        "signature": signature,
        "stateMutability": stateMutability,
        "visibility": visibility,
        "src": src,
        "lines": lines,
    }
    
    return {
        "id": id,
        "text": text,
        "meta": meta
    }

def create_ast_document(
    id: str,
    text: str,
    project_id: str,
    ast_hash: str,
    file: Optional[str] = None,
    contract: Optional[str] = None,
    function: Optional[str] = None,
    src: Optional[str] = None,
    lines: Optional[tuple[int, int]] = None
) -> Document:
    """Create a document for AST nodes."""
    meta: DocumentMeta = {
        "type": "ast",
        "project_id": project_id,
        "ast_hash": ast_hash,
        "file": file,
        "contract": contract,
        "function": function,
        "src": src,
        "lines": lines,
    }
    
    return {
        "id": id,
        "text": text,
        "meta": meta
    }

def create_edge_document(
    id: str,
    text: str,
    project_id: str,
    ast_hash: str,
    graph_component: Optional[str] = None,
    edge_type: Optional[str] = None,
    file: Optional[str] = None,
    contract: Optional[str] = None,
    function: Optional[str] = None,
    src: Optional[str] = None,
    lines: Optional[tuple[int, int]] = None
) -> Document:
    """Create a document for graph edges."""
    meta: DocumentMeta = {
        "type": "edge",
        "project_id": project_id,
        "ast_hash": ast_hash,
        "file": file,
        "contract": contract,
        "function": function,
        "src": src,
        "lines": lines,
        "graph_component": graph_component,
        "edge_type": edge_type,
    }
    
    return {
        "id": id,
        "text": text,
        "meta": meta
    }

def create_slither_document(
    id: str,
    text: str,
    project_id: str,
    ast_hash: str,
    severity: Optional[str] = None,
    detector: Optional[str] = None,
    file: Optional[str] = None,
    contract: Optional[str] = None,
    function: Optional[str] = None,
    src: Optional[str] = None,
    lines: Optional[tuple[int, int]] = None
) -> Document:
    """Create a document for Slither findings."""
    meta: DocumentMeta = {
        "type": "slither",
        "project_id": project_id,
        "ast_hash": ast_hash,
        "file": file,
        "contract": contract,
        "function": function,
        "src": src,
        "lines": lines,
        "severity": severity,
        "detector": detector,
    }
    
    return {
        "id": id,
        "text": text,
        "meta": meta
    }

def function_to_documents(
    func: FunctionDefinition,
    project_id: str,
    ast_hash: str,
    file: Optional[str] = None,
    contract: Optional[str] = None
) -> List[Document]:
    """Convert a function definition to multiple documents."""
    documents: List[Document] = []
    
    func_id = f"{contract}::{func['name']}" if contract else func['name']
    func_text = func.get('signature', func['name'])
    
    documents.append(create_code_document(
        id=func_id,
        text=func_text,
        project_id=project_id,
        ast_hash=ast_hash,
        file=file,
        contract=contract,
        function=func['name'],
        signature=func.get('signature'),
        stateMutability=func.get('stateMutability') or func.get('state_mutability'),
        visibility=func.get('visibility'),
        src=func.get('src'),
        lines=func.get('position', {}).get('line')
    ))
    
    for param in func.get('inputs', []):
        param_id = f"{func_id}::param::{param['name']}"
        param_text = f"{param['type']} {param['name']}"
        
        documents.append(create_code_document(
            id=param_id,
            text=param_text,
            project_id=project_id,
            ast_hash=ast_hash,
            file=file,
            contract=contract,
            function=func['name'],
            signature=func.get('signature')
        ))
    
    for param in func.get('returns', []):
        param_id = f"{func_id}::return::{param['name']}"
        param_text = f"{param['type']} {param['name']}"
        
        documents.append(create_code_document(
            id=param_id,
            text=param_text,
            project_id=project_id,
            ast_hash=ast_hash,
            file=file,
            contract=contract,
            function=func['name'],
            signature=func.get('signature')
        ))
    
    return documents

def contract_to_documents(
    contract: ContractDefinition,
    project_id: str,
    ast_hash: str,
    file: Optional[str] = None
) -> List[Document]:
    """Convert a contract definition to multiple documents."""
    documents: List[Document] = []
    
    contract_id = contract['name']
    contract_text = f"{contract.get('kind', 'contract')} {contract['name']}"
    
    documents.append(create_code_document(
        id=contract_id,
        text=contract_text,
        project_id=project_id,
        ast_hash=ast_hash,
        file=file,
        contract=contract['name']
    ))
    
    for func in contract.get('functions', []):
        documents.extend(function_to_documents(
            func, project_id, ast_hash, file, contract['name']
        ))
    
    return documents

def slither_finding_to_document(
    finding: SlitherFinding,
    project_id: str,
    ast_hash: str,
    file: Optional[str] = None,
    contract: Optional[str] = None
) -> Document:
    """Convert a Slither finding to a document."""
    finding_id = f"slither::{finding['check']}::{contract or 'unknown'}"
    finding_text = f"{finding['severity']}: {finding['description']}"
    
    lines = None
    if finding.get('elements'):
        element = finding['elements'][0]
        if element.get('line'):
            lines = (element['line'], element['line'])
    
    return create_slither_document(
        id=finding_id,
        text=finding_text,
        project_id=project_id,
        ast_hash=ast_hash,
        severity=finding.get('severity'),
        detector=finding.get('check'),
        file=file,
        contract=contract,
        function=finding.get('elements', [{}])[0].get('name') if finding.get('elements') else None,
        lines=lines
    )
