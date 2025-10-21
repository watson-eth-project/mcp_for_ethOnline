from typing import TypedDict, Literal, Any, List, Dict, Optional

# Base status type
Status = Literal["ok", "error"]

# Module identifiers
ModuleType = Literal["parser_solidity", "slither_wrapper", "surya_wrapper"]

# Base result structure that all modules inherit from
class BaseResult(TypedDict, total=False):
    """Base result structure for all MCP modules."""
    status: Status
    module: ModuleType
    warnings: List[str]
    errors: List[str]
    meta: Dict[str, Any]  # durations, versions, options, etc.

# Function parameter structure
class FunctionParameter(TypedDict, total=False):
    """Structure for function parameters (inputs/returns)."""
    name: str
    type: str

# Function position information
class FunctionPosition(TypedDict, total=False):
    """Position information for functions."""
    offset_start: int
    offset_end: int
    line: int

# Function definition structure
class FunctionDefinition(TypedDict, total=False):
    """Structure for function definitions."""
    name: str
    kind: str  # "function", "constructor", "fallback", "receive"
    visibility: str  # "public", "private", "internal", "external"
    state_mutability: str  # "pure", "view", "nonpayable", "payable"
    modifiers: List[str]
    inputs: List[FunctionParameter]
    returns: List[FunctionParameter]
    signature: str
    is_constructor: bool
    is_fallback: bool
    is_receive: bool
    position: FunctionPosition
    src: str
    # Backward compatibility field
    stateMutability: str  # camelCase version for backward compatibility

# Modifier definition structure
class ModifierDefinition(TypedDict, total=False):
    """Structure for modifier definitions."""
    name: str
    parameters: List[FunctionParameter]

# Event definition structure
class EventDefinition(TypedDict, total=False):
    """Structure for event definitions."""
    name: str
    parameters: List[FunctionParameter]

# Struct member structure
class StructMember(TypedDict, total=False):
    """Structure for struct members."""
    name: str
    type: str

# Struct definition structure
class StructDefinition(TypedDict, total=False):
    """Structure for struct definitions."""
    name: str
    members: List[StructMember]

# Enum definition structure
class EnumDefinition(TypedDict, total=False):
    """Structure for enum definitions."""
    name: str
    values: List[str]

# Contract definition structure
class ContractDefinition(TypedDict, total=False):
    """Structure for contract definitions."""
    name: str
    kind: str  # "contract", "interface", "library"
    bases: List[str]  # inherited contracts
    functions: List[FunctionDefinition]
    modifiers: List[ModifierDefinition]
    events: List[EventDefinition]
    structs: List[StructDefinition]
    enums: List[EnumDefinition]

# Source mapping structure
class SourceMapping(TypedDict, total=False):
    """Structure for source code mapping."""
    filename: str
    start: int
    length: int

# Parser Solidity specific payload
class ParserSolidityPayload(TypedDict, total=False):
    """Payload structure for parser_solidity module."""
    ast_uri: str  # URI to cached AST
    contracts: List[ContractDefinition]
    # Legacy fields for backward compatibility
    functions: List[FunctionDefinition]  # flattened functions from all contracts
    events: List[EventDefinition]  # flattened events from all contracts
    modifiers: List[ModifierDefinition]  # flattened modifiers from all contracts
    structs: List[StructDefinition]  # flattened structs from all contracts
    enums: List[EnumDefinition]  # flattened enums from all contracts

# Slither finding element structure
class SlitherElement(TypedDict, total=False):
    """Structure for Slither finding elements."""
    type: str
    name: str
    line: int
    filename: str

# Slither finding structure
class SlitherFinding(TypedDict, total=False):
    """Structure for individual Slither findings."""
    check: str  # detector check name
    severity: str  # impact level
    confidence: str  # confidence level
    description: str
    elements: List[SlitherElement]

# Slither metrics structure
class SlitherMetrics(TypedDict, total=False):
    """Structure for Slither metrics."""
    count: int
    solc_version: str
    pragma: str

# Slither specific payload
class SlitherPayload(TypedDict, total=False):
    """Payload structure for slither_wrapper module."""
    findings: List[SlitherFinding]
    metrics: SlitherMetrics
    raw: Dict[str, Any]  # optional raw Slither JSON output

# Complete result types that combine BaseResult with module-specific payloads
class ParserSolidityResult(BaseResult, ParserSolidityPayload):
    """Complete result type for parser_solidity module."""
    pass

class SlitherResult(BaseResult, SlitherPayload):
    """Complete result type for slither_wrapper module."""
    pass

# Meta field structures for better type safety
class ParserSolidityMeta(TypedDict, total=False):
    """Meta information structure for parser_solidity."""
    duration_ms: int
    engine: str  # "solc", "treesitter"
    solc_version: str
    pragma: str
    source_list: List[str]
    src_mapping: Dict[str, SourceMapping]
    log: List[Dict[str, Any]]
    ast_hash: str
    ast_size_bytes: int
    cache_dir: str
    module_version: str

class SlitherMeta(TypedDict, total=False):
    """Meta information structure for slither_wrapper."""
    duration_ms: int
    solc_version: str
    solc_bin: str
    exit_code: int
    module_version: str

# Union type for all possible results
ModuleResult = ParserSolidityResult | SlitherResult

# Universal document schema for all data types (code/AST/graph/Slither)
class DocumentMeta(TypedDict, total=False):
    """Universal metadata structure for all document types."""
    type: Literal["code", "ast", "edge", "slither"]
    project_id: str
    ast_hash: str
    file: str | None
    contract: str | None
    function: str | None
    signature: str | None
    stateMutability: str | None  # camelCase for backward compatibility
    visibility: str | None
    src: str | None  # "start:length:fileIndex"
    lines: tuple[int, int] | None
    # Graph/detector specific fields
    graph_component: Literal["callgraph", "inheritance"] | None
    edge_type: Literal["internal", "external", "customError"] | None
    severity: Literal["High", "Medium", "Low", "Informational"] | None
    detector: str | None

class Document(TypedDict, total=False):
    """Universal document format for any data type (code/AST/graph/Slither)."""
    id: str  # unique key
    text: str  # brief, meaningful text fragment
    meta: DocumentMeta

# Type aliases for common patterns
ModuleVersion = str
DurationMs = int
