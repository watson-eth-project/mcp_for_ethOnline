# MCP Solidity Analysis Tools

A Model Context Protocol (MCP) server for Solidity smart contract analysis using Slither and custom parsing tools.

## What This Code Does

This project provides a comprehensive toolkit for analyzing Solidity smart contracts through multiple approaches:

### **Solidity Parser (`parser_solidity.py`)**
- **AST Generation**: Compiles Solidity code to Abstract Syntax Trees using the Solidity compiler
- **Contract Extraction**: Parses contracts, functions, modifiers, events, structs, and enums
- **Version Management**: Automatically detects and installs the correct Solidity compiler version based on pragma directives
- **Position Tracking**: Provides precise source code positions (line numbers, offsets) for each function
- **Caching**: Caches compiled ASTs to improve performance on repeated analysis
- **Progress Logging**: Tracks compilation steps for debugging and monitoring

### **Slither Integration (`slither_wrapper.py`)**
- **Static Analysis**: Runs Slither security analysis on Solidity contracts
- **Vulnerability Detection**: Identifies common security issues, gas optimizations, and best practice violations
- **Compiler Management**: Handles Solidity compiler installation and version switching
- **Streaming Output**: Provides analysis progress and results
- **Error Handling**: Gracefully handles compilation errors and missing dependencies

### **MCP Server (`server.py`)**
- **Protocol Implementation**: Implements the Model Context Protocol for AI tool integration
- **Tool Registration**: Exposes analysis functions as MCP tools
- **Resource Management**: Provides access to cached ASTs and analysis results
- **API Endpoints**: Offers REST-like interface for contract analysis

## Installation

### Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd mcp_modules
```

2. Install dependencies:
```bash
uv sync
```

3. Install Slither (if not already installed):
```bash
pip install slither-analyzer
```

## Usage

### Running Tests

```bash
uv run pytest -q
```

### Starting the MCP Server

```bash
uv run server.py
```

### Development Mode

```bash
uv run mcp dev server.py
```

### Example Usage

#### Parse Solidity Code
```python
from mcp_modules.parser_solidity import run

# Parse a simple contract
code = """
pragma solidity ^0.8.0;

contract Example {
    function hello() public pure returns (string memory) {
        return "Hello, World!";
    }
}
"""

result = run(code, engine="solc", auto_version=True)
print(f"Found {len(result['contracts'])} contracts")
```

#### Analyze with Slither
```python
from mcp_modules.slither_wrapper import analyze_contract

# Run security analysis
analysis = analyze_contract(code, stream_logs=True)
print(f"Found {len(analysis['findings'])} security issues")
```

#### Access Cached AST
```python
# Get specific AST node by position
ast_node = get_ast_node(
    hash="abc12345", 
    src="1:10:0",  # start:length:fileIndex
    limit=5
)
```

## Project Structure

```
mcp_modules/
├── mcp_modules/           # Main package
│   ├── common/            # Common utilities
│   ├── validation/        # Validation schemas
│   ├── parser_solidity.py # Solidity parser
│   └── slither_wrapper.py # Slither integration
├── tests/                 # Test suite
├── server.py             # MCP server entry point
├── pyproject.toml        # Project configuration
└── README.md             # This file
```
## TODO

- парсер пока скидывает все в одну кучу (если код передается для нескольких контрактов сразу)

## API

### MCP Tools Available

The server exposes the following MCP tools:

#### `parse_solidity`
- **Purpose**: Parse Solidity code and extract contract structure
- **Input**: Solidity source code, optional compiler version
- **Output**: Contract definitions, functions, events, with position information
- **Features**: Automatic version detection, AST caching, progress logging

#### `analyze_with_slither`
- **Purpose**: Run comprehensive security analysis using Slither
- **Input**: Solidity source code, analysis options
- **Output**: Security findings, gas optimizations, best practice violations
- **Features**: Real-time streaming, error recovery, compiler management

#### `get_ast_node`
- **Purpose**: Retrieve specific AST nodes from cached compilation results
- **Input**: AST hash, source position or JSON pointer
- **Output**: AST subtree with metadata
- **Features**: Flexible querying, position-based lookup

#### `list_cached_asts`
- **Purpose**: List all cached AST files with metadata
- **Input**: None
- **Output**: Cache statistics, file information
- **Features**: Cache management, debugging support

### Slither Wrapper

The `slither_wrapper.py` module provides:

- `analyze_contract()`: Main function for contract analysis
- `_ensure_solc_for_source()`: Solidity compiler version management
- `_ensure_solc_auto_by_pragma()`: Automatic pragma-based compiler detection

### Solidity Parser

The `parser_solidity.py` module provides:

- `run()`: Main parsing function with comprehensive options
- `_extract_contracts_with_members()`: Contract structure extraction
- `_src_span()`: Source position extraction from AST nodes
- `_line_from_offset()`: Line number calculation from character offset
- Contract structure analysis
- Function and variable extraction
- Import dependency resolution

## Configuration

The project uses `pyproject.toml` for configuration. Key dependencies include:

- `slither-analyzer`: Static analysis tool
- `solc-select`: Solidity compiler management
- `pytest`: Testing framework

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Slither](https://github.com/crytic/slither) for static analysis capabilities
- [Model Context Protocol](https://github.com/modelcontextprotocol) for the MCP framework