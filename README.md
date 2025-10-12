# MCP Solidity Analysis Tools

A Model Context Protocol (MCP) server for Solidity smart contract analysis using Slither and custom parsing tools.

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

### Slither Wrapper

The `slither_wrapper.py` module provides:

- `analyze_contract()`: Main function for contract analysis
- `_ensure_solc_for_source()`: Solidity compiler version management
- `_ensure_solc_auto_by_pragma()`: Automatic pragma-based compiler detection

### Solidity Parser

The `parser_solidity.py` module provides:

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