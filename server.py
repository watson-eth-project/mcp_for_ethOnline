from mcp_modules import parser_solidity, slither_wrapper   
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Demo")

@mcp.tool()
def parse_solidity(input_code: str, engine: str = "solc") -> dict:
    """
    Parse Solidity source code and extract functions, modifiers, visibility, etc.
    """
    return parser_solidity.run(input_code, engine=engine)

@mcp.tool()
def slither_scan(input_code: str, timeout_seconds: int = 120) -> dict:
    """Run Slither static analysis and return normalized findings."""
    return slither_wrapper.run(input_code, timeout_seconds=timeout_seconds)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")  