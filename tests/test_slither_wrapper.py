import pytest
from mcp_modules.slither_wrapper import run

code = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

contract Ownable {
    address public owner;
    modifier onlyOwner(){ require(msg.sender==owner,"!owner"); _; }
}

contract Vault is Ownable {
    uint256 public total;
    constructor(){ owner = msg.sender; }
    receive() external payable { total += msg.value; }
    function withdraw(address payable to, uint256 amount) public onlyOwner {
        (bool ok,) = to.call{value: amount}("");
        require(ok, "x");
        total -= amount;
    }
}
"""


extra_args = ["--exclude-informational", "--exclude-low"]
res = run(code, timeout_seconds=60, return_raw=False, stream_logs=False)

print(res["status"], res["metrics"])
print("findings:", len(res["findings"]))
for f in res["findings"][:3]:
    print(f["check"], f["severity"], "->", (f["elements"][:1] if f["elements"] else []))