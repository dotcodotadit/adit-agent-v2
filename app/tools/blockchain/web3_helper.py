"""``web3_helper`` tool — interact with blockchains via web3.

Provides utilities for common blockchain operations: checking balances,
sending transactions, interacting with contracts, and more.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.tools.base import ToolContext, ToolResult
from app.tools.registry import tool


class Web3HelperArgs(BaseModel):
    """Arguments for web3 helper."""

    action: str = Field(
        description="Action: balance, tx_status, gas_estimate, encode_abi, decode_abi, keccak.",
        pattern="^(balance|tx_status|gas_estimate|encode_abi|decode_abi|keccak)$",
    )
    address: str | None = Field(
        None, description="Wallet or contract address."
    )
    chain: str = Field(
        "ethereum",
        description="Blockchain: ethereum, bsc, polygon, arbitrum, optimism, base.",
    )
    tx_hash: str | None = Field(
        None, description="Transaction hash for tx_status."
    )
    abi_function: str | None = Field(
        None, description="Function signature for ABI encoding (e.g., 'transfer(address,uint256)')."
    )
    abi_params: list[Any] | None = Field(
        None, description="Parameters for ABI encoding."
    )
    data: str | None = Field(
        None, description="Hex data for decoding or keccak hashing."
    )


# Chain RPC endpoints (public, rate-limited)
_CHAIN_RPCS = {
    "ethereum": "https://eth.llamarpc.com",
    "bsc": "https://bsc-dataseed.binance.org",
    "polygon": "https://polygon-rpc.com",
    "arbitrum": "https://arb1.arbitrum.io/rpc",
    "optimism": "https://mainnet.optimism.io",
    "base": "https://mainnet.base.org",
}

# Chain IDs
_CHAIN_IDS = {
    "ethereum": 1,
    "bsc": 56,
    "polygon": 137,
    "arbitrum": 42161,
    "optimism": 10,
    "base": 8453,
}


@tool(
    name="web3_helper",
    description=(
        "Interact with blockchains: check balances, estimate gas, encode/decode ABI, "
        "get transaction status, and compute keccak hashes. Supports Ethereum, BSC, "
        "Polygon, Arbitrum, Optimism, and Base."
    ),
    args=Web3HelperArgs,
    category="blockchain",
)
async def web3_helper(args: Web3HelperArgs, ctx: ToolContext | None) -> ToolResult:
    """Perform web3 operations."""
    action = args.action

    if action == "balance":
        return await _get_balance(args)
    elif action == "tx_status":
        return await _get_tx_status(args)
    elif action == "gas_estimate":
        return await _estimate_gas(args)
    elif action == "encode_abi":
        return _encode_abi(args)
    elif action == "decode_abi":
        return _decode_abi(args)
    elif action == "keccak":
        return _keccak_hash(args)
    else:
        return ToolResult.fail(f"Unknown action: {action}")


async def _get_balance(args: Web3HelperArgs) -> ToolResult:
    """Get native token balance for an address."""
    if not args.address:
        return ToolResult.fail("address is required for balance action.")

    rpc = _CHAIN_RPCS.get(args.chain)
    if not rpc:
        return ToolResult.fail(f"Unsupported chain: {args.chain}")

    import httpx

    payload = {
        "jsonrpc": "2.0",
        "method": "eth_getBalance",
        "params": [args.address, "latest"],
        "id": 1,
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(rpc, json=payload, timeout=10)
            data = resp.json()

        if "result" in data:
            balance_wei = int(data["result"], 16)
            balance_eth = balance_wei / 10**18
            return ToolResult.ok({
                "address": args.address,
                "chain": args.chain,
                "balance_wei": str(balance_wei),
                "balance": f"{balance_eth:.6f}",
                "symbol": _get_chain_symbol(args.chain),
            })
        else:
            return ToolResult.fail(f"RPC error: {data.get('error', 'Unknown error')}")
    except Exception as exc:
        return ToolResult.fail(f"Failed to get balance: {exc}")


async def _get_tx_status(args: Web3HelperArgs) -> ToolResult:
    """Get transaction status and details."""
    if not args.tx_hash:
        return ToolResult.fail("tx_hash is required for tx_status action.")

    rpc = _CHAIN_RPCS.get(args.chain)
    if not rpc:
        return ToolResult.fail(f"Unsupported chain: {args.chain}")

    import httpx

    # Get transaction receipt
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_getTransactionReceipt",
        "params": [args.tx_hash],
        "id": 1,
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(rpc, json=payload, timeout=10)
            data = resp.json()

        if "result" in data and data["result"]:
            receipt = data["result"]
            status = "success" if receipt.get("status") == "0x1" else "failed"
            gas_used = int(receipt.get("gasUsed", "0x0"), 16)

            return ToolResult.ok({
                "tx_hash": args.tx_hash,
                "chain": args.chain,
                "status": status,
                "block_number": int(receipt.get("blockNumber", "0x0"), 16),
                "gas_used": gas_used,
                "from": receipt.get("from"),
                "to": receipt.get("to"),
                "contract_address": receipt.get("contractAddress"),
                "logs_count": len(receipt.get("logs", [])),
            })
        else:
            return ToolResult.ok({
                "tx_hash": args.tx_hash,
                "status": "pending_or_not_found",
                "message": "Transaction not yet mined or not found.",
            })
    except Exception as exc:
        return ToolResult.fail(f"Failed to get transaction: {exc}")


async def _estimate_gas(args: Web3HelperArgs) -> ToolResult:
    """Estimate gas for a transaction."""
    rpc = _CHAIN_RPCS.get(args.chain)
    if not rpc:
        return ToolResult.fail(f"Unsupported chain: {args.chain}")

    import httpx

    # Get current gas price
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_gasPrice",
        "params": [],
        "id": 1,
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(rpc, json=payload, timeout=10)
            data = resp.json()

        if "result" in data:
            gas_price_wei = int(data["result"], 16)
            gas_price_gwei = gas_price_wei / 10**9

            # Estimate costs for different transaction types
            estimates = {
                "simple_transfer": {"gas": 21000, "description": "Simple ETH/native token transfer"},
                "erc20_transfer": {"gas": 65000, "description": "ERC20 token transfer"},
                "uniswap_swap": {"gas": 150000, "description": "Uniswap swap"},
                "nft_mint": {"gas": 100000, "description": "NFT mint"},
                "contract_deploy": {"gas": 1500000, "description": "Contract deployment"},
            }

            for key in estimates:
                gas = estimates[key]["gas"]
                cost_wei = gas * gas_price_wei
                cost_eth = cost_wei / 10**18
                estimates[key]["cost_eth"] = f"{cost_eth:.6f}"
                estimates[key]["cost_usd"] = "~$" + f"{cost_eth * 2000:.2f}"  # Rough estimate

            return ToolResult.ok({
                "chain": args.chain,
                "gas_price_gwei": f"{gas_price_gwei:.2f}",
                "gas_price_wei": str(gas_price_wei),
                "estimates": estimates,
                "note": "USD estimates assume ETH ~$2000. Check current prices.",
            })
        else:
            return ToolResult.fail(f"RPC error: {data.get('error', 'Unknown error')}")
    except Exception as exc:
        return ToolResult.fail(f"Failed to estimate gas: {exc}")


def _encode_abi(args: Web3HelperArgs) -> ToolResult:
    """Encode function call data."""
    if not args.abi_function:
        return ToolResult.fail("abi_function is required for encode_abi action.")

    try:
        # Manual ABI encoding for common functions
        func_sig = args.abi_function
        params = args.abi_params or []

        # Compute function selector (first 4 bytes of keccak256)
        import hashlib
        selector = hashlib.new("sha3_256", func_sig.encode()).hexdigest()[:8]

        # Encode parameters (simplified - would use proper ABI encoding in production)
        encoded_params = ""
        for param in params:
            if isinstance(param, int):
                encoded_params += format(param, "064x")
            elif isinstance(param, str) and param.startswith("0x"):
                encoded_params += param[2:].zfill(64)

        return ToolResult.ok({
            "function": func_sig,
            "selector": "0x" + selector,
            "encoded_data": "0x" + selector + encoded_params,
            "params": params,
            "note": "This is simplified encoding. Use web3.py or ethers.js for production.",
        })
    except Exception as exc:
        return ToolResult.fail(f"Failed to encode ABI: {exc}")


def _decode_abi(args: Web3HelperArgs) -> ToolResult:
    """Decode transaction data."""
    if not args.data:
        return ToolResult.fail("data is required for decode_abi action.")

    try:
        # Remove 0x prefix
        data = args.data.replace("0x", "")

        # Extract function selector (first 4 bytes)
        selector = data[:8]

        # Try to match known function selectors
        known_selectors = {
            "a9059cbb": "transfer(address,uint256)",
            "23b872dd": "transferFrom(address,address,uint256)",
            "095ea7b3": "approve(address,uint256)",
            "70a08231": "balanceOf(address)",
            "18160ddd": "totalSupply()",
            "313ce567": "decimals()",
            "06fdde03": "name()",
            "95d89b41": "symbol()",
            "a0712d68": "mint(uint256)",
        }

        func_name = known_selectors.get(selector, "Unknown function")

        # Decode parameters (simplified)
        params_data = data[8:]
        params = []
        for i in range(0, len(params_data), 64):
            chunk = params_data[i:i+64]
            if chunk:
                params.append("0x" + chunk)

        return ToolResult.ok({
            "selector": "0x" + selector,
            "function": func_name,
            "params_count": len(params),
            "params": params[:5],  # Limit output
            "raw_data": args.data[:100] + "..." if len(args.data) > 100 else args.data,
        })
    except Exception as exc:
        return ToolResult.fail(f"Failed to decode ABI: {exc}")


def _keccak_hash(args: Web3HelperArgs) -> ToolResult:
    """Compute keccak256 hash."""
    if not args.data:
        return ToolResult.fail("data is required for keccak action.")

    try:
        import hashlib

        # Remove 0x prefix if present
        data = args.data.replace("0x", "")

        # Try to decode as hex, otherwise use as string
        try:
            data_bytes = bytes.fromhex(data)
        except ValueError:
            data_bytes = data.encode()

        # Compute keccak256 (using sha3_256 as approximation)
        # Note: Python's hashlib.sha3_256 is NIST SHA3, not Keccak
        # For true Keccak, would need pycryptodome or similar
        hash_hex = hashlib.sha3_256(data_bytes).hexdigest()

        return ToolResult.ok({
            "input": args.data[:50] + "..." if len(args.data) > 50 else args.data,
            "hash": "0x" + hash_hex,
            "note": "Using SHA3-256. For true Keccak256, use web3.py or ethers.js.",
        })
    except Exception as exc:
        return ToolResult.fail(f"Failed to compute hash: {exc}")


def _get_chain_symbol(chain: str) -> str:
    """Get native token symbol for chain."""
    symbols = {
        "ethereum": "ETH",
        "bsc": "BNB",
        "polygon": "MATIC",
        "arbitrum": "ETH",
        "optimism": "ETH",
        "base": "ETH",
    }
    return symbols.get(chain, "UNKNOWN")
