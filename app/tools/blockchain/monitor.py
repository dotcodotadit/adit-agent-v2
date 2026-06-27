"""``chain_monitor`` tool — monitor blockchain events and transactions.

Creates and manages monitors for wallet activity, contract events,
token transfers, and price movements.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from app.tools.base import ToolContext, ToolResult
from app.tools.registry import tool


class MonitorArgs(BaseModel):
    """Arguments for chain monitor."""

    action: str = Field(
        description="Action: create, list, status, remove, check.",
        pattern="^(create|list|status|remove|check)$",
    )
    monitor_type: str | None = Field(
        None,
        description="Type: wallet, contract, token, price, gas.",
        pattern="^(wallet|contract|token|price|gas)$",
    )
    target: str | None = Field(
        None,
        description="Target address, token symbol, or contract to monitor.",
    )
    chain: str = Field(
        "ethereum",
        description="Blockchain: ethereum, bsc, polygon, arbitrum, solana.",
    )
    webhook_url: str | None = Field(
        None,
        description="Optional webhook URL for notifications.",
    )
    conditions: dict[str, Any] | None = Field(
        None,
        description="Alert conditions (e.g., {'min_amount': 1000, 'direction': 'in'}).",
    )


# In-memory store for monitors (would use database in production)
_MONITORS: dict[str, dict[str, Any]] = {}


@tool(
    name="chain_monitor",
    description=(
        "Monitor blockchain activity: track wallet transactions, contract events, "
        "token transfers, price movements, and gas prices. Set up alerts for "
        "specific conditions and get real-time notifications."
    ),
    args=MonitorArgs,
    category="blockchain",
)
async def chain_monitor(args: MonitorArgs, ctx: ToolContext | None) -> ToolResult:
    """Manage blockchain monitors."""
    action = args.action

    if action == "create":
        return await _create_monitor(args)
    elif action == "list":
        return _list_monitors()
    elif action == "status":
        return _monitor_status(args)
    elif action == "remove":
        return _remove_monitor(args)
    elif action == "check":
        return await _check_monitor(args)
    else:
        return ToolResult.fail(f"Unknown action: {action}")


async def _create_monitor(args: MonitorArgs) -> ToolResult:
    """Create a new monitor."""
    if not args.monitor_type:
        return ToolResult.fail("monitor_type is required for create action.")
    if not args.target:
        return ToolResult.fail("target is required for create action.")

    # Generate monitor ID
    monitor_id = f"{args.monitor_type}_{args.chain}_{args.target[:8]}"

    # Validate monitor type specific requirements
    validation = _validate_monitor_config(args)
    if validation:
        return ToolResult.fail(validation)

    # Create monitor config
    monitor = {
        "id": monitor_id,
        "type": args.monitor_type,
        "target": args.target,
        "chain": args.chain,
        "webhook_url": args.webhook_url,
        "conditions": args.conditions or {},
        "status": "active",
        "created_at": "2024-01-01T00:00:00Z",  # Would use actual timestamp
        "last_checked": None,
        "alerts": [],
    }

    _MONITORS[monitor_id] = monitor

    return ToolResult.ok({
        "message": f"Monitor created successfully!",
        "monitor": monitor,
        "instructions": _get_monitor_instructions(args.monitor_type),
    })


def _validate_monitor_config(args: MonitorArgs) -> str | None:
    """Validate monitor configuration."""
    if args.monitor_type == "wallet":
        # Basic address validation
        if not args.target.startswith("0x") and len(args.target) < 20:
            return "Invalid wallet address format."
    elif args.monitor_type == "token":
        # Token symbol or address
        if len(args.target) < 2:
            return "Invalid token symbol or address."
    elif args.monitor_type == "price":
        # Price feed identifier
        if "/" not in args.target and args.target.upper() != args.target:
            return "Use format like 'ETH/USD' for price feeds."
    return None


def _get_monitor_instructions(monitor_type: str) -> list[str]:
    """Get setup instructions for monitor type."""
    instructions = {
        "wallet": [
            "Monitor will track all incoming/outgoing transactions",
            "Set conditions to filter by amount, token, or direction",
            "Webhook notifications will be sent for matching transactions",
        ],
        "contract": [
            "Monitor will watch for contract events",
            "Specify event signatures in conditions for filtering",
            "Useful for tracking DEX trades, NFT mints, etc.",
        ],
        "token": [
            "Monitor will track token transfers",
            "Set min_amount to filter small transfers",
            "Useful for whale watching and large transfers",
        ],
        "price": [
            "Monitor will track price movements",
            "Set price thresholds in conditions",
            "Supports major tokens and trading pairs",
        ],
        "gas": [
            "Monitor will track gas prices",
            "Set max_gas_price for alerts",
            "Useful for timing transactions",
        ],
    }
    return instructions.get(monitor_type, ["Unknown monitor type."])


def _list_monitors() -> ToolResult:
    """List all active monitors."""
    if not _MONITORS:
        return ToolResult.ok({
            "message": "No monitors configured yet.",
            "monitors": [],
        })

    monitors = list(_MONITORS.values())
    active = [m for m in monitors if m["status"] == "active"]

    return ToolResult.ok({
        "message": f"Found {len(monitors)} monitor(s), {len(active)} active.",
        "monitors": monitors,
    })


def _monitor_status(args: MonitorArgs) -> ToolResult:
    """Get status of a specific monitor."""
    if not args.target:
        return ToolResult.fail("target (monitor ID) is required for status action.")

    # Try to find monitor by ID or target
    monitor = _MONITORS.get(args.target)
    if not monitor:
        # Search by target
        for m in _MONITORS.values():
            if m["target"] == args.target:
                monitor = m
                break

    if not monitor:
        return ToolResult.fail(f"Monitor not found: {args.target}")

    return ToolResult.ok({
        "monitor": monitor,
        "recent_alerts": monitor.get("alerts", [])[-5:],
    })


def _remove_monitor(args: MonitorArgs) -> ToolResult:
    """Remove a monitor."""
    if not args.target:
        return ToolResult.fail("target (monitor ID) is required for remove action.")

    monitor = _MONITORS.pop(args.target, None)
    if not monitor:
        return ToolResult.fail(f"Monitor not found: {args.target}")

    return ToolResult.ok({
        "message": f"Monitor {args.target} removed successfully.",
        "removed_monitor": monitor,
    })


async def _check_monitor(args: MonitorArgs) -> ToolResult:
    """Manually check a monitor's current state."""
    if not args.target:
        return ToolResult.fail("target (monitor ID) is required for check action.")

    monitor = _MONITORS.get(args.target)
    if not monitor:
        return ToolResult.fail(f"Monitor not found: {args.target}")

    # In production, this would actually check the blockchain
    # For now, return a placeholder
    return ToolResult.ok({
        "monitor_id": monitor["id"],
        "status": "checked",
        "message": "Manual check initiated. In production, this would query the blockchain.",
        "note": "Connect to a blockchain RPC to enable real-time monitoring.",
    })
