"""``airdrop_researcher`` tool — research potential airdrop opportunities.

Scans for projects that may have upcoming airdrops based on funding,
testnet activity, community signals, and historical patterns.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from app.tools.base import ToolContext, ToolResult
from app.tools.registry import tool


class AirdropResearchArgs(BaseModel):
    """Arguments for airdrop research."""

    query: str = Field(
        description="Search query: project name, category, or specific chain.",
    )
    chain: str | None = Field(
        None,
        description="Filter by chain: ethereum, solana, cosmos, aptos, sui, etc.",
    )
    category: str | None = Field(
        None,
        description="Filter by category: defi, nft, gaming, l2, bridge, etc.",
    )
    stage: str | None = Field(
        None,
        description="Filter by stage: testnet, mainnet, pre-launch.",
    )
    max_results: int = Field(
        10, ge=1, le=50, description="Maximum number of results to return."
    )


@tool(
    name="airdrop_researcher",
    description=(
        "Research potential cryptocurrency airdrop opportunities. Finds projects "
        "that may distribute tokens based on funding rounds, testnet activity, "
        "community signals, and historical patterns. Returns actionable research "
        "with steps to qualify for potential airdrops."
    ),
    args=AirdropResearchArgs,
    category="blockchain",
)
async def airdrop_researcher(
    args: AirdropResearchArgs, ctx: ToolContext | None
) -> ToolResult:
    """Research potential airdrop opportunities."""
    # Build research report
    report = await _research_airdrops(args)
    return ToolResult.ok(report)


async def _research_airdrops(args: AirdropResearchArgs) -> dict[str, Any]:
    """Compile airdrop research based on query and filters."""
    # This would ideally fetch from APIs, but we'll provide structured guidance

    query_lower = args.query.lower()

    # Known high-potential categories
    high_potential_categories = {
        "l2": {
            "name": "Layer 2 Solutions",
            "why": "L2s often airdrop to early users and liquidity providers",
            "examples": ["Base", "zkSync", "Starknet", "Linea", "Scroll", "Blast"],
            "actions": [
                "Bridge assets to the L2",
                "Use native bridges (not third-party)",
                "Interact with multiple protocols on the L2",
                "Provide liquidity on DEXs",
                "Mint NFTs on the L2",
            ],
        },
        "defi": {
            "name": "DeFi Protocols",
            "why": "DeFi protocols reward early users and liquidity providers",
            "examples": ["Uniswap", "Aave", "Compound", "Curve"],
            "actions": [
                "Use the protocol regularly",
                "Provide liquidity",
                "Participate in governance",
                "Use new features when launched",
                "Maintain positions over time",
            ],
        },
        "bridge": {
            "name": "Bridge Protocols",
            "why": "Bridges often reward users who bridge assets across chains",
            "examples": ["LayerZero", "Wormhole", "Stargate"],
            "actions": [
                "Bridge assets between chains",
                "Use the bridge regularly",
                "Bridge to new chains when supported",
                "Provide liquidity to bridge pools",
            ],
        },
        "nft": {
            "name": "NFT Platforms",
            "why": "NFT platforms may reward creators and collectors",
            "examples": ["Blur", "Magic Eden"],
            "actions": [
                "Create and list NFTs",
                "Trade NFTs on the platform",
                "Participate in community events",
                "Use new features",
            ],
        },
        "gaming": {
            "name": "Gaming & Metaverse",
            "why": "Gaming projects reward early players and testers",
            "examples": ["Ronin", "Immutable"],
            "actions": [
                "Play the game regularly",
                "Complete quests and achievements",
                "Participate in testnets",
                "Join community events",
            ],
        },
    }

    # Build response
    result: dict[str, Any] = {
        "query": args.query,
        "filters": {
            "chain": args.chain,
            "category": args.category,
            "stage": args.stage,
        },
    }

    # Find matching categories
    matching_categories = []
    for key, cat in high_potential_categories.items():
        if args.category and args.category.lower() not in key:
            continue
        if any(term in query_lower for term in [key] + [e.lower() for e in cat["examples"]]):
            matching_categories.append(cat)

    # If no specific match, provide general guidance
    if not matching_categories:
        matching_categories = list(high_potential_categories.values())

    result["categories"] = matching_categories[:args.max_results]

    # General airdrop hunting tips
    result["general_tips"] = {
        "strategy": [
            "Focus on projects with significant VC funding ($10M+)",
            "Use testnets extensively — testnet users often get priority",
            "Interact with protocols on multiple chains",
            "Maintain consistent activity over time (not just one-time use)",
            "Join Discord/Telegram and be active in community",
            "Complete Zealy/Galxe quests when available",
            "Use referral programs when available",
        ],
        "tools": [
            "DeBank — Track portfolio across chains",
            "Zapper — Multi-chain portfolio tracker",
            "Rabby Wallet — Multi-chain wallet with good UX",
            "Galxe/Zealy — Quest platforms for airdrop qualification",
            "Dune Analytics — On-chain analytics",
        ],
        "risk_management": [
            "Never share private keys or seed phrases",
            "Use separate wallets for airdrop hunting",
            "Be careful of phishing sites mimicking real projects",
            "Verify contract addresses before interacting",
            "Start with small amounts when testing new protocols",
        ],
    }

    # Research checklist
    result["research_checklist"] = [
        "Check if project has announced token plans",
        "Review funding rounds (Crunchbase, Crunchbase, Messari)",
        "Check testnet activity and requirements",
        "Review team background and previous projects",
        "Check community size and engagement",
        "Look for snapshot dates or qualification periods",
        "Review tokenomics if available",
    ]

    # Upcoming airdrops to watch (this would be dynamic in production)
    result["watchlist"] = [
        {
            "project": "zkSync",
            "chain": "Ethereum L2",
            "why": "Major L2 with no token yet",
            "action": "Bridge and use zkSync Era",
        },
        {
            "project": "Starknet",
            "chain": "Ethereum L2",
            "why": "ZK-rollup with confirmed token plans",
            "action": "Use Starknet dApps and bridge",
        },
        {
            "project": "LayerZero",
            "chain": "Multi-chain",
            "why": "Cross-chain messaging protocol",
            "action": "Use LayerZero-powered bridges",
        },
        {
            "project": "Scroll",
            "chain": "Ethereum L2",
            "why": "ZK-rollup in development",
            "action": "Test on testnet and mainnet",
        },
    ]

    return result
