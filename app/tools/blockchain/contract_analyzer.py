"""``contract_analyzer`` tool — analyze smart contracts.

Scans Solidity source code for common vulnerabilities, gas optimizations,
and best practices. Can also fetch and analyze verified contracts from
block explorers (Etherscan, BSCScan, etc.).
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from app.tools.base import ToolContext, ToolResult
from app.tools.registry import tool


class ContractAnalysisArgs(BaseModel):
    """Arguments for contract analysis."""

    source_code: str | None = Field(
        None,
        description="Solidity source code to analyze. Provide this OR contract_address.",
    )
    contract_address: str | None = Field(
        None,
        description="Contract address to fetch from block explorer (requires api_key in context).",
    )
    chain: str = Field(
        "ethereum",
        description="Blockchain: ethereum, bsc, polygon, arbitrum, optimism, base.",
    )
    analysis_type: str = Field(
        "full",
        description="Type of analysis: full, security, gas, summary.",
        pattern="^(full|security|gas|summary)$",
    )


# Common vulnerability patterns
_VULN_PATTERNS: list[tuple[str, str, str, str]] = [
    # (pattern, severity, name, description)
    (r"tx\.origin", "HIGH", "tx.origin Usage",
     "Using tx.origin for authentication is vulnerable to phishing attacks."),
    (r"\.call\.value\(", "MEDIUM", "Unchecked Call Return",
     "Low-level call return value should be checked."),
    (r"suicide\(|selfdestruct\(", "HIGH", "selfdestruct",
     "Contract can be destroyed, sending all funds to an arbitrary address."),
    (r"block\.timestamp", "LOW", "Timestamp Dependence",
     "Block timestamp can be manipulated by miners within ~15 seconds."),
    (r"block\.number", "LOW", "Block Number Dependence",
     "Block number is predictable but varies across chains."),
    (r"assembly\s*\{", "INFO", "Inline Assembly",
     "Assembly usage detected — review for safety."),
    (r"delegatecall", "HIGH", "delegatecall",
     "Delegatecall executes in caller's context — ensure proper access control."),
    (r"ecrecover", "MEDIUM", "ecrecover",
     "Signature recovery — ensure proper validation to prevent replay attacks."),
    (r"mapping\s*\(.*\s*=>\s*address\)", "INFO", "Address Mapping",
     "Address mapping detected — verify access control."),
    (r"onlyOwner|Ownable", "INFO", "Access Control",
     "Access control pattern detected — verify implementation."),
    (r"require\(", "INFO", "Input Validation",
     "Require statements found — good practice for input validation."),
    (r"emit\s+\w+", "INFO", "Events",
     "Events emitted — good for off-chain indexing."),
]

# Gas optimization patterns
_GAS_PATTERNS: list[tuple[str, str, str]] = [
    # (pattern, impact, suggestion)
    (r"uint256\s+\w+\s*=\s*0;", "LOW", "Consider using unchecked { ++i } for loop counters."),
    (r"for\s*\(", "MEDIUM", "Cache array length outside loop to save gas."),
    (r"string\s+memory", "LOW", "Use 'bytes' instead of 'string' when possible for gas savings."),
    (r"public\s+", "INFO", "Public functions generate getters — use external if not called internally."),
    (r"require\(\w+\s*!=\s*0\)", "LOW", "Consider using custom errors instead of require strings."),
    (r"modifier\s+\w+", "INFO", "Modifiers increase deployment cost — consider internal functions."),
]


@tool(
    name="contract_analyzer",
    description=(
        "Analyze Solidity smart contracts for security vulnerabilities, gas optimizations, "
        "and best practices. Provide source code directly or a contract address to fetch "
        "from a block explorer. Returns a detailed report with findings."
    ),
    args=ContractAnalysisArgs,
    category="blockchain",
)
async def contract_analyzer(args: ContractAnalysisArgs, ctx: ToolContext | None) -> ToolResult:
    """Analyze a smart contract for vulnerabilities and optimizations."""
    source = args.source_code

    # If address provided, fetch from explorer
    if args.contract_address and not source:
        source = await _fetch_from_explorer(args.contract_address, args.chain, ctx)
        if source.startswith("Error"):
            return ToolResult.fail(source)

    if not source:
        return ToolResult.fail(
            "Provide source_code or contract_address to analyze."
        )

    # Run analysis
    findings: list[dict[str, Any]] = []
    findings.extend(_check_vulnerabilities(source, args.analysis_type))
    findings.extend(_check_gas_optimizations(source, args.analysis_type))

    # Compile report
    report = _compile_report(source, findings, args.analysis_type)
    return ToolResult.ok(report)


async def _fetch_from_explorer(
    address: str, chain: str, ctx: ToolContext | None
) -> str:
    """Fetch verified contract source from a block explorer API."""
    explorers = {
        "ethereum": "https://api.etherscan.io/api",
        "bsc": "https://api.bscscan.com/api",
        "polygon": "https://api.polygonscan.com/api",
        "arbitrum": "https://api.arbiscan.io/api",
        "optimism": "https://api-optimistic.etherscan.io/api",
        "base": "https://api.basescan.org/api",
    }

    base_url = explorers.get(chain.lower())
    if not base_url:
        return f"Error: Unsupported chain '{chain}'. Supported: {', '.join(explorers)}"

    # Try to get API key from settings or context
    api_key = ""
    if ctx and hasattr(ctx, "extra"):
        api_key = ctx.extra.get("explorer_api_key", "")

    import httpx

    url = f"{base_url}?module=contract&action=getsourcecode&address={address}"
    if api_key:
        url += f"&apikey={api_key}"

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=30)
            data = resp.json()

        if data.get("status") != "1" or not data.get("result"):
            return f"Error: Could not fetch contract. {data.get('message', 'Unknown error')}"

        result = data["result"][0]
        if result.get("SourceCode"):
            return result["SourceCode"]
        return "Error: Contract source code is not verified on this explorer."
    except Exception as exc:
        return f"Error fetching contract: {exc}"


def _check_vulnerabilities(
    source: str, analysis_type: str
) -> list[dict[str, Any]]:
    """Check for common vulnerability patterns."""
    if analysis_type == "gas":
        return []

    findings = []
    for pattern, severity, name, description in _VULN_PATTERNS:
        matches = list(re.finditer(pattern, source))
        if matches:
            lines = []
            for match in matches[:5]:  # Limit to 5 occurrences
                line_num = source[:match.start()].count("\n") + 1
                lines.append(line_num)

            findings.append({
                "type": "vulnerability",
                "severity": severity,
                "name": name,
                "description": description,
                "occurrences": len(matches),
                "lines": lines,
            })

    return findings


def _check_gas_optimizations(
    source: str, analysis_type: str
) -> list[dict[str, Any]]:
    """Check for gas optimization opportunities."""
    if analysis_type == "security":
        return []

    findings = []
    for pattern, impact, suggestion in _GAS_PATTERNS:
        matches = list(re.finditer(pattern, source))
        if matches:
            findings.append({
                "type": "gas_optimization",
                "impact": impact,
                "suggestion": suggestion,
                "occurrences": len(matches),
            })

    return findings


def _compile_report(
    source: str, findings: list[dict[str, Any]], analysis_type: str
) -> dict[str, Any]:
    """Compile findings into a structured report."""
    lines = source.split("\n")

    # Categorize findings
    vulns = [f for f in findings if f["type"] == "vulnerability"]
    gas_opts = [f for f in findings if f["type"] == "gas_optimization"]

    # Calculate risk score
    risk_score = 0
    severity_weights = {"HIGH": 10, "MEDIUM": 5, "LOW": 2, "INFO": 0}
    for vuln in vulns:
        risk_score += severity_weights.get(vuln.get("severity", "INFO"), 0)

    risk_level = "LOW"
    if risk_score >= 20:
        risk_level = "CRITICAL"
    elif risk_score >= 10:
        risk_level = "HIGH"
    elif risk_score >= 5:
        risk_level = "MEDIUM"

    report: dict[str, Any] = {
        "summary": {
            "total_lines": len(lines),
            "total_findings": len(findings),
            "vulnerabilities": len(vulns),
            "gas_optimizations": len(gas_opts),
            "risk_score": risk_score,
            "risk_level": risk_level,
        },
        "analysis_type": analysis_type,
    }

    if vulns:
        report["vulnerabilities"] = vulns
    if gas_opts:
        report["gas_optimizations"] = gas_opts

    # Add recommendations
    report["recommendations"] = _generate_recommendations(findings)

    return report


def _generate_recommendations(findings: list[dict[str, Any]]) -> list[str]:
    """Generate actionable recommendations based on findings."""
    recs = []

    high_vulns = [f for f in findings if f.get("severity") == "HIGH"]
    if high_vulns:
        recs.append("⚠️ CRITICAL: Address HIGH severity vulnerabilities before deployment.")

    has_delegatecall = any("delegatecall" in f.get("name", "").lower() for f in findings)
    if has_delegatecall:
        recs.append("Review delegatecall usage — ensure only trusted contracts are called.")

    has_tx_origin = any("tx.origin" in f.get("name", "") for f in findings)
    if has_tx_origin:
        recs.append("Replace tx.origin with msg.sender for authentication.")

    has_selfdestruct = any("selfdestruct" in f.get("name", "").lower() for f in findings)
    if has_selfdestruct:
        recs.append("Consider removing selfdestruct or adding strict access control.")

    gas_finds = [f for f in findings if f["type"] == "gas_optimization"]
    if gas_finds:
        recs.append(f"Found {len(gas_finds)} gas optimization opportunities.")

    if not recs:
        recs.append("✅ No critical issues found. Contract looks good!")

    return recs
