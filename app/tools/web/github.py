"""``github`` tool — interact with GitHub repositories.

Fetches repository info, file contents, issues, pull requests, and more
via the GitHub API (no authentication required for public repos).
"""

from __future__ import annotations

import base64
import re
from typing import Any

from pydantic import BaseModel, Field

from app.tools.base import ToolContext, ToolResult
from app.tools.registry import tool
from app.utils.logger import get_logger

log = get_logger(__name__)

_GITHUB_API = "https://api.github.com"
_USER_AGENT = "Adit-Agent/1.0"


class GitHubArgs(BaseModel):
    """Arguments for GitHub tool."""

    action: str = Field(
        description="Action: repo_info, file_content, list_files, readme, issues, pulls, search.",
        pattern="^(repo_info|file_content|list_files|readme|issues|pulls|search)$",
    )
    owner: str | None = Field(
        None, description="Repository owner (e.g., 'dotcodotadit')."
    )
    repo: str | None = Field(
        None, description="Repository name (e.g., 'adit-agent-v2')."
    )
    path: str | None = Field(
        None, description="File path for file_content action (e.g., 'README.md')."
    )
    branch: str = Field(
        "main", description="Branch name (default: main)."
    )
    query: str | None = Field(
        None, description="Search query for search action."
    )
    max_results: int = Field(
        10, ge=1, le=100, description="Maximum results to return."
    )


@tool(
    name="github",
    description=(
        "Interact with GitHub repositories: get repo info, read files, list files, "
        "view README, check issues and pull requests, and search repositories. "
        "Works with public repos without authentication."
    ),
    args=GitHubArgs,
    category="web",
    dangerous=False,
)
async def github(args: GitHubArgs, ctx: ToolContext | None) -> ToolResult:
    """Interact with GitHub."""
    try:
        import httpx
    except ImportError as exc:
        return ToolResult.fail("httpx is required for github tool.")

    action = args.action

    # Parse owner/repo from URL if provided
    owner, repo = _parse_github_url(args.owner, args.repo)

    if action == "search":
        return await _search_repos(args, ctx)

    if not owner or not repo:
        return ToolResult.fail("owner and repo are required. Example: owner='dotcodotadit', repo='adit-agent-v2'")

    if action == "repo_info":
        return await _get_repo_info(owner, repo, ctx)
    elif action == "file_content":
        return await _get_file_content(owner, repo, args.path, args.branch, ctx)
    elif action == "list_files":
        return await _list_files(owner, repo, args.path, args.branch, ctx)
    elif action == "readme":
        return await _get_readme(owner, repo, args.branch, ctx)
    elif action == "issues":
        return await _get_issues(owner, repo, args.max_results, ctx)
    elif action == "pulls":
        return await _get_pulls(owner, repo, args.max_results, ctx)
    else:
        return ToolResult.fail(f"Unknown action: {action}")


def _parse_github_url(owner: str | None, repo: str | None) -> tuple[str | None, str | None]:
    """Parse owner/repo from various input formats."""
    if owner and "/" in owner:
        # Handle "owner/repo" format
        parts = owner.split("/")
        return parts[0], parts[1] if len(parts) > 1 else repo
    if owner and "github.com" in owner:
        # Handle full URL
        match = re.search(r"github\.com/([^/]+)/([^/]+)", owner)
        if match:
            return match.group(1), match.group(2).rstrip(".git")
    return owner, repo


async def _make_github_request(url: str, ctx: ToolContext | None) -> dict[str, Any]:
    """Make a request to GitHub API."""
    try:
        import httpx
    except ImportError:
        return {"error": "httpx not available"}

    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "application/vnd.github.v3+json",
    }

    # Add token if available (for higher rate limits)
    import os
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=headers, timeout=30)

            if resp.status_code == 404:
                return {"error": f"Not found: {url}"}
            if resp.status_code == 403:
                return {"error": "Rate limited. Try again later or set GITHUB_TOKEN."}

            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        return {"error": str(exc)}


async def _get_repo_info(owner: str, repo: str, ctx: ToolContext | None) -> ToolResult:
    """Get repository information."""
    data = await _make_github_request(f"{_GITHUB_API}/repos/{owner}/{repo}", ctx)

    if "error" in data:
        return ToolResult.fail(data["error"])

    return ToolResult.ok({
        "name": data.get("full_name"),
        "description": data.get("description"),
        "stars": data.get("stargazers_count"),
        "forks": data.get("forks_count"),
        "watchers": data.get("watchers_count"),
        "language": data.get("language"),
        "topics": data.get("topics", []),
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
        "default_branch": data.get("default_branch"),
        "homepage": data.get("homepage"),
        "license": data.get("license", {}).get("name") if data.get("license") else None,
        "open_issues": data.get("open_issues_count"),
        "url": data.get("html_url"),
    })


async def _get_file_content(
    owner: str, repo: str, path: str | None, branch: str, ctx: ToolContext | None
) -> ToolResult:
    """Get file content from repository."""
    if not path:
        return ToolResult.fail("path is required for file_content action.")

    url = f"{_GITHUB_API}/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    data = await _make_github_request(url, ctx)

    if "error" in data:
        return ToolResult.fail(data["error"])

    # Decode content
    content = ""
    if "content" in data:
        try:
            content = base64.b64decode(data["content"]).decode("utf-8")
        except Exception:
            content = data.get("content", "")

    return ToolResult.ok({
        "name": data.get("name"),
        "path": data.get("path"),
        "size": data.get("size"),
        "type": data.get("type"),
        "content": content[:50000],  # Limit to 50KB
        "encoding": data.get("encoding"),
        "url": data.get("html_url"),
    })


async def _list_files(
    owner: str, repo: str, path: str | None, branch: str, ctx: ToolContext | None
) -> ToolResult:
    """List files in a directory."""
    url = f"{_GITHUB_API}/repos/{owner}/{repo}/contents/{path or ''}?ref={branch}"
    data = await _make_github_request(url, ctx)

    if "error" in data:
        return ToolResult.fail(data["error"])

    if not isinstance(data, list):
        return ToolResult.fail("Path is not a directory.")

    files = []
    for item in data:
        files.append({
            "name": item.get("name"),
            "path": item.get("path"),
            "type": item.get("type"),
            "size": item.get("size"),
        })

    return ToolResult.ok({
        "path": path or "/",
        "files": files,
        "count": len(files),
    })


async def _get_readme(
    owner: str, repo: str, branch: str, ctx: ToolContext | None
) -> ToolResult:
    """Get README content."""
    # Try common README filenames
    for filename in ["README.md", "readme.md", "README.rst", "README"]:
        result = await _get_file_content(owner, repo, filename, branch, ctx)
        if result.success:
            return result

    return ToolResult.fail("No README found.")


async def _get_issues(
    owner: str, repo: str, max_results: int, ctx: ToolContext | None
) -> ToolResult:
    """Get repository issues."""
    url = f"{_GITHUB_API}/repos/{owner}/{repo}/issues?state=open&per_page={max_results}"
    data = await _make_github_request(url, ctx)

    if "error" in data:
        return ToolResult.fail(data["error"])

    issues = []
    for issue in data:
        if "pull_request" not in issue:  # Exclude PRs
            issues.append({
                "number": issue.get("number"),
                "title": issue.get("title"),
                "state": issue.get("state"),
                "created_at": issue.get("created_at"),
                "user": issue.get("user", {}).get("login"),
                "labels": [l.get("name") for l in issue.get("labels", [])],
                "url": issue.get("html_url"),
            })

    return ToolResult.ok({
        "issues": issues,
        "count": len(issues),
    })


async def _get_pulls(
    owner: str, repo: str, max_results: int, ctx: ToolContext | None
) -> ToolResult:
    """Get repository pull requests."""
    url = f"{_GITHUB_API}/repos/{owner}/{repo}/pulls?state=open&per_page={max_results}"
    data = await _make_github_request(url, ctx)

    if "error" in data:
        return ToolResult.fail(data["error"])

    pulls = []
    for pr in data:
        pulls.append({
            "number": pr.get("number"),
            "title": pr.get("title"),
            "state": pr.get("state"),
            "created_at": pr.get("created_at"),
            "user": pr.get("user", {}).get("login"),
            "url": pr.get("html_url"),
        })

    return ToolResult.ok({
        "pull_requests": pulls,
        "count": len(pulls),
    })


async def _search_repos(args: GitHubArgs, ctx: ToolContext | None) -> ToolResult:
    """Search repositories."""
    if not args.query:
        return ToolResult.fail("query is required for search action.")

    url = f"{_GITHUB_API}/search/repositories?q={args.query}&per_page={args.max_results}"
    data = await _make_github_request(url, ctx)

    if "error" in data:
        return ToolResult.fail(data["error"])

    repos = []
    for item in data.get("items", []):
        repos.append({
            "name": item.get("full_name"),
            "description": item.get("description"),
            "stars": item.get("stargazers_count"),
            "language": item.get("language"),
            "url": item.get("html_url"),
        })

    return ToolResult.ok({
        "query": args.query,
        "repos": repos,
        "count": len(repos),
        "total_count": data.get("total_count", 0),
    })
