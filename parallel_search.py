"""Anonymous Parallel Search MCP research using the official HTTP MCP client."""

import asyncio
import json
from urllib.parse import urlsplit
from uuid import uuid4

import anyio
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = (
    "Langflow-Research-Web-Agent/1.0 "
    "(+https://github.com/abrarhussain01/Langflow-Research-Web-Agent)"
)


class ParallelSearchError(RuntimeError):
    """The selected provider failed; callers must not silently switch providers."""


def _decode(result, tool):
    text = "\n".join(block.text for block in result.content if block.type == "text")
    if result.isError:
        raise ParallelSearchError(f"Parallel {tool} failed: {text}")
    data = result.structuredContent
    if data is None:
        try:
            data = json.loads(text)
        except (ValueError, TypeError) as exc:
            raise ParallelSearchError(f"Parallel {tool} returned invalid JSON") from exc
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ParallelSearchError(f"Parallel {tool} returned an invalid results object")
    return data


def _source_urls(results):
    urls = []
    for result in results:
        url = result.get("url") if isinstance(result, dict) else None
        if not isinstance(url, str):
            continue
        try:
            parsed = urlsplit(url)
            valid = parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password
        except ValueError:
            valid = False
        if valid and url not in urls:
            urls.append(url)
        if len(urls) == 3:
            break
    return urls


async def research(question, *, session_id=None, timeout=60):
    """Search and fetch up to three sources within one cancellable time budget.

    A session ID belongs to a research question and is shared by search and fetch.
    No environment credentials are read and no application retries are made.
    """
    if not question or not question.strip():
        raise ValueError("A research question is required")
    session_id = session_id or str(uuid4())
    arguments = {
        "objective": question,
        "search_queries": [question],
        "session_id": session_id,
        "model_name": "gpt-4o",
    }
    try:
        with anyio.fail_after(timeout):
            async with httpx.AsyncClient(
                headers={"User-Agent": USER_AGENT},
                timeout=httpx.Timeout(timeout),
                follow_redirects=False,
                trust_env=False,
            ) as client:
                async with streamable_http_client(MCP_URL, http_client=client) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools = await session.list_tools()
                        if not {"web_search", "web_fetch"}.issubset({tool.name for tool in tools.tools}):
                            raise ParallelSearchError("Parallel MCP is missing web_search or web_fetch")
                        search = _decode(await session.call_tool("web_search", arguments), "web_search")
                        urls = _source_urls(search["results"])
                        pages = {"results": []}
                        if urls:
                            pages = _decode(await session.call_tool("web_fetch", {
                                "urls": urls,
                                "objective": question[:200],
                                "search_queries": [question],
                                "session_id": session_id,
                                "model_name": "gpt-4o",
                            }), "web_fetch")
                        return {"search": search, "pages": pages}
    except TimeoutError as exc:
        raise ParallelSearchError(f"Parallel research timed out after {timeout} seconds") from exc
    except ExceptionGroup as exc:
        # MCP's task groups can wrap tool/HTTP errors. Keep the actionable cause.
        def causes(group):
            for error in group.exceptions:
                if isinstance(error, ExceptionGroup):
                    yield from causes(error)
                else:
                    yield str(error)
        raise ParallelSearchError("; ".join(causes(exc))) from exc
    except httpx.HTTPError as exc:
        raise ParallelSearchError(f"Parallel MCP request failed: {exc}") from exc


def parallel_research(question):
    """Synchronous adapter for the existing LangGraph research node."""
    return asyncio.run(research(question))
