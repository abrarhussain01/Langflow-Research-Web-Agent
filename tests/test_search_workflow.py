"""Real LangGraph and MCP HTTP paths with deterministic evidence and scripted inference."""

import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from dotenv import load_dotenv

import main
import parallel_search
from search_provider import resolve_search_provider

URL = "https://example.org/source?q=research&lang=en"
QUESTION = "What does the research source say?"
SEARCH = {"results": [{"title": "Research source", "url": URL, "excerpts": ["Search evidence"]}]}
PAGES = {"results": [{"url": URL, "excerpts": ["Fetched evidence: the answer is 42."]}], "errors": []}
ASYNC_CLIENT = httpx.AsyncClient


class ProtocolFixture:
    def __init__(self, *, search=SEARCH, pages=PAGES, error=None, http_error=None,
                 stall=None, structured=False, missing_tool=False):
        self.search = search
        self.pages = pages
        self.error = error
        self.http_error = http_error
        self.stall = stall
        self.structured = structured
        self.missing_tool = missing_tool
        self.requests = []
        self.calls = []
        self.closed = False

    async def handle(self, request):
        self.requests.append(request)
        if request.method != "POST":
            return httpx.Response(405)
        message = json.loads(request.content)
        method = message["method"]
        self.calls.append(message)
        if self.stall == method:
            try:
                await asyncio.Event().wait()
            finally:
                self.closed = True
        if method.startswith("notifications/"):
            return httpx.Response(202)
        if self.http_error and method == "tools/call":
            return httpx.Response(self.http_error, text="Provider unavailable")
        if method == "initialize":
            result = {"protocolVersion": message["params"]["protocolVersion"],
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": "test-search", "version": "1"}}
        elif method == "tools/list":
            names = ["web_search"] if self.missing_tool else ["web_search", "web_fetch"]
            result = {"tools": [{"name": name, "inputSchema": {"type": "object"}} for name in names]}
        elif method == "tools/call":
            tool = message["params"]["name"]
            data = self.search if tool == "web_search" else self.pages
            if self.error == tool:
                result = {"content": [{"type": "text", "text": "anonymous quota exceeded"}], "isError": True}
            elif isinstance(data, str):
                result = {"content": [{"type": "text", "text": data}]}
            else:
                result = {"content": [{"type": "text", "text": json.dumps(data)}]}
                if self.structured:
                    result["structuredContent"] = data
        else:
            raise AssertionError(f"Unexpected MCP method {method}")
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": result})

    @contextmanager
    def installed(self):
        def client(**kwargs):
            return ASYNC_CLIENT(transport=httpx.MockTransport(self.handle), **kwargs)
        with patch("parallel_search.httpx.AsyncClient", side_effect=client):
            yield self

    def tool_calls(self):
        return [c["params"] for c in self.calls if c["method"] == "tools/call"]


class ScriptedModel:
    """Scripted inference, not a live model evaluation."""
    def __init__(self):
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        return SimpleNamespace(content=f"Answer 42, source: {URL}")

    def with_structured_output(self, schema):
        return SimpleNamespace(invoke=lambda messages: schema(selected_urls=["https://reddit.com/r/test/comments/1"]))


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.model = ScriptedModel()
        self.model_patch = patch.object(main, "llm", self.model)
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)

    def test_fresh_configuration_through_mcp_graph_and_synthesis(self):
        fixture = ProtocolFixture()
        with fixture.installed(), patch("main.serp_search") as brightdata:
            state = main.graph.invoke({"messages": [], "user_question": QUESTION})
        self.assertEqual(state["search_provider"], "parallel")
        self.assertIn(URL, state["final_answer"])
        self.assertIn("Fetched evidence: the answer is 42.", str(self.model.messages[0]))
        self.assertIn(URL, str(self.model.messages[1]))
        self.assertIn("Parallel", str(self.model.messages[1]))
        brightdata.assert_not_called()
        calls = fixture.tool_calls()
        self.assertEqual([c["name"] for c in calls], ["web_search", "web_fetch"])
        self.assertEqual(calls[0]["arguments"]["session_id"], calls[1]["arguments"]["session_id"])
        self.assertEqual(calls[0]["arguments"]["model_name"], "gpt-4o")
        self.assertEqual(calls[1]["arguments"]["urls"], [URL])
        self.assertEqual(calls[0]["arguments"]["search_queries"], [QUESTION])
        self.assertTrue(any(c["method"] == "tools/list" for c in fixture.calls))
        for request in fixture.requests:
            self.assertEqual(str(request.url), parallel_search.MCP_URL)
            self.assertEqual(request.headers["User-Agent"], parallel_search.USER_AGENT)
            self.assertNotIn("authorization", request.headers)
            self.assertNotIn("cookie", request.headers)
            self.assertEqual(request.extensions["timeout"]["read"], 60)

    def test_configured_brightdata_keeps_google_bing_reddit_workflow(self):
        os.environ["BRIGHTDATA_API_KEY"] = "configured-brightdata"
        with patch("main.parallel_research") as parallel, \
             patch("main.serp_search", return_value={"organic": [{"url": URL}]}) as serp, \
             patch("main.reddit_search_api", return_value={"parsed_posts": [{"url": "https://reddit.com/r/test/comments/1"}]}) as reddit, \
             patch("main.reddit_post_retrieval", return_value={"comments": [{"content": "A comment"}]}) as posts:
            state = main.graph.invoke({"messages": [], "user_question": QUESTION})
        self.assertEqual(state["search_provider"], "brightdata")
        self.assertEqual({c.kwargs["engine"] for c in serp.call_args_list}, {"google", "bing"})
        reddit.assert_called_once_with(keyword=QUESTION)
        posts.assert_called_once_with(["https://reddit.com/r/test/comments/1"])
        parallel.assert_not_called()
        self.assertTrue(state["final_answer"])

    def test_explicit_parallel_overrides_saved_brightdata_and_ambient_credentials(self):
        os.environ.update(SEARCH_PROVIDER="parallel", BRIGHTDATA_API_KEY="configured-brightdata",
                          PARALLEL_API_KEY="unused-credential", HTTP_PROXY="http://127.0.0.1:1")
        with ProtocolFixture(structured=True).installed() as fixture, patch("main.serp_search") as serp:
            state = main.graph.invoke({"messages": [], "user_question": QUESTION})
        self.assertEqual(state["search_provider"], "parallel")
        serp.assert_not_called()
        self.assertTrue(all("authorization" not in r.headers for r in fixture.requests))

    def test_disabled_search_sends_no_requests_or_inference(self):
        os.environ.update(SEARCH_PROVIDER="none", BRIGHTDATA_API_KEY="configured-brightdata")
        with patch("main.parallel_research") as parallel, patch("main.serp_search") as serp, \
             patch("main.reddit_search_api") as reddit:
            state = main.graph.invoke({"messages": [], "user_question": QUESTION})
        parallel.assert_not_called()
        serp.assert_not_called()
        reddit.assert_not_called()
        self.assertEqual(self.model.messages, [])
        self.assertIn("disabled", state["final_answer"])

    def test_resolver_validation_and_explicit_brightdata(self):
        for provider in ["typo", "brightdata", ""]:
            with self.subTest(provider=provider):
                os.environ["SEARCH_PROVIDER"] = provider
                with self.assertRaises(ValueError):
                    resolve_search_provider()
        os.environ.update(SEARCH_PROVIDER=" BRIGHTDATA ", BRIGHTDATA_API_KEY="configured-brightdata")
        self.assertEqual(resolve_search_provider(), "brightdata")
        os.environ.update(SEARCH_PROVIDER="auto", BRIGHTDATA_API_KEY="  ")
        self.assertEqual(resolve_search_provider(), "parallel")

    def test_saved_dotenv_choices_and_environment_precedence(self):
        with TemporaryDirectory() as directory:
            config = Path(directory) / ".env"
            config.write_text("BRIGHTDATA_API_KEY=configured-brightdata\nSEARCH_PROVIDER=brightdata\n")
            load_dotenv(config)
            self.assertEqual(resolve_search_provider(), "brightdata")
            os.environ["SEARCH_PROVIDER"] = "none"
            load_dotenv(config)
            self.assertEqual(resolve_search_provider(), "none")

    def test_explicit_failure_never_falls_back_or_synthesizes(self):
        os.environ.update(SEARCH_PROVIDER="parallel", BRIGHTDATA_API_KEY="configured-brightdata")
        with ProtocolFixture(error="web_search").installed(), patch("main.serp_search") as serp:
            with self.assertRaisesRegex(parallel_search.ParallelSearchError, "quota exceeded"):
                main.graph.invoke({"messages": [], "user_question": QUESTION})
        serp.assert_not_called()
        self.assertEqual(self.model.messages, [])

    def test_distinct_question_sessions_are_not_reused(self):
        with ProtocolFixture().installed() as fixture:
            for _ in range(2):
                main.graph.invoke({"messages": [], "user_question": QUESTION})
        searches = [c for c in fixture.tool_calls() if c["name"] == "web_search"]
        self.assertNotEqual(searches[0]["arguments"]["session_id"], searches[1]["arguments"]["session_id"])


class MCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_url_filtering_deduplication_and_limit(self):
        urls = [None, "file:///etc/passwd", "javascript:alert(1)", "https://user:pass@example.org/",
                "https://[broken", URL, URL, "http://example.net/two", "https://example.com/three", "https://example.com/four"]
        fixture = ProtocolFixture(search={"results": [{"url": u} for u in urls]})
        with fixture.installed():
            await parallel_search.research(QUESTION, session_id="stable-question")
        calls = fixture.tool_calls()
        self.assertEqual(calls[1]["arguments"]["urls"], [URL, "http://example.net/two", "https://example.com/three"])
        self.assertTrue(all(c["arguments"]["session_id"] == "stable-question" for c in calls))

    async def test_empty_results_do_not_fetch(self):
        with ProtocolFixture(search={"results": []}).installed() as fixture:
            result = await parallel_search.research(QUESTION)
        self.assertEqual(result["pages"], {"results": []})
        self.assertEqual(len(fixture.tool_calls()), 1)

    async def test_page_errors_remain_evidence(self):
        pages = {"results": [], "errors": [{"url": URL, "http_status_code": 403}]}
        with ProtocolFixture(pages=pages).installed():
            result = await parallel_search.research(QUESTION)
        self.assertEqual(result["pages"]["errors"][0]["http_status_code"], 403)

    async def test_malformed_and_tool_errors(self):
        cases = [ProtocolFixture(search="bad-json"), ProtocolFixture(search={"unexpected": []}),
                 ProtocolFixture(error="web_fetch"), ProtocolFixture(missing_tool=True),
                 ProtocolFixture(http_error=401), ProtocolFixture(http_error=429), ProtocolFixture(http_error=503)]
        for fixture in cases:
            with self.subTest(fixture=fixture.__dict__), fixture.installed():
                with self.assertRaises(parallel_search.ParallelSearchError):
                    await parallel_search.research(QUESTION)
                # No application-level retry or fallback after an HTTP error.
                self.assertLessEqual(len(fixture.tool_calls()), 2)

    async def test_total_timeout_cleans_up(self):
        with ProtocolFixture(stall="tools/call").installed() as fixture:
            with self.assertRaisesRegex(parallel_search.ParallelSearchError, "timed out"):
                await parallel_search.research(QUESTION, timeout=0.03)
        self.assertTrue(fixture.closed)

    async def test_cancellation_propagates_and_cleans_up(self):
        with ProtocolFixture(stall="tools/call").installed() as fixture:
            task = asyncio.create_task(parallel_search.research(QUESTION))
            for _ in range(100):
                if fixture.tool_calls():
                    break
                await asyncio.sleep(0.001)
            self.assertTrue(fixture.tool_calls())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(fixture.closed)

    async def test_blank_question_has_no_network_effect(self):
        with ProtocolFixture().installed() as fixture:
            with self.assertRaises(ValueError):
                await parallel_search.research("  ")
        self.assertEqual(fixture.requests, [])


if __name__ == "__main__":
    unittest.main()
