# 🌐 LangGraph Research Web Agent

An autonomous AI research agent that can search the live internet, open webpages through proxies, read articles, and generate structured research reports using **LangGraph + LangChain + OpenAI**, with **Parallel Search MCP** or **Bright Data** for research.

Unlike a normal chatbot, this system does not rely only on pretrained knowledge.
It actively performs real-time research before answering.

---

## 🧠 What This Project Is

This project implements a **tool-using AI agent**.

The agent:

* Understands a research question
* Plans how to gather information
* Searches the internet
* Reads source pages with Parallel or uses Bright Data proxy and Reddit tools
* Extracts important content
* Summarizes findings using an LLM
* Produces a final researched answer

It behaves closer to a **human research assistant** than a chatbot.

---

## 🚀 Key Features

* Autonomous multi-step reasoning
* Live internet research
* Free, anonymous Parallel web search and source page fetching
* Bright Data proxy browsing and Reddit datasets when configured
* LLM-based summarization
* Stateful workflow using LangGraph
* Memory between reasoning steps

---

## 🧩 Technologies Used

| Technology         | Purpose                                       |
| ------------------ | --------------------------------------------- |
| **LangGraph**      | Agent workflow and state machine              |
| **LangChain**      | Tool integration & LLM interface              |
| **OpenAI API**     | Reasoning and summarization                   |
| **Parallel MCP**   | Anonymous web search and source page excerpts |
| **Bright Data**    | Google/Bing SERPs and Reddit datasets when configured |
| **Pydantic**       | Structured agent state                        |

---

## 🏗️ Agent Workflow

User Question
↓
Planner (decides next step)
↓
Select Search Provider
↓
Parallel Search + Source Pages, or Bright Data Google/Bing + Reddit
↓
Extract Page Content
↓
LLM Analysis
↓
Iterate (if more info needed)
↓
Final Research Report

This workflow is implemented using a **LangGraph state graph** instead of a single prompt chain.

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key
# Optional: retain the original Google/Bing/Reddit workflow
# BRIGHTDATA_API_KEY=your_brightdata_api_key
```

With no search configuration, the agent uses [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
for free web search and page fetching, without a search account or API key. Anonymous
search uses Fast mode. OpenAI credentials are still required for analysis and synthesis.

`SEARCH_PROVIDER` controls routing (set it in `.env` or the shell):

| Value | Behavior |
| --- | --- |
| `auto` (default) | Use Bright Data when `BRIGHTDATA_API_KEY` is nonempty; otherwise use Parallel. |
| `parallel` | Use anonymous Parallel even if a Bright Data key is configured. |
| `brightdata` | Use the original Google/Bing/Reddit workflow; requires `BRIGHTDATA_API_KEY`. |
| `none` | Disable all research requests and return a disabled-search message. |

Existing environment values take precedence over `.env`. Invalid settings and provider
failures are reported; they never silently switch providers. Parallel does not use
ambient search credentials or access the Bright Data Reddit datasets. It searches
once per question and fetches excerpts from up to three distinct HTTP(S) source URLs
before analysis and synthesis. Source URLs remain available for citations. Page-level
retrieval errors remain in the evidence; a failed MCP request stops that question.
Each Parallel research operation has a 60-second total timeout. Free service limits
apply; a quota error is reported without automatic retries. Questions and selected
source URLs are sent to Parallel only when that provider is selected.

---

## ▶️ Running the Agent

Use Python 3.12 and install the runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Then enter a research query:

Example:

```
Impact of AI in modern healthcare systems
```

The agent will:

1. Plan research
2. Search the internet
3. Fetch source pages (Parallel) or Reddit datasets (Bright Data)
4. Extract content
5. Produce a researched answer

---

## 🧠 How LangGraph Is Used

This project does **not** use a simple prompt chain.

Instead it uses a *stateful reasoning graph*.

Agent nodes:

* Planner → decides next action
* Provider selection → preserves configured Bright Data access or selects Parallel
* Search Tool → finds sources
* Source retrieval → fetches page excerpts or Bright Data Reddit comments
* Analyzer → LLM reads content
* Summarizer → produces final answer

The graph loops until enough information is collected.

This makes the system an **AI Agent**, not a chatbot.

---

## 📂 Project Structure

```
Langflow-Research-Web-Agent/
│
├── main.py
├── parallel_search.py
├── search_provider.py
├── requirements.txt
├── web_operations.py/
├── snapshot_operations.py/
├── prompts.py/
├── .python-version/
└── .env
```

---

## ⚠️ Common Issues

### 400 Bad Request (Bright Data)

Usually:

* Wrong zone name
* Invalid API key
* No proxy permission

### OpenAI error

Check your API key in `.env`.

---

## 🎯 What You Learn From This Project

* How real AI agents operate
* Tool calling with LLMs
* LangGraph state machines
* Autonomous research pipelines
* Proxy-based web retrieval
* Multi-step reasoning systems

---


## Verification

Run the deterministic suite without API credentials:

```bash
python -m unittest discover -s tests -v
```

The suite uses the real LangGraph workflow and HTTP MCP client with protocol fixtures
and a scripted analysis model. It does not call OpenAI or the live search service.
