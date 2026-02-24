# 🌐 LangGraph Research Web Agent (BrightData Powered)

An autonomous AI research agent that can search the live internet, open webpages through proxies, read articles, and generate structured research reports using **LangGraph + LangChain + OpenAI + Bright Data**.

Unlike a normal chatbot, this system does not rely only on pretrained knowledge.
It actively performs real-time research before answering.

---

## 🧠 What This Project Is

This project implements a **tool-using AI agent**.

The agent:

* Understands a research question
* Plans how to gather information
* Searches the internet
* Visits websites using Bright Data proxies
* Extracts important content
* Summarizes findings using an LLM
* Produces a final researched answer

It behaves closer to a **human research assistant** than a chatbot.

---

## 🚀 Key Features

* Autonomous multi-step reasoning
* Live internet research
* Bright Data proxy browsing
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
| **Bright Data**    | Web search & page retrieval via proxy network |
| **Pydantic**       | Structured agent state                        |

---

## 🏗️ Agent Workflow

User Question
↓
Planner (decides next step)
↓
Bright Data Search Request
↓
Open Webpage via Proxy
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
BRIGHTDATA_API_KEY=your_brightdata_api_key
```

> Bright Data is required because the agent fetches real webpages.
> Without proxy access, many websites block automated requests.

---

## ▶️ Running the Agent

Run:

```bash
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
3. Visit webpages
4. Extract content
5. Produce a researched answer

---

## 🧠 How LangGraph Is Used

This project does **not** use a simple prompt chain.

Instead it uses a *stateful reasoning graph*.

Agent nodes:

* Planner → decides next action
* Search Tool → finds sources
* Browser Tool → opens pages via Bright Data
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

