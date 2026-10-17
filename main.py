from dotenv import load_dotenv
import json
from typing import Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from web_operations import serp_search, reddit_search_api, reddit_post_retrieval
from parallel_search import parallel_research, ParallelSearchError
from search_provider import resolve_search_provider
from promts import (
    get_reddit_analysis_messages,
    get_google_analysis_messages,
    get_bing_analysis_messages,
    get_reddit_url_analysis_messages,
    get_synthesis_messages,
    get_parallel_analysis_messages,
    get_parallel_synthesis_messages,
)

load_dotenv()

llm = None


def get_model():
    global llm
    if llm is None:
        llm = init_chat_model("gpt-4o")
    return llm

class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_question: str | None
    google_results: str | None
    bing_results: str | None
    reddit_results: str | None
    selected_reddit_urls: list[str] | None
    reddit_post_data: list | None
    google_analysis: str | None
    bing_analysis: str | None
    reddit_analysis: str | None
    final_answer: str | None
    search_provider: str | None
    parallel_results: dict | None
    parallel_analysis: str | None


class RedditURLAnalysis(BaseModel):
    selected_urls: List[str] = Field(description="List of Reddit URLs that contain valuable information for answering the user's question")


def select_search(state: State):
    provider = resolve_search_provider()
    # Initialize once before the original graph fans out into concurrent analyses.
    if provider != "none":
        get_model()
    return {"search_provider": provider}


def search_routes(state: State):
    if state["search_provider"] == "brightdata":
        return ["google_search", "bing_search", "reddit_search"]
    return "parallel_search" if state["search_provider"] == "parallel" else "no_search"


def parallel_search(state: State):
    print(f"Searching with Parallel for: {state['user_question']}")
    return {"parallel_results": parallel_research(state["user_question"])}


def analyze_parallel_results(state: State):
    messages = get_parallel_analysis_messages(
        state["user_question"], json.dumps(state["parallel_results"], ensure_ascii=False)
    )
    return {"parallel_analysis": get_model().invoke(messages).content}


def no_search(state: State):
    answer = "Web search is disabled (SEARCH_PROVIDER=none)."
    return {"final_answer": answer, "messages": [{"role": "assistant", "content": answer}]}


def google_search(state: State):
    user_question = state.get("user_question", "")
    print(f"Searching Google for: {user_question}")

    google_results = serp_search(user_question, engine="google")

    return {"google_results": google_results}


def bing_search(state: State):
    user_question = state.get("user_question", "")
    print(f"Searching Bing for: {user_question}")

    bing_results = serp_search(user_question, engine="bing")

    return {"bing_results": bing_results}


def reddit_search(state: State):
    user_question = state.get("user_question", "")
    print(f"Searching Reddit for: {user_question}")

    reddit_results = reddit_search_api(keyword=user_question)
    print(reddit_results)

    return {"reddit_results": reddit_results}


def analyze_reddit_posts(state: State):
    user_question = state.get("user_question", "")
    reddit_results = state.get("reddit_results", "")

    if not reddit_results:
        return {"selected_reddit_urls": []}


    messages = get_reddit_url_analysis_messages(user_question, reddit_results)
    print("________________")
    print(messages)
    print("----------------")
    structured_llm = get_model().with_structured_output(RedditURLAnalysis)

    try:
        analysis = structured_llm.invoke(messages)
        selected_urls = analysis.selected_urls

        print("Selected URLs:")
        for i, url in enumerate(selected_urls, 1):
            print(f"   {i}. {url}")

    except Exception as e:
        print(e)
        selected_urls = []

    return {"selected_reddit_urls": selected_urls}


def retrieve_reddit_posts(state: State):
    print("Getting reddit post comments")

    selected_urls = state.get("selected_reddit_urls", [])

    if not selected_urls:
        return {"reddit_post_data": []}

    print(f"Processing {len(selected_urls)} Reddit URLs")

    reddit_post_data = reddit_post_retrieval(selected_urls)

    if reddit_post_data:
        print(f"Successfully got {len(reddit_post_data)} posts")
    else:
        print("Failed to get post data")
        reddit_post_data = []

    print(reddit_post_data)
    return {"reddit_post_data": reddit_post_data}


def analyze_google_results(state: State):
    print("Analyzing google search results")

    user_question = state.get("user_question", "")
    google_results = state.get("google_results", "")

    messages = get_google_analysis_messages(user_question, google_results)
    reply = get_model().invoke(messages)

    return {"google_analysis": reply.content}


def analyze_bing_results(state: State):
    print("Analyzing bing search results")

    user_question = state.get("user_question", "")
    bing_results = state.get("bing_results", "")

    messages = get_bing_analysis_messages(user_question, bing_results)
    reply = get_model().invoke(messages)

    return {"bing_analysis": reply.content}


def analyze_reddit_results(state: State):
    print("Analyzing reddit search results")

    user_question = state.get("user_question", "")
    reddit_results = state.get("reddit_results", "")
    reddit_post_data = state.get("reddit_post_data", "")

    messages = get_reddit_analysis_messages(user_question, reddit_results, reddit_post_data)
    reply = get_model().invoke(messages)

    return {"reddit_analysis": reply.content}


def synthesize_analyses(state: State):
    print("Combine all results together")

    user_question = state.get("user_question", "")
    google_analysis = state.get("google_analysis", "")
    bing_analysis = state.get("bing_analysis", "")
    reddit_analysis = state.get("reddit_analysis", "")

    if state.get("search_provider") == "parallel":
        messages = get_parallel_synthesis_messages(user_question, state["parallel_analysis"])
    else:
        messages = get_synthesis_messages(
            user_question, google_analysis, bing_analysis, reddit_analysis
        )

    reply = get_model().invoke(messages)
    final_answer = reply.content

    return {"final_answer": final_answer, "messages": [{"role": "assistant", "content": final_answer}]}





graph_builder = StateGraph(State)

graph_builder.add_node("select_search", select_search)
graph_builder.add_node("parallel_search", parallel_search)
graph_builder.add_node("analyze_parallel_results", analyze_parallel_results)
graph_builder.add_node("no_search", no_search)
graph_builder.add_node("google_search", google_search)
graph_builder.add_node("bing_search", bing_search)
graph_builder.add_node("reddit_search", reddit_search)
graph_builder.add_node("analyze_reddit_posts", analyze_reddit_posts)
graph_builder.add_node("retrieve_reddit_posts", retrieve_reddit_posts)
graph_builder.add_node("analyze_google_results", analyze_google_results)
graph_builder.add_node("analyze_bing_results", analyze_bing_results)
graph_builder.add_node("analyze_reddit_results", analyze_reddit_results)
graph_builder.add_node("synthesize_analyses", synthesize_analyses)

graph_builder.add_edge(START, "select_search")
graph_builder.add_conditional_edges("select_search", search_routes,
    ["google_search", "bing_search", "reddit_search", "parallel_search", "no_search"])
graph_builder.add_edge("parallel_search", "analyze_parallel_results")
graph_builder.add_edge("analyze_parallel_results", "synthesize_analyses")
graph_builder.add_edge("no_search", END)

graph_builder.add_edge("google_search", "analyze_reddit_posts")
graph_builder.add_edge("bing_search", "analyze_reddit_posts")
graph_builder.add_edge("reddit_search", "analyze_reddit_posts")
graph_builder.add_edge("analyze_reddit_posts", "retrieve_reddit_posts")

graph_builder.add_edge("retrieve_reddit_posts", "analyze_google_results")
graph_builder.add_edge("retrieve_reddit_posts", "analyze_bing_results")
graph_builder.add_edge("retrieve_reddit_posts", "analyze_reddit_results")

graph_builder.add_edge("analyze_google_results", "synthesize_analyses")
graph_builder.add_edge("analyze_bing_results", "synthesize_analyses")
graph_builder.add_edge("analyze_reddit_results", "synthesize_analyses")

graph_builder.add_edge("synthesize_analyses", END)

graph = graph_builder.compile()

def run_chatbot():
    print("Multi-Source Research Agent")
    print("Type 'exit' to quit\n")

    while True:
        user_input = input("Ask me anything: ").strip()
        if user_input.lower() == "exit":
            print("Bye")
            break
        if not user_input:
            continue

        state = {
            "messages": [{"role": "user", "content": user_input}],
            "user_question": user_input,
            "google_results": None,
            "bing_results": None,
            "reddit_results": None,
            "selected_reddit_urls": None,
            "reddit_post_data": None,
            "google_analysis": None,
            "bing_analysis": None,
            "reddit_analysis": None,
            "final_answer": None,
            "search_provider": None,
            "parallel_results": None,
            "parallel_analysis": None,
        }

        print("\nStarting parallel research process...")
        try:
            final_state = graph.invoke(state)
        except (ParallelSearchError, ValueError) as exc:
            print(f"Research failed: {exc}")
            continue

        if final_state.get("final_answer"):
            print(f"\nFinal Answer:\n{final_state.get('final_answer')}\n")

        print("-" * 80)


if __name__ == "__main__":
    run_chatbot()
