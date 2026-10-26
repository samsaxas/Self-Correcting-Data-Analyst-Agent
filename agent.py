"""Self-correcting Data Analyst agent built with LangGraph.

Flow: profile -> write_code -> run_code -> (error? retry write_code) -> summarize
"""
import os
import re
import sys
from typing import TypedDict

import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

MAX_ATTEMPTS = 3
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # change if your key uses another model
llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0)

# The generated code may only use these builtins (no imports, no file or network access).
SAFE_BUILTINS = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
                 for n in ["len", "sum", "min", "max", "round", "sorted", "range", "int",
                           "float", "str", "list", "dict", "set", "abs", "enumerate", "zip"]}


class State(TypedDict):
    question: str
    schema: str
    code: str
    result: str
    error: str
    attempts: int
    answer: str


def _text(msg) -> str:
    c = msg.content
    return c if isinstance(c, str) else "".join(p if isinstance(p, str) else p.get("text", "") for p in c)


def build_graph(df: pd.DataFrame):
    def profile(state: State):
        schema = f"Columns and dtypes:\n{df.dtypes.to_string()}\n\nFirst rows:\n{df.head(3).to_string()}"
        return {"schema": schema, "attempts": 0, "error": ""}

    def write_code(state: State):
        prompt = (
            "You are a data analyst. Write Python (pandas) code that answers the question.\n"
            "Rules: a DataFrame named df and pandas as pd are already available; do not import "
            "anything; store the final answer in a variable named result; return code only.\n\n"
            f"{state['schema']}\n\nQuestion: {state['question']}\n"
        )
        if state["error"]:
            prompt += f"\nYour previous code:\n{state['code']}\nfailed with: {state['error']}\nFix it."
        code = re.sub(r"```(?:python)?", "", _text(llm.invoke(prompt))).strip()
        return {"code": code, "attempts": state["attempts"] + 1}

    def run_code(state: State):
        scope = {"df": df.copy(), "pd": pd}
        try:
            exec(state["code"], {"__builtins__": SAFE_BUILTINS}, scope)
            if "result" not in scope:
                raise ValueError("code must set a variable named result")
            return {"result": str(scope["result"]), "error": ""}
        except Exception as e:  # feed the error back to the model
            return {"result": "", "error": f"{type(e).__name__}: {e}"}

    def route(state: State):
        if not state["error"]:
            return "summarize"
        return "write_code" if state["attempts"] < MAX_ATTEMPTS else "give_up"

    def summarize(state: State):
        prompt = (f"Question: {state['question']}\nComputed result: {state['result']}\n"
                  "Answer the question in one clear sentence using the result.")
        return {"answer": _text(llm.invoke(prompt)).strip()}

    def give_up(state: State):
        return {"answer": f"Could not answer after {state['attempts']} attempts. Last error: {state['error']}"}

    g = StateGraph(State)
    for name, fn in [("profile", profile), ("write_code", write_code), ("run_code", run_code),
                     ("summarize", summarize), ("give_up", give_up)]:
        g.add_node(name, fn)
    g.add_edge(START, "profile")
    g.add_edge("profile", "write_code")
    g.add_edge("write_code", "run_code")
    g.add_conditional_edges("run_code", route, ["write_code", "summarize", "give_up"])
    g.add_edge("summarize", END)
    g.add_edge("give_up", END)
    return g.compile()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sales.csv"
    app = build_graph(pd.read_csv(path))
    while True:
        q = input("\nAsk a question about the data (or 'quit'): ").strip()
        if q.lower() in {"quit", "exit", ""}:
            break
        out = app.invoke({"question": q})
        print(f"\nCode:\n{out['code']}\n\nAnswer: {out['answer']}  (attempts: {out['attempts']})")
