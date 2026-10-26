# Self-Correcting Data Analyst Agent (LangGraph)

Ask a plain-English question about a CSV. A LangGraph state machine profiles the data, has Gemini
write pandas code, runs it in a restricted scope, and **loops back with the error message** if the
code fails (up to 3 attempts) before summarising the answer.

```
START -> profile -> write_code -> run_code --ok--> summarize -> END
                        ^             |
                        +---error-----+ (max 3 attempts, then give_up -> END)
```

## Run
```bash
pip install -r requirements.txt
export GOOGLE_API_KEY=your_key        # Windows: set GOOGLE_API_KEY=your_key
python eval.py                        # creates data/sales.csv and runs 5 checked questions
python agent.py data/sales.csv        # interactive mode
```

## Files
- `agent.py`  LangGraph graph (5 nodes, 1 conditional retry edge) and CLI
- `eval.py`   synthetic sales data and 5 questions checked against pandas ground truth
- `requirements.txt`


## Results

Evaluated on a synthetic sales table (300 orders, 4 regions, 4 products) with 5 questions whose
ground truth is computed directly in pandas.

| Metric | Result |
|---|---|
| Correct answers | 5/5 |
| Answered on first attempt | 5/5 |
| Self-corrections used | 0 |


## Limitations
- Generated code runs with a restricted builtins set and no imports, which is not a full sandbox.
  Do not point it at untrusted users or sensitive data without container isolation.
- Single table only; no multi-turn memory.




