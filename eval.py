"""Tiny eval: 5 questions with ground truth computed directly in pandas."""
import os
import time

import numpy as np
import pandas as pd

from agent import build_graph

CSV = "data/sales.csv"


def make_data():
    rng = np.random.default_rng(42)
    prices = {"Laptop": 900, "Phone": 600, "Tablet": 400, "Headphones": 80}
    n = 300
    df = pd.DataFrame({
        "date": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 180, n), unit="D"),
        "region": rng.choice(["North", "South", "East", "West"], n),
        "product": rng.choice(list(prices), n),
        "units": rng.integers(1, 21, n),
    })
    df["revenue"] = df["units"] * df["product"].map(prices)
    os.makedirs("data", exist_ok=True)
    df.to_csv(CSV, index=False)


def correct(result: str, expected) -> bool:
    try:
        return abs(float(result) - float(expected)) <= 0.01 * abs(float(expected))
    except (ValueError, TypeError):
        return str(expected).lower() in result.lower()


if __name__ == "__main__":
    if not os.path.exists(CSV):
        make_data()
    df = pd.read_csv(CSV, parse_dates=["date"])
    app = build_graph(df)
    tests = [
        ("What is the total revenue?", df["revenue"].sum()),
        ("Which region has the highest total revenue?", df.groupby("region")["revenue"].sum().idxmax()),
        ("What is the average units per order?", df["units"].mean()),
        ("Which product has the lowest total revenue?", df.groupby("product")["revenue"].sum().idxmin()),
        ("How many orders were placed in March 2025?", int(((df.date.dt.year == 2025) & (df.date.dt.month == 3)).sum())),
    ]
    right, retries = 0, 0
    for q, expected in tests:
        out = app.invoke({"question": q})
        ok = correct(out["result"], expected)
        right += ok
        retries += out["attempts"] - 1
        print(f"[{'OK' if ok else 'FAIL'}] {q} -> {out['result']} (expected {expected}, attempts {out['attempts']})")
        time.sleep(4)  # stay under free-tier rate limits
    print(f"\nAccuracy: {right}/{len(tests)}   self-corrections used: {retries}")
