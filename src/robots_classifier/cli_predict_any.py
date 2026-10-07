from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from joblib import load

from .io import fetch_robots_txt, read_text_file


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def topk(classes, proba_row, k: int = 5):
    pairs = list(zip(classes, proba_row))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:k]


def main():
    p = argparse.ArgumentParser(description="Ad-hoc prediction for ANY exported model bundle (LR/SVM/SGD).")
    p.add_argument("--model", required=True, help="Path to joblib bundle created by cli_tune.py")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="Local robots.txt file")
    src.add_argument("--url", help="Base URL (fetch /robots.txt)")
    p.add_argument("--timeout", type=int, default=10)
    p.add_argument("--topk", type=int, default=5)
    args = p.parse_args()

    bundle = load(args.model)

    if args.file:
        text = read_text_file(args.file)
        log(f"Loaded robots.txt from file: {args.file}")
    else:
        text = fetch_robots_txt(args.url, timeout=args.timeout)
        log(f"Fetched robots.txt from: {args.url}")

    X = bundle.vectorizer.transform([text])
    pred = bundle.model.predict(X)[0]
    print(f"Predicted category: {pred}")

    if hasattr(bundle.model, "predict_proba"):
        proba = bundle.model.predict_proba(X)[0]
        best = topk(bundle.model.classes_, proba, k=args.topk)
        print("Top probabilities:")
        for cls, p_ in best:
            print(f"  {cls:20s} {float(p_):.4f}")
    else:
        # LinearSVC etc.
        log("Model has no predict_proba (this is normal for LinearSVC).")


if __name__ == "__main__":
    main()