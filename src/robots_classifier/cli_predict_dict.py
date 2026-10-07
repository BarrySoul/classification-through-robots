from __future__ import annotations

import argparse
import time
import numpy as np
from joblib import load

from .io import fetch_robots_txt, read_text_file


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Ad-hoc prediction (LinearSVC + TF-IDF).")
    p.add_argument("--model", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--file")
    src.add_argument("--url")
    p.add_argument("--timeout", type=int, default=10)
    p.add_argument("--topk", type=int, default=5)
    args = p.parse_args()

    bundle = load(args.model)
    vec = bundle["vectorizer"]
    clf = bundle["model"]

    if args.file:
        text = read_text_file(args.file)
        log(f"Loaded robots.txt from file: {args.file}")
    else:
        text = fetch_robots_txt(args.url, timeout=args.timeout)
        log(f"Fetched robots.txt from: {args.url}")

    X = vec.transform([text])

    # Predicted class
    pred = clf.predict(X)[0]
    print(f"\nPredicted category: {pred}")

    # Top scores using decision_function (SVM margin scores)
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)[0]

        # Sort classes by score descending
        idx_sorted = np.argsort(scores)[::-1]
        classes = clf.classes_

        print("Top scores:")
        for rank, idx in enumerate(idx_sorted[:args.topk], start=1):
            print(f"{rank}. {classes[idx]}")
    else:
        print("Model does not support decision_function.")


if __name__ == "__main__":
    main()