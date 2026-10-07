from __future__ import annotations

import argparse

from .evaluation import topk_probabilities
from .io import fetch_robots_txt, read_text_file
from .models import load_bundle, predict_lr


def main():
    p = argparse.ArgumentParser(description="Part (4): ad-hoc prediction (file or URL).")
    p.add_argument("--model", required=True, help="Path to joblib bundle produced by cli_train")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="Local robots.txt file path")
    src.add_argument("--url", help="Base URL (we fetch /robots.txt)")
    p.add_argument("--timeout", type=int, default=10)
    p.add_argument("--topk", type=int, default=5)
    args = p.parse_args()

    bundle = load_bundle(args.model)

    if args.file:
        text = read_text_file(args.file)
    else:
        text = fetch_robots_txt(args.url, timeout=args.timeout)

    pred, proba = predict_lr(bundle, [text])
    label = pred[0]
    print(f"Predicted category: {label}")

    if proba is not None:
        top = topk_probabilities(bundle.model.classes_, proba[0], k=args.topk)
        print("Top probabilities:")
        for cls, p_ in top:
            print(f"  {cls:20s} {float(p_):.4f}")


if __name__ == "__main__":
    main()