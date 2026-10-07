from __future__ import annotations

import argparse
import time
from pathlib import Path

from joblib import dump
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_df(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "robots_content" not in df.columns or "category" not in df.columns:
        raise ValueError("CSV must contain columns: robots_content, category")
    df = df.copy()
    df["robots_content"] = df["robots_content"].fillna("").astype(str)
    df["category"] = df["category"].astype(str)
    return df


def main():
    p = argparse.ArgumentParser(
        description="Train FINAL best model (LinearSVC + char TF-IDF) for ad-hoc robots.txt classification."
    )
    p.add_argument("--csv", required=True)
    p.add_argument("--out_model", required=True, help="Output joblib file, e.g. models/final_best_2500.joblib")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)

    # Best config (defaults set to your winner)
    p.add_argument("--ngram_min", type=int, default=3)
    p.add_argument("--ngram_max", type=int, default=4)
    p.add_argument("--max_features", type=int, default=8000)
    p.add_argument("--min_df", type=int, default=2)
    p.add_argument("--sublinear_tf", action="store_true", help="Enable sublinear_tf (recommended)")
    p.add_argument("--C", type=float, default=2.0)

    args = p.parse_args()

    out_path = Path(args.out_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_df(args.csv)
    log(f"Loaded {len(df)} rows, {df['category'].nunique()} classes from {args.csv}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["robots_content"],
        df["category"],
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["category"],
    )
    log(f"Split train={len(X_train)} test={len(X_test)} (stratified)")

    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(args.ngram_min, args.ngram_max),
        max_features=args.max_features,
        min_df=args.min_df,
        sublinear_tf=bool(args.sublinear_tf),
    )

    log("Fitting TF-IDF...")
    t0 = time.time()
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    log(f"TF-IDF done in {time.time() - t0:.1f}s")

    clf = LinearSVC(C=args.C, class_weight="balanced")

    log("Training LinearSVC...")
    t0 = time.time()
    clf.fit(Xtr, y_train)
    log(f"LinearSVC trained in {time.time() - t0:.1f}s")

    log("Evaluating...")
    pred = clf.predict(Xte)
    acc = accuracy_score(y_test, pred)
    f1m = f1_score(y_test, pred, average="macro", zero_division=0)

    print("\n=== FINAL RESULTS ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro-F1:  {f1m:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, pred, zero_division=0))

    # Save as plain dict (robust, no custom class pickling)
    bundle = {
        "vectorizer": vec,
        "model": clf,
        "meta": {
            "model_name": "linearsvc",
            "ngram_range": (args.ngram_min, args.ngram_max),
            "max_features": args.max_features,
            "min_df": args.min_df,
            "sublinear_tf": bool(args.sublinear_tf),
            "C": args.C,
            "seed": args.seed,
            "test_size": args.test_size,
            "n_rows": len(df),
            "n_classes": int(df["category"].nunique()),
        },
    }

    dump(bundle, out_path)
    log(f"Saved final bundle -> {out_path}")


if __name__ == "__main__":
    main()