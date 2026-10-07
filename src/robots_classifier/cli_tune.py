from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass
class AnyModelBundle:
    """Generic bundle for ad-hoc usage: vectorizer + fitted model."""
    vectorizer: TfidfVectorizer
    model: Any
    model_name: str
    params: Dict[str, Any]


def load_df(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"robots_content", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")

    df = df.copy()
    df["robots_content"] = df["robots_content"].fillna("").astype(str)
    df["category"] = df["category"].astype(str)
    return df


def make_vectorizer(
    ngram_range: Tuple[int, int],
    max_features: int,
    min_df: int,
    sublinear_tf: bool,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
    )


def fit_and_eval(
    model_name: str,
    vectorizer_params: Dict[str, Any],
    model_params: Dict[str, Any],
    X_train_text: pd.Series,
    y_train: pd.Series,
    X_val_text: pd.Series,
    y_val: pd.Series,
) -> Dict[str, Any]:
    vec = make_vectorizer(**vectorizer_params)
    t0 = time.time()
    Xtr = vec.fit_transform(X_train_text)
    Xva = vec.transform(X_val_text)
    vec_time = time.time() - t0

    if model_name == "lr":
        model = LogisticRegression(
            max_iter=model_params["max_iter"],
            solver=model_params["solver"],
            #n_jobs=model_params["n_jobs"],
            class_weight=model_params["class_weight"],
            C=model_params["C"],
        )
    elif model_name == "linearsvc":
        model = LinearSVC(
            C=model_params["C"],
            class_weight=model_params["class_weight"],
        )
    elif model_name == "sgd_log":
        model = SGDClassifier(
            loss="log_loss",
            alpha=model_params["alpha"],
            max_iter=model_params["max_iter"],
            tol=model_params["tol"],
            class_weight=model_params["class_weight"],
            random_state=model_params["random_state"],
        )
    elif model_name == "sgd_svm":
        model = SGDClassifier(
            loss="hinge",
            alpha=model_params["alpha"],
            max_iter=model_params["max_iter"],
            tol=model_params["tol"],
            class_weight=model_params["class_weight"],
            random_state=model_params["random_state"],
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    t0 = time.time()
    model.fit(Xtr, y_train)
    fit_time = time.time() - t0

    pred = model.predict(Xva)
    acc = float(accuracy_score(y_val, pred))
    macro_f1 = float(f1_score(y_val, pred, average="macro", zero_division=0))

    return {
        "model_name": model_name,
        "vectorizer_params": vectorizer_params,
        "model_params": model_params,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "vec_time_s": vec_time,
        "fit_time_s": fit_time,
        "bundle": AnyModelBundle(vec, model, model_name, {"vec": vectorizer_params, "model": model_params}),
    }


def main():
    p = argparse.ArgumentParser(description="Tune models on a dataset and export the best model for ad-hoc usage.")
    p.add_argument("--csv", required=True, help="CSV path (must include robots_content, category)")
    p.add_argument("--out_csv", default="data/results/tuning_results.csv", help="Where to save the tuning table")
    p.add_argument("--out_model", default="models/best_model.joblib", help="Where to save the best model bundle")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)

    # grid controls (small but effective)
    p.add_argument("--max_features_list", nargs="+", type=int, default=[4000, 8000])
    p.add_argument("--ngram_list", nargs="+", default=["3,4", "3,5"])
    p.add_argument("--C_list", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--min_df_list", nargs="+", type=int, default=[1, 2])
    p.add_argument("--sublinear_tf", action="store_true", help="Use sublinear_tf=True")

    # models to include
    p.add_argument("--include_lr", action="store_true")
    p.add_argument("--include_linearsvc", action="store_true")
    p.add_argument("--include_sgd", action="store_true")

    # choose best by
    p.add_argument("--select_by", choices=["macro_f1", "accuracy"], default="macro_f1")

    args = p.parse_args()

    # default: include lr + linearsvc (good baseline)
    if not (args.include_lr or args.include_linearsvc or args.include_sgd):
        args.include_lr = True
        args.include_linearsvc = True

    out_csv_path = Path(args.out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_model_path = Path(args.out_model)
    out_model_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_df(args.csv)
    n_classes = df["category"].nunique()
    log(f"Loaded {len(df)} rows, {n_classes} classes from {args.csv}")

    X_train_text, X_val_text, y_train, y_val = train_test_split(
        df["robots_content"],
        df["category"],
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["category"],
    )
    log(f"Split: train={len(X_train_text)} val={len(X_val_text)} (stratified)")

    # parse ngram list like "3,4"
    ngram_ranges: List[Tuple[int, int]] = []
    for s in args.ngram_list:
        a, b = s.split(",")
        ngram_ranges.append((int(a), int(b)))

    # Build grid
    vectorizer_grid = []
    for ngram_range in ngram_ranges:
        for max_features in args.max_features_list:
            for min_df in args.min_df_list:
                vectorizer_grid.append(
                    {
                        "ngram_range": ngram_range,
                        "max_features": max_features,
                        "min_df": min_df,
                        "sublinear_tf": bool(args.sublinear_tf),
                    }
                )

    # Model grids
    lr_grid = []
    if args.include_lr:
        for C in args.C_list:
            lr_grid.append(
                {
                    "C": C,
                    "max_iter": 1200,
                    "solver": "saga",
                    "n_jobs": -1,
                    "class_weight": "balanced",
                }
            )

    svc_grid = []
    if args.include_linearsvc:
        for C in args.C_list:
            svc_grid.append(
                {
                    "C": C,
                    "class_weight": "balanced",
                }
            )

    sgd_grid = []
    if args.include_sgd:
        # small set: SGD is fast; alpha is the key
        for alpha in [1e-6, 1e-5, 1e-4]:
            sgd_grid.append(
                {
                    "alpha": alpha,
                    "max_iter": 30,
                    "tol": 1e-3,
                    "class_weight": "balanced",
                    "random_state": args.seed,
                }
            )

    candidates: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    for v in vectorizer_grid:
        for m in lr_grid:
            candidates.append(("lr", v, m))
        for m in svc_grid:
            candidates.append(("linearsvc", v, m))
        for m in sgd_grid:
            candidates.append(("sgd_log", v, m))

    # Also optionally: SGD hinge (SVM) – uncomment if you want
    # for v in vectorizer_grid:
    #     for m in sgd_grid:
    #         candidates.append(("sgd_svm", v, m))

    log(f"Tuning candidates: {len(candidates)}")

    results = []
    best_score = -1.0
    best_bundle: Optional[AnyModelBundle] = None

    for i, (model_name, v_params, m_params) in enumerate(candidates, start=1):
        log(f"[{i}/{len(candidates)}] Train {model_name} | vec={v_params} | model={m_params}")
        r = fit_and_eval(
            model_name,
            v_params,
            m_params,
            X_train_text,
            y_train,
            X_val_text,
            y_val,
        )

        score = r[args.select_by]
        log(f"    -> acc={r['accuracy']:.4f} macroF1={r['macro_f1']:.4f} (vec={r['vec_time_s']:.1f}s fit={r['fit_time_s']:.1f}s)")

        results.append(
            {
                "model_name": r["model_name"],
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "vec_time_s": r["vec_time_s"],
                "fit_time_s": r["fit_time_s"],
                "ngram_range": str(v_params["ngram_range"]),
                "max_features": v_params["max_features"],
                "min_df": v_params["min_df"],
                "sublinear_tf": v_params["sublinear_tf"],
                "model_params": str(m_params),
            }
        )

        if score > best_score:
            best_score = score
            best_bundle = r["bundle"]

    # Save table
    res_df = pd.DataFrame(results).sort_values(by=args.select_by, ascending=False)
    res_df.to_csv(out_csv_path, index=False)
    log(f"Saved tuning table -> {out_csv_path}")

    if best_bundle is None:
        raise RuntimeError("No model trained (empty candidate list).")

    # Save best bundle
    dump(best_bundle, out_model_path)
    log(f"Saved BEST model bundle -> {out_model_path}")
    log(f"Best by {args.select_by}: {best_score:.4f} | model={best_bundle.model_name} | params={best_bundle.params}")


if __name__ == "__main__":
    main()