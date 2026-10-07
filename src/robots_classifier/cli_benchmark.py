from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from .config import DEFAULT_TARGET_BOTS, LRConfig, RFConfig, SplitConfig, TfidfConfig
from .data import load_dataset, stratified_split
from .evaluation import compute_metrics
from .models import train_lr, train_rf, predict_lr, predict_rf


def log(msg: str) -> None:
    # Logging simple et lisible
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser(
        description=(
            "Part (3): automatic assessment benchmarks.\n"
            "- LR (TF-IDF) runs on all datasets.\n"
            "- RF (TF-IDF + numeric bots/exclusions) runs only on selected datasets."
        )
    )
    p.add_argument("--csvs", nargs="+", required=True, help="List of dataset CSVs")
    p.add_argument("--out", default="benchmark_results.csv")

    p.add_argument("--max_features_lr", type=int, default=4000)
    p.add_argument("--C", type=float, default=1.0)

    # RF params (used only if dataset is allowed for RF)
    p.add_argument("--max_features_rf_text", type=int, default=1500)
    p.add_argument("--rf_trees", type=int, default=150)

    # Ultra simple control: run RF only if dataset name contains one of these tokens
    p.add_argument(
        "--rf_only_on",
        nargs="*",
        default=["2500"],
        help="Run RF only on datasets whose filename contains any of these tokens (default: 2500).",
    )

    args = p.parse_args()

    split_cfg = SplitConfig()
    lr_cfg = LRConfig(C=args.C)
    tfidf_lr_cfg = TfidfConfig(max_features=args.max_features_lr)
    rf_cfg = RFConfig(n_estimators=args.rf_trees)

    rows = []

    log(f"Starting benchmark on {len(args.csvs)} dataset(s)")
    log(f"LR: TF-IDF max_features={args.max_features_lr}, C={args.C}")
    log(f"RF: TF-IDF max_features={args.max_features_rf_text}, trees={args.rf_trees}, only_on={args.rf_only_on}")

    for idx, csv in enumerate(args.csvs, start=1):
        dataset_timer = time.time()
        csv_path = Path(csv)

        log(f"[{idx}/{len(args.csvs)}] Loading dataset: {csv_path}")
        ds = load_dataset(csv_path)

        log(f"[{idx}/{len(args.csvs)}] Loaded rows={ds.n_rows}, classes={ds.n_classes}")
        log(f"[{idx}/{len(args.csvs)}] Train/test split (stratified), test_size={split_cfg.test_size}")
        train_df, test_df = stratified_split(ds.df, split_cfg.test_size, split_cfg.random_state)

        # -------------------
        # LR on all datasets
        # -------------------
        log(f"[{idx}/{len(args.csvs)}] Training LR...")
        t0 = time.time()
        lr_bundle = train_lr(train_df["robots_content"], train_df["category"], tfidf_lr_cfg, lr_cfg)
        log(f"[{idx}/{len(args.csvs)}] LR trained in {time.time() - t0:.1f}s")

        log(f"[{idx}/{len(args.csvs)}] Evaluating LR...")
        lr_pred, _ = predict_lr(lr_bundle, list(test_df["robots_content"]))
        lr_metrics = compute_metrics(test_df["category"], lr_pred)
        log(f"[{idx}/{len(args.csvs)}] LR acc={lr_metrics.accuracy:.4f} macroF1={lr_metrics.macro_f1:.4f}")

        # -------------------
        # RF only on selected datasets (default: only dataset_2500)
        # -------------------
        run_rf = any(token in csv_path.name for token in args.rf_only_on)

        rf_metrics = None
        if run_rf:
            log(f"[{idx}/{len(args.csvs)}] Training RF (bots/exclusions demo)...")
            t0 = time.time()
            rf_bundle = train_rf(
                train_df["robots_content"],
                train_df["category"],
                tfidf_cfg=TfidfConfig(max_features=args.max_features_rf_text),
                rf_cfg=rf_cfg,
                max_features_text=args.max_features_rf_text,
                target_bots=DEFAULT_TARGET_BOTS,
            )
            log(f"[{idx}/{len(args.csvs)}] RF trained in {time.time() - t0:.1f}s")

            log(f"[{idx}/{len(args.csvs)}] Evaluating RF...")
            rf_pred, _ = predict_rf(rf_bundle, list(test_df["robots_content"]))
            rf_metrics = compute_metrics(test_df["category"], rf_pred)
            log(f"[{idx}/{len(args.csvs)}] RF acc={rf_metrics.accuracy:.4f} macroF1={rf_metrics.macro_f1:.4f}")
        else:
            log(f"[{idx}/{len(args.csvs)}] Skipping RF (dataset not in rf_only_on)")

        # Save row
        row = {
            "dataset": csv_path.name,
            "n_rows": ds.n_rows,
            "n_classes": ds.n_classes,
            "lr_acc": lr_metrics.accuracy,
            "lr_macro_f1": lr_metrics.macro_f1,
            "rf_acc": None if rf_metrics is None else rf_metrics.accuracy,
            "rf_macro_f1": None if rf_metrics is None else rf_metrics.macro_f1,
            "elapsed_s": time.time() - dataset_timer,
        }
        rows.append(row)

        log(f"[{idx}/{len(args.csvs)}] Done: {csv_path.name} in {row['elapsed_s']:.1f}s")
        print("")  # ligne vide

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    log(f"Saved benchmark table -> {out_path}")


if __name__ == "__main__":
    main()