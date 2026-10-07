from __future__ import annotations

import argparse

from .config import LRConfig, SplitConfig, TfidfConfig
from .data import load_dataset, stratified_split
from .evaluation import compute_metrics
from .models import train_lr, predict_lr, save_bundle


def main():
    p = argparse.ArgumentParser(description="Part (4): train and export an ad-hoc classifier bundle (LR TF-IDF).")
    p.add_argument("--csv", required=True)
    p.add_argument("--out_model", required=True, help="Output joblib bundle path, e.g. model.joblib")
    p.add_argument("--max_features", type=int, default=4000)
    p.add_argument("--C", type=float, default=1.0)
    args = p.parse_args()

    split_cfg = SplitConfig()
    tfidf_cfg = TfidfConfig(max_features=args.max_features)
    lr_cfg = LRConfig(C=args.C)

    ds = load_dataset(args.csv)
    train_df, test_df = stratified_split(ds.df, split_cfg.test_size, split_cfg.random_state)

    bundle = train_lr(train_df["robots_content"], train_df["category"], tfidf_cfg, lr_cfg)
    pred, _ = predict_lr(bundle, list(test_df["robots_content"]))
    metrics = compute_metrics(test_df["category"], pred)

    save_bundle(bundle, args.out_model)

    print(f"Dataset: {ds.name} | rows={ds.n_rows} | classes={ds.n_classes}")
    print(f"Saved bundle -> {args.out_model}")
    print(f"Accuracy: {metrics.accuracy:.4f}")
    print(f"Macro-F1: {metrics.macro_f1:.4f}")
    print(metrics.report)


if __name__ == "__main__":
    main()