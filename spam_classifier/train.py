"""Train and evaluate the per-channel spam models.

Downloads the datasets (cached in data/raw/), trains both models and writes them to models/:

    python -m spam_classifier.train
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from . import __version__
from .data import DEFAULT_DATA_DIR, SOURCES, Example, deduplicate, load_examples
from .model import CHANNELS, DEFAULT_MODEL_PATH, DEFAULT_THRESHOLD, SpamClassifier, build_pipeline

# Corpora each channel's model learns from. The email model also learns from SMS so that
# casual, personal messages (rare in Enron/SpamAssassin ham) aren't mistaken for spam; the
# SMS model stays SMS-only because the email data makes it flag far more normal texts as
# spam (1.6% vs 0.2% of the held-out legitimate SMS, see models/metrics.json).
TRAINING_SOURCES = {"email": ("enron", "spamassassin", "sms"), "sms": ("sms",)}
# Held-out corpora behind each channel's headline metrics.
PRIMARY_SOURCES = {"email": ("enron", "spamassassin"), "sms": ("sms",)}


def evaluate(y_true: Sequence[int], proba: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, pred, labels=[0, 1]).ravel())
    return {
        "messages": int(len(y_true)),
        "spam": int(y_true.sum()),
        "ham": int(len(y_true) - y_true.sum()),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4) if len(set(y_true)) == 2 else None,
        "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def fit_and_evaluate(
    channel: str, train_set: list[Example], test_set: list[Example], threshold: float
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit one channel's pipeline and evaluate it on the held-out messages of its corpora."""
    train_examples = [ex for ex in train_set if ex.source in TRAINING_SOURCES[channel]]
    test_examples = [ex for ex in test_set if ex.source in TRAINING_SOURCES[channel]]
    if not train_examples or not test_examples:
        raise ValueError(f"No training/test examples for channel {channel!r}")

    pipeline = build_pipeline().fit([ex.text for ex in train_examples], [ex.label for ex in train_examples])
    proba = pipeline.predict_proba([ex.text for ex in test_examples])[:, 1]
    labels = np.array([ex.label for ex in test_examples])
    sources = np.array([ex.source for ex in test_examples])
    primary = np.isin(sources, PRIMARY_SOURCES[channel])
    metrics = {
        "train_messages": len(train_examples),
        "test": evaluate(labels[primary], proba[primary], threshold),
        "test_by_source": {
            source: evaluate(labels[sources == source], proba[sources == source], threshold)
            for source in sorted(set(sources))
        },
    }
    return pipeline, metrics


def train(
    output: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
    test_size: float = 0.2,
    seed: int = 42,
    threshold: float = DEFAULT_THRESHOLD,
    verify: bool = True,
    examples: list[Example] | None = None,
) -> SpamClassifier:
    """Train both channels, save the bundle to ``output`` and the metrics to ``metrics_path``."""
    if examples is None:
        examples = load_examples(data_dir=data_dir, verify=verify)
    examples = deduplicate(examples)
    # One global split (stratified by corpus and label) so no channel ever trains on
    # a message that another channel is evaluated on.
    train_set, test_set = train_test_split(
        examples,
        test_size=test_size,
        random_state=seed,
        stratify=[f"{ex.source}:{ex.label}" for ex in examples],
    )

    models: dict[str, Pipeline] = {}
    metrics: dict[str, Any] = {}
    for channel in CHANNELS:
        started = time.perf_counter()
        models[channel], metrics[channel] = fit_and_evaluate(channel, train_set, test_set, threshold)
        print(
            f"Trained {channel} model on {metrics[channel]['train_messages']} messages "
            f"in {time.perf_counter() - started:.0f}s",
            file=sys.stderr,
        )

    metadata = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "package_version": __version__,
        "sklearn_version": sklearn.__version__,
        "threshold": threshold,
        "test_size": test_size,
        "seed": seed,
        "datasets": {name: {"description": s.description, "url": s.url} for name, s in SOURCES.items()},
        "metrics": metrics,
    }
    classifier = SpamClassifier(models, threshold, metadata)
    classifier.save(output)
    metrics_path = Path(metrics_path or Path(output).with_name("metrics.json"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return classifier


def format_report(metrics: dict[str, Any]) -> str:
    """Plain-text table of the held-out metrics, one row per channel and test corpus."""
    row = "{:<8} {:<23} {:>6} {:>9} {:>7} {:>6} {:>7} {:>7}".format
    lines = [row("channel", "test set", "msgs", "precision", "recall", "f1", "auc", "fpr")]
    for channel, channel_metrics in metrics.items():
        rows = {"all " + "+".join(PRIMARY_SOURCES[channel]): channel_metrics["test"]}
        if len(channel_metrics["test_by_source"]) > 1:
            for source, source_metrics in channel_metrics["test_by_source"].items():
                suffix = "" if source in PRIMARY_SOURCES[channel] else " (fallback)"
                rows[source + suffix] = source_metrics
        for name, m in rows.items():
            auc = "-" if m["roc_auc"] is None else f"{m['roc_auc']:.4f}"
            scores = (f"{m[key]:.3f}" for key in ("precision", "recall", "f1"))
            lines.append(row(channel, name, m["messages"], *scores, auc, f"{m['false_positive_rate']:.4f}"))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the email and SMS spam classifiers.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="where datasets are cached")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH, help="model bundle to write")
    parser.add_argument("--metrics", type=Path, default=None, help="metrics JSON (default: next to the model)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--skip-checksum", action="store_true", help="don't verify dataset SHA-256 checksums")
    args = parser.parse_args(argv)

    classifier = train(
        output=args.output,
        metrics_path=args.metrics,
        data_dir=args.data_dir,
        test_size=args.test_size,
        seed=args.seed,
        threshold=args.threshold,
        verify=not args.skip_checksum,
    )
    print(format_report(classifier.metadata["metrics"]))
    print(f"\nSaved model to {args.output}")


if __name__ == "__main__":
    main()
