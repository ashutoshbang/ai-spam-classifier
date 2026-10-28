"""Spam classification models: one pipeline per channel (``email`` and ``sms``).

SMS and email look very different (length, vocabulary, style). The email model, trained on all
corpora, wrongly flags 1.6% of the legitimate SMS in the test set, while a dedicated SMS model flags
0.2%, so the bundle holds one pipeline per channel and routes between them.
"""

from __future__ import annotations

import os
import re
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .preprocess import (
    MAX_NUMBER_LEN,
    NUMBER_TOKEN_PREFIX,
    looks_like_html,
    looks_like_raw_email,
    normalize_text,
    prepare_text,
)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "spam_classifier.joblib"
CHANNELS = ("email", "sms")
DEFAULT_THRESHOLD = 0.5
# Words plus a few symbols that are strong spam markers ("WIN $$$ !!!", "80% off").
TOKEN_PATTERN = r"(?u)[^\W_]+|[!$£€%]"
# Single-line texts up to this length are treated as SMS when the channel is "auto".
SMS_MAX_CHARS = 300
_BUNDLE_FORMAT = 1
_NUMBER_TOKEN = re.compile(rf"^{NUMBER_TOKEN_PREFIX}(\d+)$")


def _display_token(token: str) -> str:
    """Render internal number tokens readably: ``numlen6`` -> ``<6-digit number>``."""
    match = _NUMBER_TOKEN.match(token)
    if not match:
        return token
    length = int(match.group(1))
    return f"<{length}+ digit number>" if length >= MAX_NUMBER_LEN else f"<{length}-digit number>"


def build_pipeline() -> Pipeline:
    """TF-IDF word (1-2 gram) + character (2-5 gram) features into a logistic regression."""
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    token_pattern=TOKEN_PATTERN,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=200_000,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=2,
                    max_features=100_000,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    classifier = LogisticRegression(C=30.0, class_weight="balanced", max_iter=3000)
    return Pipeline([("features", features), ("clf", classifier)])


def detect_channel(text: str, subject: str | None = None) -> str:
    """Guess whether ``text`` is an email or an SMS-style short message."""
    stripped = text.strip()
    if (
        (subject and subject.strip())
        or looks_like_raw_email(stripped)
        or looks_like_html(stripped)
        or "\n" in stripped
        or len(stripped) > SMS_MAX_CHARS
    ):
        return "email"
    return "sms"


@dataclass(frozen=True)
class Prediction:
    label: str
    is_spam: bool
    spam_probability: float
    channel: str
    top_signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SpamClassifier:
    """Loads the trained bundle and classifies emails and SMS messages."""

    def __init__(
        self,
        models: Mapping[str, Pipeline],
        threshold: float = DEFAULT_THRESHOLD,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        missing = set(CHANNELS) - set(models)
        if missing:
            raise ValueError(f"Model bundle is missing channel(s): {sorted(missing)}")
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.models = dict(models)
        self.threshold = threshold
        self.metadata = dict(metadata or {})
        self._explainers: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None, threshold: float | None = None) -> SpamClassifier:
        path = Path(path or os.environ.get("SPAM_MODEL_PATH") or DEFAULT_MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(f"Model not found at {path}. Train one with: python -m spam_classifier.train")
        bundle = joblib.load(path)  # pickle-based: only load model files you trust
        if not isinstance(bundle, dict) or bundle.get("format") != _BUNDLE_FORMAT:
            raise ValueError(f"Unsupported model bundle format in {path}")
        trained_with = bundle.get("metadata", {}).get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            warnings.warn(
                f"Model was trained with scikit-learn {trained_with} but {sklearn.__version__} is installed; "
                "install the pinned requirements or retrain.",
                stacklevel=2,
            )
        threshold = bundle["threshold"] if threshold is None else threshold
        return cls(bundle["models"], threshold, bundle.get("metadata"))

    def save(self, path: str | os.PathLike[str]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "format": _BUNDLE_FORMAT,
            "models": self.models,
            "threshold": self.threshold,
            "metadata": self.metadata,
        }
        joblib.dump(bundle, path, compress=3)
        return path

    def predict(self, text: str, subject: str | None = None, channel: str = "auto") -> Prediction:
        if not text.strip():
            raise ValueError("text must not be empty")
        if channel == "auto":
            channel = detect_channel(text, subject)
        elif channel not in CHANNELS:
            raise ValueError(f"channel must be 'auto', 'email' or 'sms', got {channel!r}")

        pipeline = self.models[channel]
        features = pipeline.named_steps["features"].transform([prepare_text(text, subject)])
        probability = float(pipeline.named_steps["clf"].predict_proba(features)[0, 1])
        is_spam = probability >= self.threshold
        return Prediction(
            label="spam" if is_spam else "ham",
            is_spam=is_spam,
            spam_probability=round(probability, 4),
            channel=channel,
            top_signals=self._top_signals(channel, features, is_spam),
        )

    def predict_many(self, items: Iterable[Mapping[str, Any]]) -> list[Prediction]:
        """Classify dicts with ``text`` and optional ``subject`` / ``channel`` keys."""
        return [self.predict(item["text"], item.get("subject"), item.get("channel") or "auto") for item in items]

    def _top_signals(self, channel: str, features: Any, is_spam: bool, limit: int = 5) -> list[str]:
        """Word n-grams that pushed the decision the most towards the predicted label."""
        if channel not in self._explainers:
            pipeline = self.models[channel]
            words = pipeline.named_steps["features"].named_transformers["word"].get_feature_names_out()
            self._explainers[channel] = (words, pipeline.named_steps["clf"].coef_[0])
        words, coef = self._explainers[channel]

        row = features.tocsr()
        contributions = row.data * coef[row.indices] * (1 if is_spam else -1)
        signals: list[str] = []
        for i in np.argsort(-contributions):
            if contributions[i] <= 0 or len(signals) == limit:
                break
            column = row.indices[i]
            if column < len(words):  # FeatureUnion puts the word columns before the char columns
                signals.append(" ".join(_display_token(token) for token in words[column].split(" ")))
        return signals
