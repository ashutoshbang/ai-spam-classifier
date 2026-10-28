import json
import random

import numpy as np

from spam_classifier.data import HAM, SPAM, Example
from spam_classifier.model import SpamClassifier
from spam_classifier.train import evaluate, format_report, main, train

SPAM_WORDS = "win free prize cash claim urgent offer winner credit loan bonus click".split()
HAM_WORDS = "meeting lunch tomorrow thanks report project dinner family weekend later notes review".split()


def synthetic_examples(per_class=40, seed=0):
    rng = random.Random(seed)
    examples = []
    for source in ("sms", "enron", "spamassassin"):
        for label, words in ((SPAM, SPAM_WORDS), (HAM, HAM_WORDS)):
            for _ in range(per_class):
                text = " ".join(rng.choices(words, k=8))
                if source != "sms":
                    text = f"{rng.choice(words)}\n{text}"
                examples.append(Example(text, label, source))
    return examples


def test_evaluate():
    metrics = evaluate([0, 0, 1, 1], np.array([0.1, 0.6, 0.4, 0.9]), threshold=0.5)
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert metrics["precision"] == metrics["recall"] == metrics["false_positive_rate"] == 0.5
    assert metrics["roc_auc"] == 0.75
    assert evaluate([0, 0], np.array([0.1, 0.2]), threshold=0.5)["roc_auc"] is None


def test_train_end_to_end(tmp_path):
    output = tmp_path / "model.joblib"
    train(output=output, examples=synthetic_examples(), test_size=0.25, seed=1)

    metadata = json.loads((tmp_path / "metrics.json").read_text())
    assert metadata["seed"] == 1
    email, sms = metadata["metrics"]["email"], metadata["metrics"]["sms"]
    # 40 messages per corpus and label, 25% held out from each.
    assert (email["train_messages"], email["test"]["messages"]) == (180, 40)
    assert (sms["train_messages"], sms["test"]["messages"]) == (60, 20)
    assert set(email["test_by_source"]) == {"enron", "spamassassin", "sms"}
    assert email["test"]["f1"] == sms["test"]["f1"] == 1.0  # the synthetic data is trivially separable

    loaded = SpamClassifier.load(output)
    assert loaded.metadata == metadata
    assert loaded.predict("win free cash prize, claim now").label == "spam"
    assert loaded.predict("thanks for lunch, see you at the meeting tomorrow").label == "ham"
    assert loaded.predict("Hi,\nthe project review notes are ready.\nThanks").label == "ham"


def test_main_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("spam_classifier.train.load_examples", lambda **kwargs: synthetic_examples())
    output = tmp_path / "models" / "bundle.joblib"
    metrics_path = tmp_path / "reports" / "metrics.json"

    main(["--output", str(output), "--metrics", str(metrics_path), "--threshold", "0.6"])

    out = capsys.readouterr().out
    assert "precision" in out and f"Saved model to {output}" in out
    assert SpamClassifier.load(output).threshold == 0.6
    assert json.loads(metrics_path.read_text())["threshold"] == 0.6


def test_format_report():
    m = evaluate([0, 1], np.array([0.2, 0.8]), threshold=0.5)
    metrics = {
        "email": {"test": m, "test_by_source": {"enron": m, "spamassassin": m, "sms": m}},
        "sms": {"test": m, "test_by_source": {"sms": m}},
    }
    report = format_report(metrics).splitlines()
    assert len(report) == 6
    assert report[0].split()[:3] == ["channel", "test", "set"]
    assert report[1].startswith("email") and "all enron+spamassassin" in report[1]
    assert "sms (fallback)" in report[4]
    assert report[5].startswith("sms") and "all sms" in report[5]
