import base64

import joblib
import pytest

from spam_classifier.model import SpamClassifier, _display_token, detect_channel
from tests.samples import FIXTURES, HAM_EMAILS, HAM_SMS, SPAM_EMAILS, SPAM_SMS

RAW_EMAIL = "From: a@example.com\nTo: b@example.com\nSubject: hi\n\nHello"


@pytest.mark.parametrize("text", SPAM_SMS)
def test_spam_sms_is_spam(classifier, text):
    result = classifier.predict(text)
    assert (result.label, result.channel) == ("spam", "sms")


@pytest.mark.parametrize("text", HAM_SMS)
def test_normal_sms_is_ham(classifier, text):
    result = classifier.predict(text)
    assert (result.label, result.channel) == ("ham", "sms")


@pytest.mark.parametrize(("subject", "body"), SPAM_EMAILS)
def test_spam_email_is_spam(classifier, subject, body):
    result = classifier.predict(body, subject=subject)
    assert (result.label, result.channel) == ("spam", "email")


@pytest.mark.parametrize(("subject", "body"), HAM_EMAILS)
def test_normal_email_is_ham(classifier, subject, body):
    result = classifier.predict(body, subject=subject)
    assert (result.label, result.channel) == ("ham", "email")


@pytest.mark.parametrize(("filename", "label"), [("spam.eml", "spam"), ("ham.eml", "ham")])
def test_raw_eml_files(classifier, filename, label):
    result = classifier.predict((FIXTURES / filename).read_text(encoding="utf-8"))
    assert (result.label, result.channel) == (label, "email")


def test_raw_email_quirks_do_not_change_the_prediction(classifier):
    subject, body = HAM_EMAILS[0]
    raw = (
        "From: Sarah <sarah@example.com>\nTo: team@example.com\nCc:\nX-MS-Has-Attach:\nSubject:" + subject + "\n"
        "MIME-Version: 1.0\nContent-Type: text/plain; charset=utf-8\nContent-Transfer-Encoding: base64\n\n"
        + base64.encodebytes(body.encode()).decode()
    )
    expected = classifier.predict(body, subject=subject)
    for variant in (raw, "\n" + raw, "\r\n " + raw.replace("\n", "\r\n")):
        assert classifier.predict(variant) == expected


def test_prediction_shape(classifier):
    result = classifier.predict(SPAM_SMS[0])
    data = result.to_dict()
    assert set(data) == {"label", "is_spam", "spam_probability", "channel", "top_signals"}
    assert 0.0 <= data["spam_probability"] <= 1.0
    assert data["is_spam"] is (data["label"] == "spam")
    assert 1 <= len(data["top_signals"]) <= 5
    assert all(isinstance(signal, str) and signal for signal in data["top_signals"])


def test_number_tokens_are_shown_readably(classifier):
    signals = classifier.predict(SPAM_SMS[0]).top_signals
    assert any("digit number>" in signal for signal in signals)
    assert not any("numlen" in signal for signal in signals)


@pytest.mark.parametrize(
    ("token", "shown"),
    [("numlen6", "<6-digit number>"), ("numlen12", "<12+ digit number>"), ("free", "free"), ("numlen", "numlen")],
)
def test_display_token(token, shown):
    assert _display_token(token) == shown


@pytest.mark.parametrize(
    ("text", "subject", "channel"),
    [
        ("Hey, are you free tonight?", None, "sms"),
        ("  Hey, are you free tonight?  \n", None, "sms"),
        ("x" * 300, None, "sms"),
        ("x" * 301, None, "email"),
        ("Hey", "A subject", "email"),
        ("Hey", "   ", "sms"),
        ("line one\nline two", None, "email"),
        ("<p>Hello</p>", None, "email"),
        (RAW_EMAIL, None, "email"),
    ],
)
def test_detect_channel(text, subject, channel):
    assert detect_channel(text, subject) == channel


def test_explicit_channel_overrides_detection(classifier):
    assert classifier.predict(HAM_SMS[0], channel="email").channel == "email"
    assert classifier.predict(HAM_EMAILS[0][1], channel="sms").channel == "sms"


def test_invalid_input_raises(classifier):
    with pytest.raises(ValueError, match="channel"):
        classifier.predict("hello", channel="fax")
    with pytest.raises(ValueError, match="empty"):
        classifier.predict("  \n ")


def test_threshold_controls_the_label(classifier):
    text = "Your OTP for login is 482913. Do not share it with anyone."
    probability = classifier.predict(text).spam_probability
    assert 0.0 < probability < 1.0

    strict = SpamClassifier(classifier.models, threshold=probability / 2)
    lenient = SpamClassifier(classifier.models, threshold=(probability + 1) / 2)
    assert strict.predict(text).label == "spam"
    assert lenient.predict(text).label == "ham"
    assert strict.predict(text).spam_probability == lenient.predict(text).spam_probability == probability


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.5, 1.5])
def test_threshold_must_be_a_probability(classifier, threshold):
    with pytest.raises(ValueError, match="threshold"):
        SpamClassifier(classifier.models, threshold=threshold)


def test_bundle_needs_both_channels(classifier):
    with pytest.raises(ValueError, match="missing"):
        SpamClassifier({"email": classifier.models["email"]})


def test_predict_many(classifier):
    results = classifier.predict_many(
        [{"text": SPAM_SMS[0]}, {"text": HAM_EMAILS[0][1], "subject": HAM_EMAILS[0][0], "channel": None}]
    )
    assert [(r.label, r.channel) for r in results] == [("spam", "sms"), ("ham", "email")]


def test_save_and_load_round_trip(classifier, tmp_path):
    path = classifier.save(tmp_path / "bundle.joblib")
    loaded = SpamClassifier.load(path, threshold=0.7)
    assert loaded.threshold == 0.7
    assert loaded.metadata == classifier.metadata
    assert loaded.predict(SPAM_SMS[1]) == classifier.predict(SPAM_SMS[1])


def test_shipped_model_metadata(classifier):
    assert classifier.threshold == 0.5
    assert set(classifier.metadata["metrics"]) == {"email", "sms"}
    assert classifier.metadata["metrics"]["email"]["test"]["f1"] > 0.95
    assert classifier.metadata["metrics"]["sms"]["test"]["f1"] > 0.9


def test_load_errors(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        SpamClassifier.load(tmp_path / "missing.joblib")

    monkeypatch.setenv("SPAM_MODEL_PATH", str(tmp_path / "from-env.joblib"))
    with pytest.raises(FileNotFoundError, match="from-env"):
        SpamClassifier.load()

    joblib.dump(["not", "a", "bundle"], tmp_path / "bad.joblib")
    with pytest.raises(ValueError, match="format"):
        SpamClassifier.load(tmp_path / "bad.joblib")
