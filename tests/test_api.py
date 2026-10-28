import pytest
from fastapi.testclient import TestClient

from spam_classifier import __version__
from spam_classifier.api import app
from tests.samples import FIXTURES, HAM_EMAILS, HAM_SMS, SPAM_SMS


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Spam Classifier" in response.text


def test_health(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["version"] == __version__
    assert data["threshold"] == 0.5
    assert data["trained_at"]


def test_openapi_docs(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/predict", "/predict/batch", "/health"} <= set(paths)


def test_predict_spam_sms(client):
    response = client.post("/predict", json={"text": SPAM_SMS[0]})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "spam"
    assert data["is_spam"] is True
    assert data["channel"] == "sms"
    assert data["spam_probability"] > 0.5
    assert data["top_signals"]


def test_predict_ham_email(client):
    subject, body = HAM_EMAILS[1]
    data = client.post("/predict", json={"text": body, "subject": subject}).json()
    assert (data["label"], data["channel"]) == ("ham", "email")


def test_predict_raw_email(client):
    raw = (FIXTURES / "spam.eml").read_text(encoding="utf-8")
    data = client.post("/predict", json={"text": raw}).json()
    assert (data["label"], data["channel"]) == ("spam", "email")


def test_predict_with_explicit_channel(client):
    data = client.post("/predict", json={"text": HAM_SMS[0], "channel": "email"}).json()
    assert data["channel"] == "email"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": ""},
        {"text": "   \n"},
        {"text": "hello", "channel": "fax"},
        {"text": "x" * 100_001},
        {"text": "hello", "subject": "s" * 1_001},
        {"text": 42},
    ],
)
def test_predict_rejects_invalid_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_batch(client):
    items = [{"text": SPAM_SMS[1]}, {"text": HAM_SMS[1]}, {"text": HAM_EMAILS[0][1], "subject": HAM_EMAILS[0][0]}]
    response = client.post("/predict/batch", json={"items": items})
    assert response.status_code == 200
    results = response.json()["results"]
    assert [(r["label"], r["channel"]) for r in results] == [("spam", "sms"), ("ham", "sms"), ("ham", "email")]


@pytest.mark.parametrize("count", [0, 101])
def test_batch_size_limits(client, count):
    items = [{"text": "hello"}] * count
    assert client.post("/predict/batch", json={"items": items}).status_code == 422
