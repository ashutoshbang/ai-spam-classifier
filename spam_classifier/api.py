"""HTTP API for the spam classifier.

Start the server, then open http://127.0.0.1:8000 (web UI) or http://127.0.0.1:8000/docs (API docs):

    uvicorn spam_classifier.api:app --reload
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import __version__
from .model import SpamClassifier

_INDEX_HTML = Path(__file__).with_name("static") / "index.html"


@lru_cache(maxsize=1)
def get_classifier() -> SpamClassifier:
    threshold = os.environ.get("SPAM_THRESHOLD")
    return SpamClassifier.load(threshold=float(threshold) if threshold else None)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    get_classifier()  # load the model at startup instead of on the first request
    yield


app = FastAPI(
    title="Spam Classifier",
    version=__version__,
    description="Classify emails and SMS messages as **spam** or **ham** (not spam).",
    lifespan=lifespan,
)

Classifier = Annotated[SpamClassifier, Depends(get_classifier)]


class PredictRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=100_000,
        description="SMS text, email body, or a complete raw email (.eml contents).",
    )
    subject: str | None = Field(default=None, max_length=1_000, description="Email subject, if available.")
    channel: Literal["auto", "email", "sms"] = Field(
        default="auto", description="Which model to use; `auto` decides from the message's shape."
    )

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"text": "WINNER!! Claim your £900 prize now. Call 09061701461", "channel": "auto"},
                {"text": "Hi team,\nThe review moved to 3pm.\nThanks", "subject": "Review time", "channel": "email"},
            ]
        }
    )


class PredictResponse(BaseModel):
    label: Literal["spam", "ham"]
    is_spam: bool
    spam_probability: float = Field(ge=0, le=1)
    channel: Literal["email", "sms"]
    top_signals: list[str] = Field(description="Words/phrases that pushed the decision the most.")


class BatchRequest(BaseModel):
    items: list[PredictRequest] = Field(min_length=1, max_length=100)


class BatchResponse(BaseModel):
    results: list[PredictResponse]


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    return _INDEX_HTML.read_text(encoding="utf-8")


@app.get("/health")
def health(classifier: Classifier) -> dict[str, object]:
    return {
        "status": "ok",
        "version": __version__,
        "threshold": classifier.threshold,
        "trained_at": classifier.metadata.get("trained_at"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, classifier: Classifier) -> dict[str, object]:
    return classifier.predict(request.text, request.subject, request.channel).to_dict()


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest, classifier: Classifier) -> dict[str, object]:
    results = classifier.predict_many(item.model_dump() for item in request.items)
    return {"results": [result.to_dict() for result in results]}
