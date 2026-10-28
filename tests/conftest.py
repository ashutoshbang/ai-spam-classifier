import pytest

from spam_classifier.model import SpamClassifier


@pytest.fixture(scope="session")
def classifier() -> SpamClassifier:
    return SpamClassifier.load()
