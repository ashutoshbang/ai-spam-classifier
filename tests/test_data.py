import csv
import hashlib
import io
import zipfile

import pytest

from spam_classifier.data import (
    HAM,
    SPAM,
    Example,
    Source,
    deduplicate,
    download,
    load_enron,
    load_examples,
    load_sms,
    load_spamassassin,
)


@pytest.fixture
def remote_file(tmp_path):
    path = tmp_path / "remote.txt"
    path.write_text("hello")
    return path


def make_source(path, sha256=None):
    return Source(
        name="test",
        description="test file",
        url=path.as_uri(),
        sha256=sha256 or hashlib.sha256(path.read_bytes()).hexdigest(),
        filename="cached.txt",
    )


def test_download_verifies_and_caches(remote_file, tmp_path):
    source, cache = make_source(remote_file), tmp_path / "cache"

    path = download(source, cache)
    assert path == cache / "cached.txt"
    assert path.read_text() == "hello"

    remote_file.unlink()  # the second call must not hit the network
    assert download(source, cache) == path


def test_download_rejects_checksum_mismatch(remote_file, tmp_path):
    source, cache = make_source(remote_file, sha256="0" * 64), tmp_path / "cache"

    with pytest.raises(ValueError, match="Checksum mismatch"):
        download(source, cache)
    assert list(cache.iterdir()) == []  # no partial download left behind

    assert download(source, cache, verify=False).read_text() == "hello"


def test_download_rejects_tampered_cache(remote_file, tmp_path):
    source, cache = make_source(remote_file), tmp_path / "cache"
    download(source, cache).write_text("tampered")

    with pytest.raises(ValueError, match="cached file"):
        download(source, cache)


def test_load_sms(tmp_path):
    path = tmp_path / "sms.tsv"
    path.write_text("ham\tSee you soon\nspam\tWIN a FREE prize! Text WIN to 80086\n", encoding="utf-8")

    assert list(load_sms(path)) == [
        Example("See you soon", HAM, "sms"),
        Example("WIN a FREE prize! Text WIN to 80086", SPAM, "sms"),
    ]


def test_load_enron(tmp_path):
    rows = io.StringIO()
    writer = csv.writer(rows)
    writer.writerow(["Message ID", "Subject", "Message", "Spam/Ham", "Date"])
    writer.writerow([0, "meeting", "see you at 3 pm\nthanks", "ham", "2001-01-01"])
    writer.writerow([1, "cheap meds", "buy now !", "spam", "2004-01-01"])
    writer.writerow([2, "", "no subject here", "ham", "2001-01-02"])
    path = tmp_path / "enron.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("enron_spam_data.csv", rows.getvalue())

    assert list(load_enron(path)) == [
        Example("meeting\nsee you at 3 pm\nthanks", HAM, "enron"),
        Example("cheap meds\nbuy now !", SPAM, "enron"),
        Example("no subject here", HAM, "enron"),
    ]


def test_load_spamassassin(tmp_path):
    ham = b"From: a@example.com\nTo: b@example.com\nSubject: Lunch\n\nSee you at noon.\n"
    spam = b"From: x@example.net\nSubject: WIN\nContent-Type: text/html\n\n<p>Claim your <b>prize</b></p>\n"
    path = tmp_path / "spamassassin.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("SpamAssassin-abc/easy_ham/00001.aaa", ham)
        archive.writestr("SpamAssassin-abc/spam_2/00002.bbb", spam)
        archive.writestr("SpamAssassin-abc/spam/", b"")  # directory entry
        archive.writestr("SpamAssassin-abc/README.md", b"not a message")
        archive.writestr("SpamAssassin-abc/other/00003.ccc", ham)

    assert list(load_spamassassin(path)) == [
        Example("Lunch\nSee you at noon.", HAM, "spamassassin"),
        Example("WIN\nClaim your prize", SPAM, "spamassassin"),
    ]


def test_load_examples_uses_cache_and_drops_empty_messages(tmp_path):
    (tmp_path / "sms.tsv").write_text("ham\tHello\nham\t   \nspam\tFree prize\n", encoding="utf-8")

    examples = load_examples(["sms"], data_dir=tmp_path, verify=False)

    assert [(ex.text, ex.label) for ex in examples] == [("Hello", HAM), ("Free prize", SPAM)]


def test_deduplicate():
    examples = [
        Example("Call 12345 now", SPAM, "sms"),
        Example("CALL 54321  NOW", SPAM, "sms"),  # identical after normalisation
        Example("see you later", HAM, "sms"),
        Example("Hello there", HAM, "sms"),
        Example("hello\nTHERE", SPAM, "enron"),  # same text with both labels: dropped
    ]
    assert deduplicate(examples) == [examples[0], examples[2]]
