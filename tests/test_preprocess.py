import time
from email.message import EmailMessage

import pytest

from spam_classifier.model import build_pipeline
from spam_classifier.preprocess import (
    MAX_CHARS,
    compose,
    html_to_text,
    looks_like_html,
    looks_like_raw_email,
    normalize_text,
    parse_email,
    prepare_text,
)


def test_normalize_text_replaces_digit_runs_with_length_tokens():
    assert normalize_text("Call 09061701461 now").split() == ["call", "numlen11", "now"]
    assert normalize_text("Your code is 482913").split()[-1] == "numlen6"
    assert normalize_text("1" * 30).split() == ["numlen12"]


def test_normalize_text_splits_digits_from_letters_and_lowercases():
    assert normalize_text("Claim code KL341").split() == ["claim", "code", "kl", "numlen3"]
    assert normalize_text("ＦＲＥＥ") == "free"  # full-width characters are NFKC-normalised


def test_raw_text_and_enron_style_text_produce_the_same_word_tokens():
    analyzer = build_pipeline().named_steps["features"].named_transformers["word"].build_analyzer()
    raw = "Visit http://6zk.net NOW!!! Only $99.99"
    enron_style = "visit http : / / 6 zk . net now ! ! ! only $ 99 . 99"
    assert analyzer(raw) == analyzer(enron_style)
    assert "numlen2" in analyzer(raw)


def test_html_to_text_keeps_visible_text_and_links():
    markup = (
        "<html><head><style>p { color: red }</style><script>alert(1)</script></head>"
        '<body><p>Hello <b>there</b></p><a href="http://x.example/win">Click</a></body></html>'
    )
    text = html_to_text(markup)
    assert "Hello there" in text
    assert "http://x.example/win" in text
    assert "Click" in text
    assert "alert" not in text and "color" not in text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<p>Hello</p>", True),
        ("<HTML><BODY>Hi</BODY></HTML>", True),
        ("I <3 you, see you at 5 > 4", False),
        ("plain text", False),
    ],
)
def test_looks_like_html(text, expected):
    assert looks_like_html(text) is expected


def test_html_detection_stays_fast_on_unclosed_tags():
    started = time.perf_counter()
    assert not looks_like_html("<p " * 70_000)  # took seconds with a quadratic regex
    assert time.perf_counter() - started < 1.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("From: a@example.com\nTo: b@example.com\nSubject: hi\n\nbody", True),
        ("From nobody Mon Jan  1 00:00:00 2024\nReturn-Path: <a@example.com>\nSubject: x\n\nbody", True),
        ("Received: from mx.example.com\n\tby mail.example.com\nFrom: a@example.com\nSubject: s\n\nbody", True),
        ("\n\nX-Custom: 1\nFrom: a@example.com\nDate: Mon, 1 Jan 2024 00:00:00 +0000\n\nbody", True),
        ("From: a@example.com\nCc:\nX-MS-Has-Attach:\nSubject:hi\n\nbody", True),  # empty values, no space
        ("X_Id: 1\nFrom: a@example.com\nTo: b@example.com\n\nbody", True),
        ("Subject: hi\n\nbody", False),  # a single header is not enough
        ("Note: call me\nTodo: buy milk\n\nok", False),  # header-shaped, but not email headers
        ("Hey, how are you?", False),
        ("From: me\nthis is not a header\nSubject: x", False),
    ],
)
def test_looks_like_raw_email(text, expected):
    assert looks_like_raw_email(text) is expected


def test_parse_email_prefers_plain_text_and_skips_attachments():
    msg = EmailMessage()
    msg["From"] = "alice@example.com"
    msg["To"] = "bob@example.com"
    msg["Subject"] = "Café meeting ☕"
    msg.set_content("Plain body text")
    msg.add_alternative("<p>HTML body text</p>", subtype="html")
    msg.add_attachment("secret attachment text", filename="notes.txt")

    subject, body = parse_email(msg.as_bytes())

    assert subject == "Café meeting ☕"
    assert body == "Plain body text"


def test_parse_email_falls_back_to_html_body():
    msg = EmailMessage()
    msg["Subject"] = "Offer"
    msg.set_content("<html><body><p>Only <b>HTML</b> here</p></body></html>", subtype="html")

    subject, body = parse_email(msg.as_string())

    assert subject == "Offer"
    assert body == "Only HTML here"


def test_prepare_text_parses_raw_emails():
    raw = "From: a@example.com\nTo: b@example.com\nSubject: Lunch\n\nSee you at noon."
    assert prepare_text(raw) == "Lunch\nSee you at noon."
    assert prepare_text(raw, subject="Other") == "Other\nSee you at noon."


def test_prepare_text_ignores_leading_blank_lines_and_crlf():
    raw = "From: a@example.com\nTo: b@example.com\nSubject: Lunch\n\nSee you at noon."
    assert prepare_text("\n" + raw) == "Lunch\nSee you at noon."
    assert prepare_text("\r\n  " + raw.replace("\n", "\r\n")) == "Lunch\nSee you at noon."


def test_prepare_text_strips_html_and_truncates():
    assert prepare_text("<p>Hello <b>world</b></p>") == "Hello world"
    assert prepare_text("Just text", subject="Hi") == "Hi\nJust text"
    assert len(prepare_text("a" * (MAX_CHARS + 100))) == MAX_CHARS


def test_compose_ignores_blank_subject():
    assert compose("   ", "body") == "body"
    assert compose(None, "body") == "body"
