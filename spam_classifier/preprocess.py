"""Text preparation shared by training and inference.

The Enron-Spam corpus ships lower-cased and pre-tokenised (``http : / / site . com``,
digits split from letters), while SMS, SpamAssassin and user input are raw text.
Everything is mapped into one canonical form so the same features fire no matter
where a message comes from.
"""

from __future__ import annotations

import email
import email.policy
import re
import unicodedata
from email.message import Message
from html.parser import HTMLParser

# Upper bound on characters fed to the model; keeps latency predictable for huge emails.
MAX_CHARS = 20_000

# Digit runs become length tokens ("482913" -> "numlen6"). The length separates OTP codes,
# prices, years and premium-rate phone numbers without memorising the digits themselves.
NUMBER_TOKEN_PREFIX = "numlen"
MAX_NUMBER_LEN = 12

# RFC 5322 field name (printable ASCII except ":"); values may be empty or start without a space.
_HEADER_LINE = re.compile(r"^([!-9;-~]+):")
_KNOWN_HEADERS = frozenset(
    "from to cc subject date received return-path message-id mime-version content-type "
    "content-transfer-encoding delivered-to reply-to x-mailer sender user-agent".split()
)
# [^<>] (not [^>]) keeps matching linear-time on input full of unclosed tags.
_HTML_TAG = re.compile(r"</?(?:html|body|head|div|p|br|table|tr|td|span|font|a|img|center)\b[^<>]*>", re.I)
_DIGIT_LETTER_BOUNDARY = re.compile(r"(?<=\d)(?=[^\W\d_])|(?<=[^\W\d_])(?=\d)")
_DIGITS = re.compile(r"\d+")
_BLANK_LINES = re.compile(r"\n\s*\n+")


class _HTMLTextExtractor(HTMLParser):
    _SKIP = frozenset({"script", "style"})
    _BLOCK = frozenset({"br", "p", "div", "tr", "li", "table", "h1", "h2", "h3", "h4", "h5", "h6"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self._parts.append("\n")
        elif tag == "a":
            # Link targets are a strong spam signal that plain-text emails expose inline.
            href = dict(attrs).get("href")
            if href:
                self._parts.append(f" {href} ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        return _BLANK_LINES.sub("\n", "".join(self._parts)).strip()


def html_to_text(markup: str) -> str:
    """Strip tags from an HTML document, keeping visible text and link targets."""
    parser = _HTMLTextExtractor()
    try:
        parser.feed(markup)
        parser.close()
    except Exception:  # pragma: no cover - HTMLParser is lenient; this is a last resort
        return re.sub(r"<[^>]+>", " ", markup)
    return parser.text()


def looks_like_html(text: str) -> bool:
    return _HTML_TAG.search(text) is not None


def looks_like_raw_email(text: str) -> bool:
    """True if ``text`` starts with an RFC 822 header block (e.g. a pasted ``.eml`` file)."""
    lines = text.lstrip().splitlines()
    if lines and lines[0].startswith("From "):  # mbox separator line
        lines = lines[1:]
    known = 0
    for line in lines[:200]:
        if not line.strip():
            break
        if line[0] in " \t":  # folded continuation of the previous header
            continue
        match = _HEADER_LINE.match(line)
        if not match:
            return False
        known += match.group(1).lower() in _KNOWN_HEADERS
    return known >= 2


def _decode_part(part: Message) -> str:
    try:
        content = part.get_content()
    except (LookupError, UnicodeError, ValueError, AssertionError):
        payload = part.get_payload(decode=True) or b""
        content = payload.decode("latin-1", errors="replace") if isinstance(payload, bytes) else ""
    return content if isinstance(content, str) else ""


def parse_email(raw: str | bytes) -> tuple[str, str]:
    """Return ``(subject, body)`` from a raw email; HTML-only bodies are converted to text."""
    if isinstance(raw, bytes):
        msg = email.message_from_bytes(raw, policy=email.policy.default)
    else:
        msg = email.message_from_string(raw, policy=email.policy.default)

    try:
        subject = str(msg.get("subject") or "")
    except Exception:  # malformed encoded-word headers
        subject = ""

    plain: list[str] = []
    html: list[str] = []
    try:
        for part in msg.walk():
            if part.is_multipart() or part.get_content_disposition() == "attachment":
                continue
            content_type = part.get_content_type()
            if content_type == "text/plain":
                plain.append(_decode_part(part))
            elif content_type == "text/html":
                html.append(html_to_text(_decode_part(part)))
    except Exception:  # badly broken MIME structure: fall back to the raw payload
        payload = msg.get_payload()
        plain = [payload if isinstance(payload, str) else ""]

    body = "\n".join(p for p in plain if p.strip()) or "\n".join(html)
    return subject.strip(), body.strip()


def compose(subject: str | None, body: str) -> str:
    """Combine an optional subject with a body the same way for every data source."""
    if looks_like_html(body):
        body = html_to_text(body)
    text = f"{subject.strip()}\n{body}" if subject and subject.strip() else body
    return text[:MAX_CHARS]


def prepare_text(text: str, subject: str | None = None) -> str:
    """Turn an SMS, an email body or a full raw email into the text the model sees."""
    if looks_like_raw_email(text):
        # Strip like looks_like_raw_email does: a leading blank line would end the headers early.
        parsed_subject, text = parse_email(text.lstrip())
        subject = subject or parsed_subject
    return compose(subject, text)


def _number_token(match: re.Match[str]) -> str:
    return f" {NUMBER_TOKEN_PREFIX}{min(len(match.group()), MAX_NUMBER_LEN)} "


def normalize_text(text: str) -> str:
    """Canonical form used by the vectoriser (matches the Enron-Spam preprocessing style)."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _DIGIT_LETTER_BOUNDARY.sub(" ", text)
    return _DIGITS.sub(_number_token, text)
