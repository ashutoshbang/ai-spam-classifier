"""Command-line interface.

Examples:

    python -m spam_classifier "WINNER!! Claim your free prize now"
    python -m spam_classifier --file message.eml
    echo "see you at 5?" | python -m spam_classifier --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .model import SpamClassifier


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m spam_classifier", description="Classify an email or SMS message as spam or ham."
    )
    parser.add_argument("text", nargs="*", help="message text (read from stdin when omitted)")
    parser.add_argument("-f", "--file", type=Path, help="read the message from a file, e.g. a raw .eml email")
    parser.add_argument("-s", "--subject", help="email subject, if not part of the message")
    parser.add_argument("-c", "--channel", choices=["auto", "email", "sms"], default="auto")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    parser.add_argument("--model", type=Path, help="path to a model bundle (default: models/spam_classifier.joblib)")
    args = parser.parse_args(argv)

    if args.file and args.text:
        parser.error("pass either TEXT or --file, not both")
    if args.file:
        text = args.file.read_text(encoding="utf-8", errors="replace")
    elif args.text:
        text = " ".join(args.text)
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.error("no message given: pass TEXT, --file, or pipe text on stdin")
    if not text.strip():
        parser.error("the message is empty")

    try:
        classifier = SpamClassifier.load(args.model)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    result = classifier.predict(text, args.subject, args.channel)
    if args.json:
        print(json.dumps(result.to_dict()))
    else:
        print(f"{result.label.upper()}  ({result.spam_probability:.1%} spam probability, {result.channel} model)")
        if result.top_signals:
            print("signals: " + ", ".join(result.top_signals))
    return 0
