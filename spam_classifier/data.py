"""Download and load the training corpora.

Datasets are fetched at training time (pinned to fixed commits and verified with
SHA-256) and are never committed to this repository.
"""

from __future__ import annotations

import csv
import hashlib
import io
import shutil
import sys
import urllib.request
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .preprocess import compose, normalize_text, parse_email, prepare_text

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

SPAM, HAM = 1, 0


@dataclass(frozen=True)
class Source:
    name: str
    description: str
    url: str
    sha256: str
    filename: str


SOURCES: dict[str, Source] = {
    "sms": Source(
        name="sms",
        description="UCI SMS Spam Collection (5,574 SMS messages)",
        url=(
            "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/"
            "5339710562cc46c4a32a369d63f1e8751a18a54e/data/sms.tsv"
        ),
        sha256="7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d",
        filename="sms.tsv",
    ),
    "enron": Source(
        name="enron",
        description="Enron-Spam corpus (33,716 emails)",
        url=(
            "https://github.com/MWiechmann/enron_spam_data/raw/"
            "e73c300263040f84b954e969ac06821c31cff47a/enron_spam_data.zip"
        ),
        sha256="e994ad3c09fd7528b397e1f921742da5dd59f19695bc25cdb3617eb3ded17634",
        filename="enron_spam_data.zip",
    ),
    "spamassassin": Source(
        name="spamassassin",
        description="SpamAssassin public corpus (6,046 raw emails)",
        url="https://codeload.github.com/dmitrynogin/SpamAssassin/zip/e9ce73fe5ab4834eaa2abe2fed0a6198b3e9d9dc",
        sha256="50c2c7ca53ae3f58736d50740fe303483fdd3517d45930c149fcfcbe647a5efa",
        filename="spamassassin.zip",
    ),
}

_SPAMASSASSIN_FOLDERS = {"easy_ham": HAM, "easy_ham_2": HAM, "hard_ham": HAM, "spam": SPAM, "spam_2": SPAM}


@dataclass(frozen=True)
class Example:
    text: str
    label: int  # 1 = spam, 0 = ham
    source: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(source: Source, data_dir: Path = DEFAULT_DATA_DIR, verify: bool = True) -> Path:
    """Download ``source`` into ``data_dir`` (cached) and verify its checksum."""
    path = data_dir / source.filename
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        print(f"Downloading {source.description} ...", file=sys.stderr)
        with urllib.request.urlopen(source.url, timeout=300) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out)
        if verify and _sha256(tmp) != source.sha256:
            tmp.unlink()
            raise ValueError(f"Checksum mismatch for {source.url}; refusing to use it.")
        tmp.replace(path)
    elif verify and _sha256(path) != source.sha256:
        raise ValueError(f"Checksum mismatch for cached file {path}; delete it and retry.")
    return path


def load_sms(path: Path) -> Iterator[Example]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            label, _, text = line.rstrip("\n").partition("\t")
            yield Example(prepare_text(text), SPAM if label == "spam" else HAM, "sms")


def load_enron(path: Path) -> Iterator[Example]:
    csv.field_size_limit(2**31 - 1)
    with zipfile.ZipFile(path) as archive, archive.open("enron_spam_data.csv") as raw:
        for row in csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline="")):
            label = SPAM if row["Spam/Ham"] == "spam" else HAM
            yield Example(compose(row["Subject"], row["Message"]), label, "enron")


def load_spamassassin(path: Path) -> Iterator[Example]:
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            parts = name.split("/")
            if len(parts) != 3 or not parts[2] or parts[1] not in _SPAMASSASSIN_FOLDERS:
                continue
            subject, body = parse_email(archive.read(name))
            yield Example(compose(subject, body), _SPAMASSASSIN_FOLDERS[parts[1]], "spamassassin")


_LOADERS = {"sms": load_sms, "enron": load_enron, "spamassassin": load_spamassassin}


def load_examples(
    sources: list[str] | None = None, data_dir: Path = DEFAULT_DATA_DIR, verify: bool = True
) -> list[Example]:
    """Download (if needed) and load the requested sources, dropping empty messages."""
    examples: list[Example] = []
    for name in sources or list(SOURCES):
        path = download(SOURCES[name], data_dir, verify=verify)
        examples.extend(ex for ex in _LOADERS[name](path) if ex.text.strip())
    return examples


def deduplicate(examples: list[Example]) -> list[Example]:
    """Keep the first copy of each message (after normalisation) so duplicates can't leak
    between the train and test splits; messages seen with both labels are dropped."""
    groups: dict[str, list[Example]] = {}
    for ex in examples:
        groups.setdefault(" ".join(normalize_text(ex.text).split()), []).append(ex)
    return [group[0] for group in groups.values() if len({ex.label for ex in group}) == 1]
