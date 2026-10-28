import io
import json

import pytest

from spam_classifier import cli
from tests.samples import FIXTURES, HAM_SMS, SPAM_SMS


@pytest.fixture
def shared_model(classifier, monkeypatch):
    # Reuse the session's model instead of loading it from disk for every CLI call.
    monkeypatch.setattr(cli.SpamClassifier, "load", lambda path=None: classifier)


@pytest.mark.usefixtures("shared_model")
def test_text_argument(capsys):
    assert cli.main([SPAM_SMS[0]]) == 0
    out = capsys.readouterr().out
    assert out.startswith("SPAM")
    assert "sms model" in out
    assert "signals:" in out


@pytest.mark.usefixtures("shared_model")
def test_file_with_json_output(capsys):
    assert cli.main(["--file", str(FIXTURES / "ham.eml"), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert (data["label"], data["channel"]) == ("ham", "email")


@pytest.mark.usefixtures("shared_model")
def test_reads_stdin(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(HAM_SMS[0]))
    assert cli.main([]) == 0
    assert capsys.readouterr().out.startswith("HAM")


@pytest.mark.usefixtures("shared_model")
def test_subject_and_channel_options(capsys):
    assert cli.main(["See you at 5", "--subject", "Plans", "--channel", "sms", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["channel"] == "sms"


@pytest.mark.parametrize(
    "argv",
    [
        ["some text", "--file", str(FIXTURES / "ham.eml")],
        ["   "],
        ["hello", "--channel", "fax"],
    ],
)
def test_usage_errors(argv):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(argv)
    assert exc_info.value.code == 2


def test_missing_model_is_reported(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["hello", "--model", str(tmp_path / "missing.joblib")])
    assert exc_info.value.code == 1
    assert "Model not found" in capsys.readouterr().err
