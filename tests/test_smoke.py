import pytest

import napoleon
from napoleon.cli import main


def test_version_is_set():
    assert napoleon.__version__


def test_cli_version_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "napoleon" in capsys.readouterr().out
