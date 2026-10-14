import importlib.util
from pathlib import Path

import pytest

from ifcb_classify.cli import build_parser

GENERATOR = Path(__file__).parent.parent / "docs" / "gen_cli_page.py"
COMMANDS = list(next(a for a in build_parser()._actions if a.dest == "command").choices)


def _render():
    spec = importlib.util.spec_from_file_location("gen_cli_page", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.render(build_parser())


@pytest.mark.parametrize("command", COMMANDS)
def test_page_has_a_section_with_the_help_of_every_command(command):
    page = _render()
    assert f"## `{command}`" in page
    assert f"usage: ifcb-classify {command}" in page


def test_page_shows_option_help_text():
    page = _render()
    assert "Transform to compute the stats with" in page
    assert "Train a classification model" in page


def test_page_does_not_depend_on_terminal_width(monkeypatch):
    monkeypatch.setenv("COLUMNS", "40")
    narrow = _render()
    monkeypatch.setenv("COLUMNS", "200")
    assert _render() == narrow
