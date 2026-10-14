"""Generate the CLI reference page from the argparse parser.

Run automatically by mkdocs-gen-files during the build: it writes ``cli.md``
with the ``--help`` output of every ``ifcb-classify`` command, so the page
always matches the code. ``ifcb_classify.cli`` only imports ``argparse`` at
module level, so the docs build needs none of the package's runtime deps.
"""

import argparse
import copy
import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ifcb_classify.cli import build_parser

HELP_WIDTH = 88

INTRO = """# Command-line interface

Run `ifcb-classify <command>` (or `python -m ifcb_classify <command>`). Options
given on the command line override the YAML config; see
[Configuration](configuration.md) for the defaults. The same text is shown by
`ifcb-classify <command> --help`.
"""


def _help_text(parser: argparse.ArgumentParser) -> str:
    """Return the parser's ``--help`` text at a fixed width."""
    fixed = copy.copy(parser)
    fixed.formatter_class = functools.partial(argparse.HelpFormatter, width=HELP_WIDTH)
    return fixed.format_help()


def render(parser: argparse.ArgumentParser) -> str:
    """Return the Markdown page: one section per command with its help text."""
    subparsers = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    summaries = {a.dest: a.help for a in subparsers._choices_actions}
    sections = [
        f"## `{name}`\n\n{summaries.get(name, '')}\n\n```text\n{_help_text(sub).rstrip()}\n```\n"
        for name, sub in subparsers.choices.items()
    ]
    return "\n".join([INTRO, *sections])


# mkdocs-gen-files runs this file with runpy.run_path; tests import render().
if __name__ == "<run_path>":
    import mkdocs_gen_files

    with mkdocs_gen_files.open("cli.md", "w") as fd:
        fd.write(render(build_parser()))
