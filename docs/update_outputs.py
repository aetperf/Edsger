"""Execute the ```python blocks of Markdown pages, in order, in one namespace per
page, and replace the marker that follows a block with its output, indented by 4
spaces (the style of the Edsger docs).

Markers, on the line after the closing fence (blank lines allowed):
  % output   stdout of the block, then the repr of its last expression
  % error    the block must raise: "ExceptionType: message"
A marker already followed by an indented output keeps being refreshed: the output
is written between the marker and % /output, which are both kept so that
the script can be rerun.

Usage, from the repository root, with edsger installed:

    python docs/update_outputs.py docs/source/quickstart.md docs/source/guide/*.md \\
        docs/source/examples/*.md docs/source/coming_from.md
"""

import ast
import contextlib
import io
import re
import sys
import warnings

import numpy as np
import pandas as pd

pd.set_option("display.width", 88)
np.set_printoptions(linewidth=88)

BLOCK = re.compile(
    r"(```python\n(?P<code>(?:(?!```).)*?)```\n)"
    r"(?P<gap>\s*)"
    r"(?P<marker>% (?P<kind>output|error))\n"
    r"(?P<old>.*?% /output\n|)",
    re.S,
)
ANY_BLOCK = re.compile(r"```python\n(.*?)```\n", re.S)


def run_block(code, namespace, expect_error):
    tree = ast.parse(code)
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(compile(tree, "<doc>", "exec"), namespace)
            if last is not None:
                # the code is the documentation's own example code
                value = eval(compile(last, "<doc>", "eval"), namespace)
                if value is not None:
                    print(repr(value))
    except Exception as exc:  # noqa: BLE001
        if not expect_error:
            raise RuntimeError(f"block failed:\n{code}") from exc
        return f"{type(exc).__name__}: {exc}"
    if expect_error:
        raise RuntimeError(f"block did not raise:\n{code}")
    return out.getvalue().rstrip("\n")


def process(path):
    text = open(path).read()
    namespace = {"__name__": "__doc__"}
    pieces, pos = [], 0
    for match in ANY_BLOCK.finditer(text):
        # code blocks without marker are executed too, for the state
        marked = BLOCK.match(text, match.start())
        if marked is None:
            run_block(match.group(1), namespace, False)
            continue
        output = run_block(marked["code"], namespace, marked["kind"] == "error")
        indented = "\n".join(("    " + line).rstrip() for line in output.split("\n"))
        pieces.append(text[pos : marked.start()])
        pieces.append(
            marked.group(1)
            + "\n"
            + marked["marker"]
            + "\n\n"
            + indented
            + "\n\n% /output\n"
        )
        pos = marked.end()
    pieces.append(text[pos:])
    open(path, "w").write("".join(pieces))
    print(f"{path}: ok")


if __name__ == "__main__":
    for page in sys.argv[1:]:
        process(page)
