"""Собирает notebooks/01_eda.ipynb из notebooks/eda_source.py (формат ячеек '# %%')."""

import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "eda_source.py"
DST = ROOT / "notebooks" / "01_eda.ipynb"


def build() -> None:
    text = SRC.read_text(encoding="utf-8")
    parts = re.split(r"^# %%(.*)$", text, flags=re.M)
    cells = []
    for header, body in zip(parts[1::2], parts[2::2], strict=True):
        body = body.strip("\n")
        if "[markdown]" in header:
            lines = []
            for line in body.splitlines():
                if line.startswith("# "):
                    lines.append(line[2:])
                elif line.strip() == "#":
                    lines.append("")
                else:
                    lines.append(line)
            cells.append(nbf.v4.new_markdown_cell("\n".join(lines)))
        else:
            cells.append(nbf.v4.new_code_cell(body))
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nbf.write(nb, DST)
    print(f"Собрано ячеек: {len(cells)} -> {DST}")


if __name__ == "__main__":
    build()
