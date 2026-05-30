"""Render the final PDF report.

Strategy: produce all figures + tables as PNGs/CSVs into report/_build/, then
shell out to quarto (preferred) or pandoc to render report/report.qmd to PDF.
The .qmd file is committed in this folder; it references the artefacts by path.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = REPO_ROOT / "report" / "_build"
QMD_PATH = REPO_ROOT / "report" / "report.qmd"


def render(cfg_path: Path) -> None:
    cfg = yaml.safe_load(cfg_path.read_text())
    out_pdf = REPO_ROOT / cfg["report"]["output"]
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if not QMD_PATH.exists():
        raise FileNotFoundError(f"missing {QMD_PATH} — write the report template first")
    # Quarto: `quarto render path/to/report.qmd --to pdf --output ...`
    cmd = ["quarto", "render", str(QMD_PATH), "--to", "pdf", "--output", str(out_pdf)]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    args = ap.parse_args()
    render(args.config)


if __name__ == "__main__":
    main()
