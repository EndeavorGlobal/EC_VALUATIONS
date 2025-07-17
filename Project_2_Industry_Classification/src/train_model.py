#!/usr/bin/env python3
"""
train_model.py
--------------
Train a HierarchicalClassifier on an Excel file
and save the fitted model as a .joblib bundle.

Compile with Nuitka, e.g.:
    python -m nuitka train_model.py ^
        --standalone --onefile ^
        --enable-plugin=numpy ^
        --output-dir=nuitka-build
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
import sklearn
import scipy
import joblib
import pandas as pd
from hierarchical_classifier import HierarchicalClassifier


# ────────────────────────── config ──────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

EXPECTED_COLUMNS = ["company", "label", "label2", "description"]


# ────────────────────────── helpers ─────────────────────────────────────────
def sanitize_path(p: str) -> str:
    """
    Strip surrounding quotes that Windows-style drag-and-drop paths often have.
    """
    p = p.strip()
    if len(p) >= 2 and p[0] == p[-1] and p[0] in ('"', "'"):
        return p[1:-1]
    return p


def read_excel(path: str, sheet: str | int) -> pd.DataFrame:
    """
    Read an Excel sheet and validate required columns.
    """
    try:
        df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    except Exception as exc:  
        sys.exit(f"❌ Error reading Excel: {exc}")

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"❌ Missing columns {missing} (need {EXPECTED_COLUMNS})")

    return df


# ────────────────────────── main ────────────────────────────────────────────
def main() -> None: 
    # ―― Input file ――
    excel_path = sanitize_path(input("Excel file path: ").strip())
    if not Path(excel_path).is_file():
        sys.exit(f"❌ File not found: {excel_path}")

    # ―― Sheet ――
    sheet_input = input("Sheet name or index: ").strip()
    sheet: str | int = int(sheet_input) if sheet_input.isdigit() else sheet_input

    df = read_excel(excel_path, sheet)

    # ―― Train ――
    texts = df["description"].astype(str).tolist()
    broad = df["label"].tolist()
    spec = df["label2"].tolist()

    hc = HierarchicalClassifier()
    print("⌛ Fitting model …")
    hc.fit(texts, broad, spec)

    # ―― Output path ――
    out_raw = sanitize_path(input("Output .joblib path (file or directory): ").strip())
    out_path = (
        Path(out_raw) / "hierarchical_classifier.joblib"
        if Path(out_raw).is_dir()
        else Path(out_raw)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ―― Save ――
    try:
        joblib.dump(hc, out_path)
    except Exception as exc:  
        sys.exit(f"❌ Failed writing model: {exc}")

    print(f"✅ Model saved to {out_path}")


if __name__ == "__main__":
    main()
