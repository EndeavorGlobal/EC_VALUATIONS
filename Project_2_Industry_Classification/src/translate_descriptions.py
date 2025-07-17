#!/usr/bin/env python3
"""
• Prompts for the input .xlsx file when missing.
• Output name defaults to <input>_translated.xlsx in the same directory.
• Supports an optional --sheet flag; otherwise asks interactively.
• Safe to bundle with PyInstaller --onefile.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm


# ───────────────────────────── helper utilities ──
def sanitize_path(p: str | None) -> str:
    """Strip surrounding quotes (Windows drag-and-drop) and whitespace."""
    if not p:
        return ""
    p = p.strip()
    return p[1:-1] if len(p) >= 2 and p[0] == p[-1] and p[0] in ("'", '"') else p


def prompt_if_empty(value: str | None, prompt_text: str) -> str:
    """Return *value* or ask the user."""
    return value or input(prompt_text).strip()


def derive_out_path(in_path: str) -> Path:
    """Return <dir>/<stem>_translated.xlsx."""
    p = Path(in_path)
    return p.with_stem(p.stem + "_translated")


# ───────────────────────────── core function ──
def translate_xlsx(
    in_file: str,
    out_file: str | None = None,
    sheet: str | int = 0,
    src_col: str = "description",
    tgt_col: str = "description_en",
) -> None:
    df = pd.read_excel(in_file, sheet_name=sheet, engine="openpyxl")
    if src_col not in df.columns:
        raise ValueError(f"Column '{src_col}' not found in sheet {sheet}")

    translator = GoogleTranslator(source="auto", target="en")
    df[tgt_col] = ""

    for i, text in enumerate(tqdm(df[src_col].fillna("").astype(str), desc="Translating")):
        if not text.strip():
            continue
        try:
            df.at[i, tgt_col] = translator.translate(text)
        except Exception:
            df.at[i, tgt_col] = ""

    out_path = Path(out_file) if out_file else derive_out_path(in_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as writer:
        df.to_excel(
            writer,
            sheet_name=sheet if isinstance(sheet, str) else "Sheet1",
            index=False,
        )
    print(f"✅ Translated descriptions written to {out_path}")


# ───────────────────────────── main entry ──
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate an .xlsx description column to English."
    )
    parser.add_argument("input_xlsx", nargs="?", help="Path to the input .xlsx file")
    parser.add_argument(
        "output_xlsx",
        nargs="?",
        help="Optional output path (defaults to <input>_translated.xlsx)",
    )
    parser.add_argument("--sheet", default=None, help="Sheet name or index (default: 0)")
    parser.add_argument("--src-col", default="description", help="Source column name")
    parser.add_argument("--tgt-col", default="description_en", help="Target column name")
    args = parser.parse_args()

    # ── interactive fallbacks ───────────────────────────────
    input_xlsx = sanitize_path(
        prompt_if_empty(args.input_xlsx, "Enter path to input .xlsx file: ")
    )
    output_xlsx = sanitize_path(args.output_xlsx) if args.output_xlsx else None

    sheet_raw = args.sheet or input("Sheet name or index [0]: ").strip() or "0"
    sheet = int(sheet_raw) if sheet_raw.isdigit() else sheet_raw

    translate_xlsx(
        in_file=input_xlsx,
        out_file=output_xlsx,
        sheet=sheet,
        src_col=args.src_col,
        tgt_col=args.tgt_col,
    )


if __name__ == "__main__":
    main()
