#!/usr/bin/env python3
"""
Note: Still getting chained assignment error in EXE but you can ignore it, just a warning.

predict_model.py · v2025-07-08-gini-2level-fix3

Adds a confidence / Gini-impurity metric that combines the ML classifier
and any LLM “second opinions”, **now with explicit Broad + Sub predictions**.

───────────────
Metric design (unchanged)
───────────────
• 50 % of the probability mass is always assigned to the ML model's sub-class.
• The remaining 50 % is split evenly among the LLM models that were run.
• For each row we build a weight vector over *sub* classes, then
      confidence_score = weight on the ML sub-class  ∈ [0.50, 1.00]
      gini_impurity   = 1 - Σ w²                      ∈ [0.00, 0.50]
"""
from __future__ import annotations

# —————————————————— stdlib & third‑party ——————————————————
import os, re, sys, warnings, joblib, pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
from packaging.version import Version
from tqdm import tqdm

import openai
from openai import OpenAI
import google.generativeai as genai

# ── silence pandas chained‑assignment warnings (optional) ────────────────
warnings.filterwarnings("ignore", category=FutureWarning, message=".*ChainedAssignmentError.*")
# Disable runtime chained‑assignment checks as well
pd.options.mode.chained_assignment = None  

# —————————————————— helpers ——————————————————

def sanitize_path(p: str) -> str:
    """Allow paths with or without quotes."""
    p = p.strip()
    return p[1:-1] if len(p) >= 2 and p[0] == p[-1] and p[0] in ("'", '"') else p


# ─────────────── prompt helpers ────────────────

def prompt_two_level(taxonomy: Dict[str, List[str]], desc: str) -> str:
    """Return a prompt that forces the model to answer `<Broad> | <Sub>`."""
    lines = [
        "You are a classification assistant.",
        "First choose EXACTLY ONE *broad* industry from the list below.",
        "Then choose EXACTLY ONE *sub-industry* that belongs to that broad.",
        "Respond in ONE line, exactly as:  <Broad> | <Sub>",
        "",
    ]
    for b, subs in taxonomy.items():
        lines.append(f"• {b}: {', '.join(subs)}")
    lines.append("")
    lines.append(f'Description: "{desc}"')
    return "\n".join(lines)


def split_prediction(txt: str) -> tuple[str, str]:
    """Parse `<Broad> | <Sub>` → (broad, sub). If only one token, return ("", sub)."""
    if "|" in txt:
        broad, sub = (p.strip() for p in txt.split("|", 1))
        return broad, sub
    return "", txt.strip()


# ─────────────── LLM back‑ends ────────────────
_O_SERIES = re.compile(r"^o\d", re.I)
_RESP_CLIENT: Optional[OpenAI] = None  # lazy init for Responses API


def classify_openai(model: str, prompt: str) -> str:
    global _RESP_CLIENT
    if _O_SERIES.match(model):  # «responses» API (o‑series)
        if _RESP_CLIENT is None:
            _RESP_CLIENT = OpenAI(api_key=openai.api_key)
        resp = _RESP_CLIENT.responses.create(model=model, input=prompt)
        return getattr(resp, "output_text", "").strip()

    # Chat Completions API
    resp = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful classifier."},
            {"role": "user",   "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


def classify_gemini(model_id: str, prompt: str) -> str:
    resp = genai.GenerativeModel(model_id).generate_content(prompt)
    if getattr(resp, "text", None):       # gemini‑2.x
        return resp.text.strip()
    try:                                    # gemini‑1.x fallback
        parts = resp.candidates[0].content.parts
        return "\n".join(p.text for p in parts if getattr(p, "text", None)).strip()
    except Exception:
        return ""


# ─────────────── model discovery ────────────────
_BANNED = ("audio", "tts", "vision", "image", "embedding", "embed", "instruct")


def newest_openai_chat_models(n: int = 100) -> List[str]:
    models = [
        m for m in openai.models.list().data
        if ((m.id.startswith("gpt-") or _O_SERIES.match(m.id))
            and not any(b in m.id for b in _BANNED))
    ]
    models.sort(key=lambda m: m.created, reverse=True)
    return [m.id for m in models[:n]]

_GEM_VER = re.compile(r"gemini[-_](\d+(?:\.\d+)*)(?:[-_].+)?", re.I)


def _gem_version(name: str) -> Version:
    m = _GEM_VER.search(name)
    return Version(m.group(1)) if m else Version("0")


def newest_gemini_chat_models(n: int = 100) -> List[str]:
    names = [
        m.name.split("/")[-1] for m in genai.list_models()
        if m.name.lower().startswith("models/gemini")
        and "generateContent" in (m.supported_generation_methods or [])
    ]
    return sorted(set(names), key=_gem_version, reverse=True)[:n]


def choose_models(label: str, models: List[str]) -> List[str]:
    if not models:
        print(f"\n(No {label} chat models available or key missing.)")
        return []
    print(f"\nAvailable {label} chat models:")
    for i, m in enumerate(models, 1):
        print(f"{i}) {m}")
    sel = input(f"Select {label} models (comma‑sep indices, blank=skip): ").strip()
    return [
        models[int(idx) - 1]
        for idx in sel.split(",")
        if idx.strip().isdigit() and 1 <= int(idx) <= len(models)
    ]


# ─────────────── confidence / Gini helper ────────────────

def compute_confidence(
    df: pd.DataFrame,
    ml_col: str,
    llm_cols: list[str],
    conf_col: str = "confidence_score",
    gini_col: str = "gini_impurity",
) -> None:
    """Write two columns into *df* without triggering chained-assignment warnings."""
    n_llms = len(llm_cols)
    if n_llms == 0:
        df.loc[:, conf_col] = 1.0
        df.loc[:, gini_col] = 0.0
        return

    base, contrib = 0.5, 0.5 / n_llms
    conf_vals: List[float] = []
    gini_vals: List[float] = []

    for _, row in df.iterrows():
        weights: Dict[str, float] = {row[ml_col]: base}
        for col in llm_cols:
            pred = row[col]
            weights[pred] = weights.get(pred, 0.0) + contrib
        conf_vals.append(weights[row[ml_col]])
        gini_vals.append(1.0 - sum(w ** 2 for w in weights.values()))

    df.loc[:, conf_col] = conf_vals
    df.loc[:, gini_col] = gini_vals


# ── main program ───────────────────────────────────────────────────────────

def main() -> None:
    # ── inputs
    model_path = sanitize_path(input("Path to .joblib model: "))
    if not Path(model_path).is_file():
        sys.exit(f"❌ Model not found: {model_path}")

    excel = sanitize_path(input("Excel file path: "))
    sheet_in = sanitize_path(input("Sheet name or index: "))
    company_col = sanitize_path(input("Company column name: "))
    desc_col = sanitize_path(input("Description column name: "))
    sheet = int(sheet_in) if sheet_in.isdigit() else sheet_in

    try:
        df = pd.read_excel(excel, sheet_name=sheet, engine="openpyxl")
    except Exception as e:
        sys.exit(f"❌ Error reading Excel: {e}")
    for col in (company_col, desc_col):
        if col not in df.columns:
            sys.exit(f"❌ Column '{col}' missing")

    # ── ML baseline
    hc = joblib.load(model_path)
    texts = df[desc_col].astype(str).tolist()
    broad_ml, spec_ml = hc.predict(texts)
    df["broad_pred_ml"], df["spec_pred_ml"] = broad_ml, spec_ml

    # ── derive taxonomy from ML output (or load your own)
    taxonomy: Dict[str, set[str]] = {}
    for b, s in zip(broad_ml, spec_ml):
        taxonomy.setdefault(b, set()).add(s)
    taxonomy_sorted: Dict[str, List[str]] = {b: sorted(subs) for b, subs in taxonomy.items()}

    # ── LLM branch on/off
    if input("Use LLMs? (Y/n): ").strip().lower() not in ("y", "yes"):
        compute_confidence(df, "spec_pred_ml", [])
        return save_and_exit(excel, df)

    # ── API keys + model selection
    openai.api_key = input("Enter OpenAI API key (blank to skip OpenAI): ").strip()
    google_key = input("Enter Google API key (blank to skip Gemini): ").strip()
    if google_key:
        genai.configure(api_key=google_key)

    openai_pick = choose_models("OpenAI", newest_openai_chat_models() if openai.api_key else [])
    gemini_pick = choose_models("Gemini", newest_gemini_chat_models() if google_key else [])

    # ── predictions: build lists first, then assign
    for mdl in openai_pick:
        bcol, scol = f"{mdl}_broad_pred", f"{mdl}_spec_pred"
        broad_list, sub_list = [], []
        print(f"\n▶ OpenAI {mdl}")
        for desc in texts:
            prompt = prompt_two_level(taxonomy_sorted, desc)
            raw = classify_openai(mdl, prompt)
            b, s = split_prediction(raw)
            broad_list.append(b); sub_list.append(s)
        df[bcol] = broad_list
        df[scol] = sub_list

    for mdl in gemini_pick:
        bcol, scol = f"{mdl}_broad_pred", f"{mdl}_spec_pred"
        broad_list, sub_list = [], []
        print(f"\n▶ Gemini {mdl}")
        for desc in texts:
            prompt = prompt_two_level(taxonomy_sorted, desc)
            raw = classify_gemini(mdl, prompt)
            b, s = split_prediction(raw)
            broad_list.append(b); sub_list.append(s)
        df[bcol] = broad_list
        df[scol] = sub_list

    # ── confidence / Gini on *sub* predictions
    llm_spec_cols = [f"{m}_spec_pred" for m in openai_pick + gemini_pick]
    compute_confidence(df, "spec_pred_ml", llm_spec_cols)

    save_and_exit(excel, df)


def save_and_exit(excel_path: str, df: pd.DataFrame) -> None:
    out = Path(excel_path).with_stem(Path(excel_path).stem + "_classified")
    df.to_excel(out, index=False)
    print(f"\n✅ Results saved to {out}")


if __name__ == "__main__":
    main()
