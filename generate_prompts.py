#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import random
from typing import Dict, List

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_MODEL_URL = (
    "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
)
MAX_KEYWORDS = 5
OUTPUT_DIR = pathlib.Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ask_llm(prompt: str, temperature: float = 0.7) -> str:
    """Send `prompt` to the Inference API and return generated text."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is missing – did you set it as a GitHub secret?"
        )

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    payload: Dict = {
        "inputs": prompt,
        "parameters": {"temperature": temperature, "max_new_tokens": 512},
    }

    res = requests.post(HF_MODEL_URL, headers=headers, json=payload, timeout=60)
    res.raise_for_status()
    data = res.json()

    # Güncel Hugging Face API çıktısını kontrol et
    # 1. outputs -> liste ise
    if isinstance(data, dict) and "outputs" in data and isinstance(data["outputs"], list):
        return data["outputs"][0].strip()
    # 2. generated_text -> direkt string ise
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"].strip()
    # 3. eski format: liste ve ilk elemanında generated_text
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    # 4. hata
    raise ValueError(f"Unexpected HF API response: {data}")

def get_trending_keywords(n: int = MAX_KEYWORDS) -> List[str]:
    """Fetch trending searches for Turkey via *pytrends*. Fallback to static list."""
    try:
        from pytrends.request import TrendReq
        pt = TrendReq(hl="en-US", tz=180)
        df = pt.trending_searches(pn="turkey")
        kw_list: List[str] = df[0].tolist()
    except Exception:
        kw_list = [
            "ai music generators",
            "freelance web dev",
            "javascript interview",
            "open source llm",
            "passive income ideas",
        ]
    random.shuffle(kw_list)
    return kw_list[:n]

TEMPLATE = (
    "[ROL]={role}\n"
    "[GÖREV]={task}\n"
    "[BAĞLAM]={context}\n"
    "[FORMAT]={format}\n"
    "[KISIT]={constraints}\n"
    "Örnek: {example}\n"
)

def build_prompt(keyword: str) -> Dict[str, str]:
    """Return a dict with title + prompt text for a given keyword."""
    variables: Dict[str, str] = {
        "role": "Deneyimli SEO danışmanı",
        "task": "Anahtar kelimeye dayalı uzun biçimli blog yazısı planı üret",
        "context": f"Hedef anahtar kelime: {keyword}",
        "format": "Markdown başlıkları (H2, H3) + madde işaretli alt başlıklar",
        "constraints": "Min 1500 kelime, İngilizce, emoji yok",
        "example": "Full guide for beginners",
    }
    prompt_text = TEMPLATE.format(**variables)
    return {
        "title": f"Blog Plan Generator – {keyword}",
        "prompt": prompt_text,
    }

# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry-point when script is executed."""
    keywords = get_trending_keywords()
    prompts: List[Dict[str, str]] = [build_prompt(kw) for kw in keywords]

    kept: List[Dict[str, str]] = []
    for item in prompts:
        try:
            sample = ask_llm(item["prompt"], temperature=0.3)
            if len(sample.split()) >= 50 and "error" not in sample.lower():
                item["sample"] = sample[:500] + "…"
                kept.append(item)
        except Exception as exc:
            print("Prompt check failed:", exc)

    if not kept:
        print("No valid prompts generated. Exiting.")
        return

    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outfile = OUTPUT_DIR / f"prompts_{timestamp}.json"
    with outfile.open("w", encoding="utf-8") as fp:
        json.dump(kept, fp, ensure_ascii=False, indent=2)

    print(f"Saved {len(kept)} prompts → {outfile}")

if __name__ == "__main__":
    main()
