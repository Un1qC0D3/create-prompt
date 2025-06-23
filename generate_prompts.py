#!/usr/bin/env python3
"""
Auto-generate prompt ideas and save them to outputs/ as JSON.

Bu sürümde hiç dış kütüphane istemcisi (huggingface_hub vb.) 
kullanmayıp, doğrudan REST API’ya post ediyoruz.
"""

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

# Model endpoint
HF_API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
# Token’ı GH Secrets olarak eklediğin HF_TOKEN’dan al
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable is missing!")

# Başlıklar
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

MAX_KEYWORDS = 5
OUTPUT_DIR = pathlib.Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ask_llm(prompt: str, temperature: float = 0.7) -> str:
    """
    Doğrudan REST API çağrısı yapar ve dönen metni alır.
    """
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": temperature,
            "max_new_tokens": 512
        },
        # bazen cache yüzünden eski dönüş yoksa:
        "options": {"use_cache": False}
    }
    resp = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # Çıktı birkaç farklı formatta olabilir:
    # 1) { "generated_text": "..." }
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"].strip()

    # 2) [ { "generated_text": "..." } ]
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()

    # 3) { "outputs": ["..."] }
    if isinstance(data, dict) and "outputs" in data and isinstance(data["outputs"], list):
        return data["outputs"][0].strip()

    raise ValueError(f"Unexpected HF response format: {data}")

def get_trending_keywords(n: int = MAX_KEYWORDS) -> List[str]:
    """Pytrends ile Türkiye trendleri, hata olursa sabit liste döner."""
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
    vars_: Dict[str, str] = {
        "role": "Deneyimli SEO danışmanı",
        "task": "Anahtar kelimeye dayalı uzun biçimli blog yazısı planı üret",
        "context": f"Hedef anahtar kelime: {keyword}",
        "format": "Markdown başlıkları (H2, H3) + madde işaretli alt başlıklar",
        "constraints": "Min 1500 kelime, İngilizce, emoji yok",
        "example": "Full guide for beginners",
    }
    return {
        "title": f"Blog Plan Generator – {keyword}",
        "prompt": TEMPLATE.format(**vars_),
    }

# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main() -> None:
    keywords = get_trending_keywords()
    prompts = [build_prompt(k) for k in keywords]

    kept: List[Dict[str, str]] = []
    for item in prompts:
        try:
            out = ask_llm(item["prompt"], temperature=0.3)
            if len(out.split()) >= 50 and "error" not in out.lower():
                item["sample"] = out[:500] + "…"
                kept.append(item)
        except Exception as e:
            print("Prompt check failed:", e)

    if not kept:
        print("No valid prompts generated. Exiting.")
        return

    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outfile = OUTPUT_DIR / f"prompts_{ts}.json"
    with outfile.open("w", encoding="utf-8") as fp:
        json.dump(kept, fp, ensure_ascii=False, indent=2)

    print(f"Saved {len(kept)} prompts → {outfile}")

if __name__ == "__main__":
    main()
