#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import random
from typing import Dict, List

from huggingface_hub import InferenceClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_REPO_ID = "tiiuae/falcon-7b-instruct"
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable is missing!")

client = InferenceClient(token=HF_TOKEN)

MAX_KEYWORDS = 5
OUTPUT_DIR = pathlib.Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ask_llm(prompt: str, temperature: float = 0.7) -> str:
    res = client.text_generation(
        model=HF_REPO_ID,
        prompt=prompt,
        parameters={"temperature": temperature, "max_new_tokens": 512},
    )
    if isinstance(res, dict) and "generated_text" in res:
        return res["generated_text"].strip()
    if isinstance(res, list) and res and "generated_text" in res[0]:
        return res[0]["generated_text"].strip()
    raise ValueError(f"Unexpected response from text_generation: {res}")

def get_trending_keywords(n: int = MAX_KEYWORDS) -> List[str]:
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
    return {"title": f"Blog Plan Generator – {keyword}", "prompt": TEMPLATE.format(**vars_)}

# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main() -> None:
    keywords = get_trending_keywords()
    prompts = [build_prompt(k) for k in keywords]

    kept: List[Dict[str, str]] = []
    for item in prompts:
        try:
            sample = ask_llm(item["prompt"], temperature=0.3)
            if len(sample.split()) >= 50 and "error" not in sample.lower():
                item["sample"] = sample[:500] + "…"
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
