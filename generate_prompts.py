#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import random
from typing import Dict, List

from huggingface_hub import InferenceApi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Model kimliğini resmen InferenceApi'ye ileteceğiz.
HF_REPO_ID = "tiiuae/falcon-7b-instruct"
# HF_TOKEN mutlaka read erişimine sahip olmalı
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable is missing!")

# InferenceApi istemcisini başlat
inference = InferenceApi(repo_id=HF_REPO_ID, token=HF_TOKEN)  # :contentReference[oaicite:0]{index=0}

MAX_KEYWORDS = 5
OUTPUT_DIR = pathlib.Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ask_llm(prompt: str, temperature: float = 0.7) -> str:
    """
    `inference` üzerinden prompt gönderir, dönen yanıtı çeker.
    - prompt: string
    - temperature: float
    """
    # Hugging Face API, OpenAI-benzeri parametreler alıyor
    res = inference(
        inputs=prompt,
        parameters={"temperature": temperature, "max_new_tokens": 512},
        # Eğer model gRPC yerine REST kullanıyorsa base_url ayarlanabilir
    )
    # Yanıt dict veya list olabilir; genelde dict["generated_text"]
    if isinstance(res, dict) and "generated_text" in res:
        return res["generated_text"].strip()
    if isinstance(res, dict) and "outputs" in res:
        return res["outputs"][0].strip()
    if isinstance(res, list) and "generated_text" in res[0]:
        return res[0]["generated_text"].strip()
    raise ValueError(f"Unexpected HF response: {res}")

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
    """Her anahtar kelime için title + prompt metni hazırlar."""
    variables: Dict[str, str] = {
        "role": "Deneyimli SEO danışmanı",
        "task": "Anahtar kelimeye dayalı uzun biçimli blog yazısı planı üret",
        "context": f"Hedef anahtar kelime: {keyword}",
        "format": "Markdown başlıkları (H2, H3) + madde işaretli alt başlıklar",
        "constraints": "Min 1500 kelime, İngilizce, emoji yok",
        "example": "Full guide for beginners",
    }
    text = TEMPLATE.format(**variables)
    return {"title": f"Blog Plan Generator – {keyword}", "prompt": text}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    keywords = get_trending_keywords()
    prompts = [build_prompt(k) for k in keywords]

    kept: List[Dict[str, str]] = []
    for itm in prompts:
        try:
            out = ask_llm(itm["prompt"], temperature=0.3)
            # Basit filtre: en az 50 kelime, error yok
            if len(out.split()) >= 50 and "error" not in out.lower():
                itm["sample"] = out[:500] + "…"
                kept.append(itm)
        except Exception as e:
            print("Prompt check failed:", e)

    if not kept:
        print("No valid prompts generated. Exiting.")
        return

    fn = f"prompts_{dt.datetime.utcnow():%Y%m%d_%H%M%S}.json"
    outp = OUTPUT_DIR / fn
    with outp.open("w", encoding="utf-8") as fp:
        json.dump(kept, fp, ensure_ascii=False, indent=2)

    print(f"Saved {len(kept)} prompts → {outp}")

if __name__ == "__main__":
    main()
