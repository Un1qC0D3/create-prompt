#!/usr/bin/env python3
    response.raise_for_status()
    data = response.json()
    # HF returns list[dict[str,str]]
    return data[0]["generated_text"] if isinstance(data, list) else str(data)

# ---------------------- KEYWORD / IDEA GENERATOR ---------------------- #

def get_trending_keywords(n: int = MAX_KEYWORDS) -> List[str]:
    """Fetches trending search keywords for Turkey using pytrends"""
    try:
        from pytrends.request import TrendReq

        pt = TrendReq(hl="en-US", tz=180)
        trends_df = pt.trending_searches(pn="turkey")
        keywords = trends_df[0].tolist()
    except Exception:
        # Fallback static keywords
        keywords = [
            "e-commerce marketing",
            "ai productivity",
            "youtube automation",
            "freelance web dev",
            "learn javascript",
        ]
    random.shuffle(keywords)
    return keywords[:n]

# ----------------------------- TEMPLATE -------------------------------- #

TEMPLATE = """[ROL]={role}
[GÖREV]={task}
[BAĞLAM]={context}
[FORMAT]={format}
[KISIT]={constraints}
Örnek: {example}
""".strip()


def build_prompt(keyword: str) -> Dict[str, str]:
    vars_ = {
        "role": "Deneyimli SEO danışmanı",
        "task": "Anahtar kelimeye dayalı uzun biçimli blog yazısı planı üret",
        "context": f"Hedef anahtar kelime: {keyword}",
        "format": "Markdown başlıklar (H2, H3) + madde işaretli alt başlıklar",
        "constraints": "Minimum 1500 kelime, İngilizce, emoji yok",
        "example": "Full guide for beginners",
    }
    prompt_text = TEMPLATE.format(**vars_)
    return {"title": f"Blog Plan Generator – {keyword}", "prompt": prompt_text}

# ------------------------------ MAIN ----------------------------------- #

def main() -> None:
    keywords = get_trending_keywords()
    prompts = [build_prompt(k) for k in keywords]

    # Basic quality check – verify each prompt returns some content
    good_prompts = []
    for item in prompts:
        try:
            sample = ask_llm(item["prompt"])
            if len(sample.split()) >= 50:  # crude filter
                item["sample"] = sample[:500] + "..."
                good_prompts.append(item)
        except Exception as exc:
            print("Prompt failed quality check:", exc)

    if not good_prompts:
        print("No valid prompts generated – exiting")
        return

    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outfile = OUTPUT_DIR / f"prompts_{timestamp}.json"
    with outfile.open("w", encoding="utf-8") as fp:
        json.dump(good_prompts, fp, ensure_ascii=False, indent=2)

    print(f"Saved {len(good_prompts)} prompts to {outfile}")

if __name__ == "__main__":
    main()
