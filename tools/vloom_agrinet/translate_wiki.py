#!/usr/bin/env python3
"""Translate AgriNet wiki.json contents via YUNWU API and output bilingual wiki."""

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

API_BASE = "https://yunwu.ai/v1"
DEFAULT_MODEL = "gemini-3.1-pro-preview"
CONCURRENT_LIMIT = 8
MAX_RETRIES = 3
RETRY_DELAY = 2.0

TRANSLATE_PROMPT = (
    "You are a professional agricultural science translator. "
    "Translate the following English agricultural/pest/disease description into accurate, natural Chinese. "
    "Preserve scientific terminology (pathogen names, species names) in their original form where appropriate, "
    "but provide Chinese explanations for symptoms and descriptions. "
    "Only return the Chinese translation, no extra commentary.\n\n"
    "Text to translate:\n{text}"
)


def load_wiki(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_wiki(path: Path, data: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


async def translate_text(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    text: str,
    sem: asyncio.Semaphore,
) -> Optional[str]:
    if not text or not text.strip():
        return ""

    prompt = TRANSLATE_PROMPT.format(text=text.strip())
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
        "temperature": 0.1,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.post(
                    f"{API_BASE}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120.0,
                )
                resp.raise_for_status()
                data = resp.json()
                choice = data.get("choices", [{}])[0]
                content = choice.get("message", {}).get("content", "")
                return content.strip()
            except Exception as exc:
                logger.warning("Translation attempt %d failed: %s", attempt + 1, exc)
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    logger.error("Translation failed after %d retries", MAX_RETRIES)
                    return None
        return None


async def translate_all(
    wiki: List[Dict[str, Any]],
    api_key: str,
    model: str,
    cache_path: Optional[Path],
) -> List[Dict[str, Any]]:
    cache: Dict[str, str] = {}
    if cache_path and cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            cache = json.load(f)
        logger.info("Loaded %d cached translations", len(cache))

    client = httpx.AsyncClient()
    sem = asyncio.Semaphore(CONCURRENT_LIMIT)

    # Gather all translation tasks
    tasks = []
    task_meta = []  # (item_idx, content_field)

    for item_idx, item in enumerate(wiki):
        for i in range(1, 6):
            field = f"content_{i}"
            text = item.get(field, "")
            if not text:
                continue
            cache_key = f"{item['code']}:{field}"
            if cache_key in cache:
                continue
            task = translate_text(client, api_key, model, text, sem)
            tasks.append(task)
            task_meta.append((item_idx, field, cache_key))

    logger.info("Total translations needed: %d", len(tasks))

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for (item_idx, field, cache_key), result in zip(task_meta, results):
            if isinstance(result, Exception):
                logger.error("Failed to translate %s: %s", cache_key, result)
                continue
            if result is not None:
                cache[cache_key] = result

    await client.aclose()

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    # Apply translations to wiki items
    for item in wiki:
        for i in range(1, 6):
            field = f"content_{i}"
            cache_key = f"{item['code']}:{field}"
            if cache_key in cache:
                item[f"cn_{field}"] = cache[cache_key]

    return wiki


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate AgriNet wiki to bilingual")
    parser.add_argument("--input", type=Path, default=Path("/opt/public/jiangwentao/agriculture/AgriNet-1K/wiki.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/vlm_data/disease_pest/wiki_bilingual.json"))
    parser.add_argument("--cache", type=Path, default=Path("outputs/vlm_data/disease_pest/translate_cache.json"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=os.environ.get("YUNWU_API_KEY", os.environ.get("OPENAI_API_KEY", "")))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=getattr(logging, args.log_level.upper()),
    )

    if not args.api_key:
        raise RuntimeError("YUNWU_API_KEY or OPENAI_API_KEY must be set in environment or passed via --api-key")

    wiki = load_wiki(args.input)
    logger.info("Loaded %d wiki items from %s", len(wiki), args.input)

    start = time.time()
    wiki = asyncio.run(translate_all(wiki, args.api_key, args.model, args.cache))
    elapsed = time.time() - start
    logger.info("Translation completed in %.1f seconds", elapsed)

    save_wiki(args.output, wiki)
    logger.info("Bilingual wiki saved to %s", args.output)

    # Also output pure Chinese wiki for convenience
    cn_wiki = []
    for item in wiki:
        cn_item = {
            "code": item["code"],
            "english_name": item.get("english_name", ""),
            "chinese_name": item.get("chinese_name", ""),
            "contents": [],
        }
        for i in range(1, 6):
            src = item.get(f"source_{i}", "")
            en = item.get(f"content_{i}", "")
            cn = item.get(f"cn_content_{i}", "")
            if en or cn:
                cn_item["contents"].append({
                    "source": src,
                    "en": en,
                    "cn": cn,
                })
        cn_wiki.append(cn_item)

    cn_path = args.output.with_name(args.output.stem + "_structured.json")
    save_wiki(cn_path, cn_wiki)
    logger.info("Structured bilingual wiki saved to %s", cn_path)


if __name__ == "__main__":
    main()
