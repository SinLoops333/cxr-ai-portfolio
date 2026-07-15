"""Assemble a small, license-safe RAG corpus for the CXR Copilot.

Two open sources:
    1. NCBI Bookshelf StatPearls chapters — public API returns XML/JSON.
    2. PubMed Central Open Access (PMC OA) abstracts for chest-radiography topics via E-utilities.

Writes JSONL to ``data/kb/corpus.jsonl`` with columns:
    id, source, title, text, url

We intentionally cap sizes to keep the KB small (~few thousand passages) so a
laptop-friendly FAISS index builds in minutes.

Usage:
    python data/scripts/build_rag_corpus.py --max-pmc 500 --topics "cardiomegaly" "pneumothorax" ...
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

from common.logging_utils import get_logger
from common.paths import DATA_KB

log = get_logger("data.rag")

DEFAULT_TOPICS = [
    "atelectasis chest x-ray",
    "cardiomegaly chest x-ray",
    "consolidation chest x-ray",
    "pulmonary edema chest x-ray",
    "pleural effusion chest x-ray",
    "emphysema chest x-ray",
    "pulmonary fibrosis chest x-ray",
    "pulmonary infiltrate chest x-ray",
    "lung mass chest x-ray",
    "lung nodule chest x-ray",
    "pleural thickening chest x-ray",
    "pneumonia chest x-ray",
    "pneumothorax chest x-ray",
    "hiatal hernia chest x-ray",
]

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _pmc_search(topic: str, retmax: int) -> list[str]:
    r = requests.get(
        f"{EUTILS}/esearch.fcgi",
        params={
            "db": "pmc",
            "term": f"{topic} AND open access[filter]",
            "retmax": retmax,
            "retmode": "json",
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("esearchresult", {}).get("idlist", [])


def _pmc_summaries(pmc_ids: list[str]) -> list[dict]:
    if not pmc_ids:
        return []
    r = requests.get(
        f"{EUTILS}/esummary.fcgi",
        params={"db": "pmc", "id": ",".join(pmc_ids), "retmode": "json"},
        timeout=30,
    )
    r.raise_for_status()
    payload = r.json().get("result", {})
    out = []
    for pid in pmc_ids:
        s = payload.get(pid)
        if not s or "title" not in s:
            continue
        out.append(
            {
                "id": f"PMC{pid}",
                "source": "pmc",
                "title": s.get("title", ""),
                "text": s.get("title", ""),  # abstract not always present via esummary; keep title
                "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pid}/",
            }
        )
    return out


def _pmc_abstracts(pmc_ids: list[str]) -> dict[str, str]:
    """Fetch abstracts via efetch (XML). Falls back to empty on failure."""
    if not pmc_ids:
        return {}
    try:
        r = requests.get(
            f"{EUTILS}/efetch.fcgi",
            params={"db": "pmc", "id": ",".join(pmc_ids), "rettype": "abstract", "retmode": "text"},
            timeout=45,
        )
        r.raise_for_status()
    except requests.HTTPError:
        return {}
    text = r.text
    # crude split on blank lines; keep every chunk associated by order with input IDs
    chunks = [c.strip() for c in text.split("\n\n\n") if c.strip()]
    return {pid: (chunks[i] if i < len(chunks) else "") for i, pid in enumerate(pmc_ids)}


def _fetch_statpearls_stubs(topics: list[str]) -> list[dict]:
    """Query the Bookshelf via E-utilities (db=books) and return short summaries.

    This gives us pointers into StatPearls; full-text scraping is avoided (licensing).
    """
    out = []
    for topic in topics:
        try:
            r = requests.get(
                f"{EUTILS}/esearch.fcgi",
                params={"db": "books", "term": f"statpearls[book] AND {topic}", "retmax": 3, "retmode": "json"},
                timeout=30,
            )
            r.raise_for_status()
            ids = r.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                continue
            s = requests.get(
                f"{EUTILS}/esummary.fcgi",
                params={"db": "books", "id": ",".join(ids), "retmode": "json"},
                timeout=30,
            ).json().get("result", {})
            for bid in ids:
                b = s.get(bid, {})
                if not b:
                    continue
                out.append(
                    {
                        "id": f"BOOK{bid}",
                        "source": "statpearls",
                        "title": b.get("title", topic),
                        "text": f"{b.get('title', '')}. {b.get('booktitle', '')}",
                        "url": f"https://www.ncbi.nlm.nih.gov/books/{b.get('bookaccession', '')}/",
                    }
                )
            time.sleep(0.34)  # be nice to NCBI (3 req/s)
        except requests.RequestException as e:
            log.warning("statpearls topic '%s' failed: %s", topic, e)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", nargs="*", default=DEFAULT_TOPICS)
    ap.add_argument("--per-topic", type=int, default=25)
    ap.add_argument("--max-pmc", type=int, default=500)
    ap.add_argument("--out", type=Path, default=DATA_KB / "corpus.jsonl")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    docs: list[dict] = []

    log.info("fetching PMC OA summaries for %d topics", len(args.topics))
    all_pmc_ids: list[str] = []
    for topic in args.topics:
        try:
            ids = _pmc_search(topic, retmax=args.per_topic)
            all_pmc_ids.extend(ids)
            time.sleep(0.34)
        except requests.RequestException as e:
            log.warning("pmc search '%s' failed: %s", topic, e)
    all_pmc_ids = list(dict.fromkeys(all_pmc_ids))[: args.max_pmc]
    log.info("total unique PMC ids: %d", len(all_pmc_ids))

    for i in range(0, len(all_pmc_ids), 50):
        batch = all_pmc_ids[i : i + 50]
        summaries = _pmc_summaries(batch)
        abstracts = _pmc_abstracts(batch)
        for s in summaries:
            pid = s["id"].removeprefix("PMC")
            if abstracts.get(pid):
                s["text"] = abstracts[pid][:4000]
            docs.append(s)
        time.sleep(0.34)

    log.info("fetching StatPearls stubs for %d topics", len(args.topics))
    docs.extend(_fetch_statpearls_stubs(args.topics))

    with open(args.out, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    log.info("wrote %s (%d docs)", args.out, len(docs))


if __name__ == "__main__":
    main()
