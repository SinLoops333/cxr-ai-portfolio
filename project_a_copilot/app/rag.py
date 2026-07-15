"""FAISS-backed retriever over the PMC OA + StatPearls corpus.

Two operations:
    - ``--build``: chunk + embed corpus.jsonl -> faiss.index + meta.jsonl
    - default runtime: ``FaissRetriever`` class loaded by the pipeline

Uses sentence-transformers by default (light, offline-capable). Swap to MedCPT
(``ncbi/MedCPT-Query-Encoder``) via config for better clinical recall.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.logging_utils import get_logger

log = get_logger("copilot.rag")


@dataclass
class RetrievedPassage:
    id: str
    title: str
    text: str
    url: str
    score: float


def _chunk(text: str, size: int, overlap: int) -> list[str]:
    if not text:
        return []
    if len(text) <= size:
        return [text]
    out = []
    i = 0
    while i < len(text):
        out.append(text[i : i + size])
        if i + size >= len(text):
            break
        i += max(1, size - overlap)
    return out


def build_index(corpus_path: Path, index_path: Path, meta_path: Path, embedder: str, chunk_chars: int, chunk_overlap: int) -> None:
    import faiss
    from sentence_transformers import SentenceTransformer

    log.info("loading embedder %s", embedder)
    model = SentenceTransformer(embedder)
    dim = model.get_sentence_embedding_dimension()

    meta: list[dict] = []
    embeddings: list[np.ndarray] = []
    with open(corpus_path) as f:
        for line in f:
            d = json.loads(line)
            for i, chunk in enumerate(_chunk(d.get("text", ""), chunk_chars, chunk_overlap)):
                meta.append({"id": f"{d['id']}::{i}", "parent_id": d["id"], "title": d.get("title", ""), "text": chunk, "url": d.get("url", ""), "source": d.get("source", "")})
    log.info("embedding %d chunks", len(meta))
    texts = [m["text"] for m in meta]
    if texts:
        embs = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        embeddings.append(np.asarray(embs, dtype=np.float32))

    all_embs = np.concatenate(embeddings, 0) if embeddings else np.zeros((0, dim), dtype=np.float32)
    index = faiss.IndexFlatIP(dim)
    if len(all_embs):
        index.add(all_embs)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(meta_path, "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    log.info("wrote %s (%d vectors) and %s", index_path, index.ntotal, meta_path)


class FaissRetriever:
    def __init__(self, index_path: Path, meta_path: Path, embedder: str, top_k: int = 5):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(embedder)
        self.index = faiss.read_index(str(index_path))
        self.meta: list[dict] = []
        with open(meta_path) as f:
            for line in f:
                self.meta.append(json.loads(line))
        self.top_k = top_k

    def query(self, text: str, k: int | None = None) -> list[RetrievedPassage]:
        if self.index.ntotal == 0 or not text.strip():
            return []
        emb = self.model.encode([text], normalize_embeddings=True).astype(np.float32)
        k = k or self.top_k
        scores, ids = self.index.search(emb, k)
        out = []
        for score, idx in zip(scores[0], ids[0], strict=False):
            if idx < 0 or idx >= len(self.meta):
                continue
            m = self.meta[idx]
            out.append(RetrievedPassage(id=m["id"], title=m["title"], text=m["text"], url=m["url"], score=float(score)))
        return out


def main() -> None:
    from common.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--query", type=str, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    corpus = Path(cfg.rag.corpus_path)
    index_p = Path(cfg.rag.index_path)
    meta_p = Path(cfg.rag.meta_path)

    if args.build:
        if not corpus.exists():
            raise SystemExit(f"corpus not found: {corpus}. Run data/scripts/build_rag_corpus.py first.")
        build_index(corpus, index_p, meta_p, cfg.rag.embedder, cfg.rag.chunk_chars, cfg.rag.chunk_overlap)

    if args.query:
        r = FaissRetriever(index_p, meta_p, cfg.rag.embedder, cfg.rag.top_k)
        for p in r.query(args.query):
            log.info("[%.3f] %s (%s) -- %s", p.score, p.title, p.url, p.text[:120])


if __name__ == "__main__":
    main()
