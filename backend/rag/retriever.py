"""
Lexical retriever (Okapi BM25) — pure Python, no heavy dependencies, so it runs
on Vercel serverless with zero cold-start cost beyond building the index once.

BM25 is a strong, classic retrieval baseline. Upgrade path to *semantic*
retrieval: precompute embeddings offline, commit the vectors, and swap _score()
for cosine similarity against a query embedding from a free embeddings API
(Jina/Cohere). The Retriever interface below stays the same.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Retriever:
    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.toks = [tokenize(d["text"]) for d in docs]
        self.lens = [len(t) for t in self.toks]
        self.avg_len = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        self.tf = [Counter(t) for t in self.toks]
        df: Counter = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(docs)
        # BM25 idf with +1 smoothing (always positive)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5))
                    for w, c in df.items()}

    def _score(self, q_tokens: list[str], i: int) -> float:
        tf, dl = self.tf[i], self.lens[i]
        s = 0.0
        for w in q_tokens:
            f = tf.get(w, 0)
            if not f:
                continue
            idf = self.idf.get(w, 0.0)
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1))
            s += idf * (f * (self.k1 + 1)) / denom
        return s

    def search(self, query: str, k: int = 4, min_score: float = 0.1) -> list[dict]:
        q = tokenize(query)
        if not q:
            return []
        scored = [(self._score(q, i), i) for i in range(len(self.docs))]
        scored.sort(reverse=True)
        out = []
        for sc, i in scored[:k]:
            if sc <= min_score:
                break
            out.append({"id": self.docs[i]["id"], "text": self.docs[i]["text"],
                        "score": round(sc, 3)})
        return out


def build_retriever(tm) -> Retriever:
    from backend.rag.corpus import build_corpus
    return Retriever(build_corpus(tm))


if __name__ == "__main__":
    from backend.models.match_simulator import build_default_simulator
    r = build_retriever(build_default_simulator(2000).tm)
    for q in ["why does a team win more corners",
              "how should I size a bet with good odds",
              "how good is Brazil and their form",
              "does playing at home matter in the world cup"]:
        hits = r.search(q, k=2)
        print(f"\nQ: {q}")
        for h in hits:
            print(f"  [{h['score']}] {h['id']}: {h['text'][:80]}")
