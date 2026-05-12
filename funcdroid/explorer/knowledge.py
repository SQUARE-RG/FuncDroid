import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer


# =========================
# 配置
# =========================

KB_ROOT = "C:\\Users\\23314\\Desktop\\FuncDroid_new\\knowledge_base"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBED_CACHE = "kb_embeddings.pkl"
SCORE_THRESHOLD = 0.3   


# =========================
# Knowledge Base Retriever
# =========================

class KnowledgeBaseRetriever:
    def __init__(self, kb_root: str = KB_ROOT):
        self.kb_root = Path(kb_root)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.kb: List[Dict] = []

        if Path(EMBED_CACHE).exists():
            self._load_cache()
        else:
            self._build_index()
            self._save_cache()

    # -------- KB 加载 --------

    def _load_kb(self) -> List[Dict]:
        kb = []
        for case_dir in self.kb_root.iterdir():
            if not case_dir.is_dir():
                continue

            desc_path = case_dir / "description.json"
            if not desc_path.exists():
                continue

            with open(desc_path, "r", encoding="utf-8") as f:
                desc = json.load(f)

            kb.append({
                "id": int(case_dir.name),
                "action": desc.get("action", ""),
                "function": desc.get("function", ""),
                "path": case_dir
            })
        return kb

    # -------- 向量构建 --------

    @staticmethod
    def _build_case_text(item: Dict) -> str:
        return (
            f"Action: {item['action']}\n"
            f"Function: {item['function']}"
        )

    def _embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)

    def _build_index(self):
        print("[KB] Building embeddings...")
        self.kb = self._load_kb()

        for item in self.kb:
            text = self._build_case_text(item)
            item["embedding"] = self._embed(text)

        print(f"[KB] Built {len(self.kb)} cases.")

    # -------- 缓存 --------

    def _save_cache(self):
        with open(EMBED_CACHE, "wb") as f:
            pickle.dump(self.kb, f)
        print(f"[KB] Saved embedding cache to {EMBED_CACHE}")

    def _load_cache(self):
        with open(EMBED_CACHE, "rb") as f:
            self.kb = pickle.load(f)
        print(f"[KB] Loaded embedding cache from {EMBED_CACHE}")

    # -------- 检索 --------

    def retrieve(self, query: str, topk: int = 2) -> List[Dict]:
        if not self.kb:
            return []

        q_emb = self._embed(query)
        scored = []

        for item in self.kb:
            score = float(np.dot(q_emb, item["embedding"]))
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, item in scored[:topk]:
            if score < SCORE_THRESHOLD:
                continue
            results.append({
                "score": score,
                "id": item["id"],
                "path": item["path"],
                "action": item["action"],
                "function": item["function"]
            })

        return results
