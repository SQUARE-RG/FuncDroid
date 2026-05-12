import os
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ENCODINGS_FILE = "bug_vectors.npy"
TEXT_FILE = "bug_texts.pkl"


class BugKnowledgeBase:
    def __init__(self, excel_path):
        self.excel_path = excel_path
        self.model = SentenceTransformer(MODEL_NAME)

        # storage
        self.bug_texts = None        # list[dict]
        self.bug_vectors = None      # numpy array

        # auto load if exists
        if os.path.exists(ENCODINGS_FILE) and os.path.exists(TEXT_FILE):
            print("🔄 Loading cached knowledge base...")
            self.load_cache()
        else:
            print("📘 No cache found. Building knowledge base from Excel...")
            self.build_from_excel()

    # -------------------------------
    # Build knowledge base from Excel
    # -------------------------------
    def build_from_excel(self):
        df = pd.read_excel(self.excel_path, dtype=str, keep_default_na=False)
        df["body"] = df["body"].str.replace("_x000D_", "\n", regex=False)
        df["body"] = df["body"].str.strip()
        self.bug_df = df

        # ensure title/body exist
        df["title"] = df["title"].fillna("")
        df["body"] = df["body"].fillna("")

        # combine title+body
        self.bug_texts = []
        bug_corpus = []

        print("⏳ Encoding bug reports...")
        for i, row in df.iterrows():
            text = row["title"] + "\n" + row["body"]
            self.bug_texts.append({
                "_id": row["_id"],
                "title": row["title"],
                "body": row["body"],
                "raw_text": text
            })
            bug_corpus.append(text)

        # encode all at once
        self.bug_vectors = self.model.encode(bug_corpus, convert_to_tensor=True, show_progress_bar=True)
        self.save_cache()

    # -------------------------------
    # Save & Load Cache
    # -------------------------------
    def save_cache(self):
        np.save(ENCODINGS_FILE, self.bug_vectors.cpu().numpy())
        with open(TEXT_FILE, "wb") as f:
            pickle.dump(self.bug_texts, f)
        print("💾 Cache saved.")

    def load_cache(self):
        print("🔄 Loading cached vectors...")

        # Load numpy vectors
        arr = np.load(ENCODINGS_FILE)

        # Convert to tensor (required for cosine similarity)
        self.bug_vectors = torch.tensor(arr)

        # Load texts
        with open(TEXT_FILE, "rb") as f:
            self.bug_texts = pickle.load(f)

        print("✅ Cache loaded.")

    # -------------------------------
    # Retrieve similar bug reports
    # -------------------------------
    def query(self, query_text, top_k=5):
        print(f"🔍 Searching top {top_k} similar bugs...")
        query_vec = self.model.encode(query_text, convert_to_tensor=True)
        scores = util.cos_sim(query_vec, self.bug_vectors)[0]

        # top-k
        top_results = scores.topk(k=top_k)

        result_list = []
        for score, idx in zip(top_results.values, top_results.indices):
            idx = int(idx)
            result_list.append({
                "score": float(score),
                "title": self.bug_texts[idx]["title"],
                "body": self.bug_texts[idx]["body"],
                "_id": self.bug_texts[idx]["_id"]
            })
        return result_list


if __name__ == "__main__":
    # 修改为你的 Excel 路径
    db = BugKnowledgeBase("C:\\Users\\23314\\Desktop\\Fim\\database\\GitHub_Bug_Report_all.xlsx")

    query = "编辑笔记后保存失败"
    results = db.query(query, top_k=5)

    for r in results:
        print("\n======================")
        print(f"Score: {r['score']:.4f}")
        print(f"ID: {r['_id']}")
        print(f"Title: {r['title']}")
        print(f"Body: {r['body']}")
