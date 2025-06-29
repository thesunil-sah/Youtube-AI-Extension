# backend/services/vector.py

import os
from typing import List, Optional

from dotenv import load_dotenv
import openai
import chromadb
from chromadb.config import Settings
from fastapi import HTTPException

# ── Load environment ─────────────────────────────────────────────────────────
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL    = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
CHROMA_DIR     = os.getenv("CHROMA_DIR", "db")  # local persistence directory

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in .env")

openai.api_key = OPENAI_API_KEY

# ── Initialize a local-persistent Chroma client ────────────────────────────
# Uses the new PersistentClient API (no more LEGACY_ERROR) :contentReference[oaicite:0]{index=0}
client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings()
)

def get_collection(video_id: str):
    """
    Return an existing Chroma collection for this video, or create it.
    """
    name = f"yt-{video_id}"
    existing = {c.name for c in client.list_collections()}
    if name in existing:
        return client.get_collection(name=name)
    return client.create_collection(name=name, metadata={"video_id": video_id})

def embed_and_upsert(video_id: str, segments: List[dict]) -> dict:
    """
    Embeds each segment via OpenAI, then upserts into Chroma under namespace=video_id.
    """
    texts = [seg["text"] for seg in segments]
    resp  = openai.Embedding.create(model=EMBED_MODEL, input=texts)
    embs  = [d["embedding"] for d in resp["data"]]

    col = get_collection(video_id)
    ids = [f"{video_id}-{i}" for i in range(len(texts))]
    metadatas = [{"start": seg["start"], "duration": seg.get("duration")} for seg in segments]

    col.upsert(
        embeddings=embs,
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    client.persist()
    return {"upserted_count": len(texts)}

def query_index(
    query_text: str,
    top_k: int = 5,
    video_id: Optional[str] = None
) -> List[dict]:
    """
    Embeds the query_text, then queries the matching Chroma collection(s).
    """
    resp = openai.Embedding.create(model=EMBED_MODEL, input=[query_text])
    q_emb = resp["data"][0]["embedding"]

    # choose one or all collections
    col_names = [f"yt-{video_id}"] if video_id else [c.name for c in client.list_collections()]
    results = []
    for name in col_names:
        col = client.get_collection(name=name)
        res = col.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances", "ids"]
        )
        for idx, doc in enumerate(res["documents"][0]):
            results.append({
                "id":       res["ids"][0][idx],
                "score":    res["distances"][0][idx],
                "text":     doc,
                "metadata": res["metadatas"][0][idx],
                "videoId":  video_id or col.metadata.get("video_id")
            })
    return results
