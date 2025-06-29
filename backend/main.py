# backend/main.py

import os
from typing import Optional, List
from fastapi import FastAPI, Query, HTTPException
from dotenv import load_dotenv
from openai import OpenAI

from .services.captions import fetch_captions, DEFAULT_LANGUAGES
from .services.vector import embed_and_upsert

# ── Load env & init OpenAI client ────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set in .env")
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

# ── Helpers for summarization ─────────────────────────────────────────────────
MAX_SEGMENTS_PER_CHUNK = 100

async def _summarize_text(text: str) -> str:
    """Call OpenAI to summarize a single piece of text."""
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": text}
        ],
        temperature=0.5,
        max_tokens=250
    )
    return resp.choices[0].message.content.strip()

async def _chunk_and_summarize(segments: List[dict]) -> str:
    """Break segments into chunks, summarize each, then combine summaries."""
    # 1) Split into sublists of MAX_SEGMENTS_PER_CHUNK
    chunks = [
        segments[i : i + MAX_SEGMENTS_PER_CHUNK]
        for i in range(0, len(segments), MAX_SEGMENTS_PER_CHUNK)
    ]

    # 2) Summarize each chunk
    chunk_summaries = []
    for chunk in chunks:
        chunk_text = " ".join(seg["text"] for seg in chunk)
        prompt = (
            "Please provide a concise summary of the following transcript segment:\n\n"
            f"{chunk_text}\n\nSummary:"
        )
        chunk_summaries.append(await _summarize_text(prompt))

    # 3) If only one chunk, return it
    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    # 4) Otherwise, summarize the summaries
    combined = "\n\n".join(chunk_summaries)
    final_prompt = (
        "The following are summaries of parts of a YouTube video transcript. "
        "Please combine them into one concise final summary:\n\n"
        f"{combined}\n\nFinal Summary:"
    )
    return await _summarize_text(final_prompt)

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/captions")
async def get_captions(
    video_id: str = Query(..., alias="videoId"),
    languages: Optional[str] = Query(None)
):
    langs = [l.strip() for l in languages.split(",")] if languages else None
    return fetch_captions(video_id, langs)

@app.post("/embed")
async def embed_video(video_id: str = Query(..., alias="videoId")):
    segments = fetch_captions(video_id)
    return embed_and_upsert(video_id, segments)

@app.get("/summary")
async def summarize_video(
    video_id: str = Query(..., alias="videoId"),
    languages: Optional[str] = Query(None)
):
    """
    Fetches captions and returns a concise summary for the given video.
    Handles long transcripts by chunking.
    """
    langs = [l.strip() for l in languages.split(",")] if languages else None
    segments = fetch_captions(video_id, langs)
    if not segments:
        raise HTTPException(status_code=404, detail="No transcript to summarize")

    try:
        summary = await _chunk_and_summarize(segments)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {e}")

    return {"summary": summary}
