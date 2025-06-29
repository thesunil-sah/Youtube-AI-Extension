from typing import Optional
from fastapi import FastAPI, Query
from .services.captions import fetch_captions, DEFAULT_LANGUAGES

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/captions")
async def get_captions(
    video_id: str = Query(
        "GWnSsjT4V68",
        alias="videoId",
        description="YouTube video ID (defaults to Rick Astley video for testing)"
    ),
    languages: Optional[str] = Query(
        None,
        description=(
            "Comma-separated list of ISO language codes in priority order. "
            f"Defaults to: {','.join(DEFAULT_LANGUAGES)}"
        )
    )
):
    """
    Fetch captions, trying manual first then auto-generated.
    • GET /captions?videoId=ID
    • GET /captions?videoId=ID&languages=es,fr,en
    """
    langs_list = [lang.strip() for lang in languages.split(",")] if languages else None
    return fetch_captions(video_id, langs_list)
