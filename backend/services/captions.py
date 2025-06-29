from typing import List, Optional
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound
)
from fastapi import HTTPException

# YouTube’s most-common auto-generated caption languages, now including Hindi
DEFAULT_LANGUAGES = [
    "en",  # English
    "es",  # Spanish
    "pt",  # Portuguese
    "fr",  # French
    "de",  # German
    "it",  # Italian
    "nl",  # Dutch
    "ru",  # Russian
    "ja",  # Japanese
    "ko",  # Korean
    "tr",  # Turkish
    "zh",  # Chinese
    "hi",  # Hindi
]

def fetch_captions(
    video_id: str,
    languages: Optional[List[str]] = None
) -> List[dict]:
    """
    Fetches the transcript for a given YouTube video ID.
    Tries any manually created transcript first, then falls back
    to auto-generated (ASR) if manual is missing.
    Supports a priority list of languages; defaults to DEFAULT_LANGUAGES.
    Returns a list of {'text', 'start', 'duration'} dicts.
    """
    langs = languages or DEFAULT_LANGUAGES

    try:
        # get_transcript will try manual tracks in order, then ASR if none found
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
        return transcript

    except TranscriptsDisabled:
        raise HTTPException(
            status_code=404,
            detail="Transcripts are disabled for this video"
        )
    except NoTranscriptFound:
        raise HTTPException(
            status_code=404,
            detail="No transcript available for this video in any of: " + ", ".join(langs)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching transcript: {e}"
        )
