import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

from core.vector_store import ingest_text

_YDL_OPTS = {
    "quiet":         True,
    "skip_download": True,
    "no_warnings":   True,
}


def scrape_youtube(url: str) -> str:
    """
    Extracts metadata and transcript from a YouTube video without any API key.
    Returns a single formatted document string ready for ingestion.
    """
    print(f"[YouTube] Fetching metadata: {url}")
    with yt_dlp.YoutubeDL(_YDL_OPTS) as ydl:
        info = ydl.extract_info(url, download=False)

    video_id   = info.get("id", "")
    title      = info.get("title",       "Unknown Title")
    channel    = info.get("uploader",    "Unknown Channel")
    views      = info.get("view_count",  0)
    date       = info.get("upload_date", "Unknown")          # YYYYMMDD
    description = info.get("description", "No description.")

    print(f"[YouTube] Fetching transcript for: '{title}'")
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript      = TextFormatter().format_transcript(transcript_list)
    except Exception as e:
        transcript = f"[Transcript unavailable: {e}]"

    return (
        f"[Source: YouTube]\n"
        f"Title:   {title}\n"
        f"Channel: {channel}\n"
        f"Date:    {date}\n"
        f"Views:   {views}\n\n"
        f"[Description]\n{description}\n\n"
        f"[Transcript]\n{transcript}"
    ).strip()


def ingest_youtube(url: str) -> None:
    """Scrape a YouTube video and save it to the vector database."""
    text = scrape_youtube(url)
    ingest_text(
        text,
        source=url,
        extra_payload={"type": "youtube", "url": url},
    )