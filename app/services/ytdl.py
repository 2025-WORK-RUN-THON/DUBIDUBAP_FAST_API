from __future__ import annotations

import os
import subprocess
from typing import Optional
from yt_dlp import YoutubeDL


def ensure_ffmpeg_installed() -> None:
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg (e.g., brew install ffmpeg)."
        ) from exc

def ensure_ytdlp_installed() -> None:
    try:
        subprocess.run(["yt-dlp", "--version"], check=True, capture_output=True)
    except Exception as exc:
        raise RuntimeError(
            "yt-dlp not found. It should be installed via pip (dependency already added). Ensure your venv is activated and PATH is set."
        ) from exc


def download_audio(url: str, out_dir: str, filename_stem: Optional[str] = None, *, skip_if_exists: bool = True) -> str:
    """Download YouTube audio-only using yt-dlp Python API and extract as m4a.

    Returns the local filepath.
    """
    ensure_ffmpeg_installed()
    ensure_ytdlp_installed()
    os.makedirs(out_dir, exist_ok=True)
    stem = filename_stem or "audio"
    outtmpl = os.path.join(out_dir, f"{stem}.%(ext)s")

    if skip_if_exists and os.path.exists(os.path.join(out_dir, f"{stem}.m4a")):
        return os.path.join(out_dir, f"{stem}.m4a")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "noprogress": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
                "preferredquality": "0",
            }
        ],
        "nocheckcertificate": True,
        "retries": 3,
        "fragment_retries": 3,
        "hls_use_mpegts": True,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as exc:
        raise RuntimeError(f"yt-dlp download failed: {exc}") from exc

    # After postprocessing, the extension should be m4a
    out_path = os.path.join(out_dir, f"{stem}.m4a")
    if not os.path.exists(out_path):
        # Fallback: try common audio extensions
        for ext in ("m4a", "mp3", "opus", "webm", "aac"):
            candidate = os.path.join(out_dir, f"{stem}.{ext}")
            if os.path.exists(candidate):
                out_path = candidate
                break
    if not os.path.exists(out_path):
        raise RuntimeError("Audio file not found after yt-dlp processing")
    return out_path


