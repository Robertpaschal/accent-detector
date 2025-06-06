#!/usr/bin/env python3
"""
accent_detector.py: Extract audio from video (MP4) and classify English accents
using a pretrained SpeechBrain model (CommonAccent XLSR).
"""

import os
import tempfile
import torchaudio
import requests

# New import path
from speechbrain.pretrained import EncoderClassifier
from moviepy import VideoFileClip

# ———————————————— Setup Accent Classifier ————————————————
# This will load the “CommonAccent” XLSR‐based accent ID model from HF:
#
#   https://huggingface.co/Jzuluaga/accent-id-commonaccent_xlsr-en-english
#
# It expects a 16 kHz single‐channel WAV and returns:
#   out_prob : probability vector over 16 accents
#   score    : confidence score (log‐posterior)
#   index    : integer index of the predicted accent
#   text_lab : string label of the predicted accent (e.g. "us", "england", "australia", etc.)
#
classifier = EncoderClassifier.from_hparams(
    source="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
    savedir="tmp/accent_classification_model"
)

def download_video(video_url: str) -> str:
    """
    Download a public video (e.g. Loom MP4) from `video_url` into a local temp file.
    Return the local path to that MP4.
    """
    response = requests.get(video_url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download video (HTTP {response.status_code})")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            tmp.write(chunk)
    tmp.close()
    return tmp.name


def extract_audio(video_path: str) -> str:
    """
    Given a local MP4 file at `video_path`, extract its audio to a 16 kHz WAV.
    Return the local path to that WAV file.
    """
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    video = VideoFileClip(video_path)
    # Write PCM16 WAV @ 16 kHz
    video.audio.write_audiofile(
        audio_path, codec="pcm_s16le", fps=16000, verbose=False, logger=None
    )
    return audio_path


def classify_accent(audio_path: str) -> dict:
    """
    Run the pretrained SpeechBrain accent‐ID classifier on a local WAV file.
    Returns a dict with:
      - "accent"    : string label (e.g., "us", "england", "australia", etc.)
      - "confidence": float in [0.0, 1.0]
      - "summary"   : human‐readable summary line
    """
    # The classifier.classify_file method:
    #   - loads & resamples the WAV internally
    #   - returns (out_prob, score, index, text_lab)
    out_prob, score, index, text_lab = classifier.classify_file(audio_path)

    accent_label = text_lab  # e.g. "us", "england", etc.
    # `score` may be a tensor; convert to float
    confidence_score = float(score.item()) if hasattr(score, "item") else float(score)

    return {
        "accent": accent_label,
        "confidence": confidence_score,
        "summary": f"Detected accent: {accent_label} (confidence: {confidence_score:.2%})",
    }


def detect_accent_from_url(video_url: str) -> dict:
    """
    Full pipeline: download video from `video_url` → extract audio → classify accent.
    Cleans up temp files after inference.
    """
    video_path = download_video(video_url)
    audio_path = extract_audio(video_path)
    result = classify_accent(audio_path)

    # Clean up temporary files
    try:
        os.remove(video_path)
    except OSError:
        pass
    try:
        os.remove(audio_path)
    except OSError:
        pass

    return result
