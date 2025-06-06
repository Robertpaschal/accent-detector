#!/usr/bin/env python3
"""
accent_detector.py

Provides functions to detect English accents from audio extracted
from video URLs or local audio/video files using a pretrained
SpeechBrain accent classification model.
"""

import os
import tempfile
import requests
from typing import Optional, Dict, Any
from moviepy import VideoFileClip
from speechbrain.pretrained import EncoderClassifier
import streamlit as st


@st.cache_resource(show_spinner=False)
def get_classifier() -> EncoderClassifier:
    """
    Load and cache the pre-trained accent classifier model from HuggingFace.
    Returns:
        EncoderClassifier: SpeechBrain encoder classifier instance.
    """
    return EncoderClassifier.from_hparams(
        source="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
        savedir="tmp/accent_classification_model",
        run_opts={"device":"cpu"},
    )


def download_video(url: str, timeout: int = 15) -> str:
    """
    Download video from a public URL to a temporary file.
    
    Args:
        url (str): URL of the video.
        timeout (int): Timeout for the request in seconds.

    Returns:
        str: Path to the downloaded temporary video file.

    Raises:
        requests.RequestException: If download fails.
    """
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                tmp_file.write(chunk)
        return tmp_file.name


def extract_audio(video_path: str) -> str:
    """
    Extract audio from a video file and save it as a WAV temporary file.

    Args:
        video_path (str): Path to the video file.

    Returns:
        str: Path to the temporary WAV audio file.

    Raises:
        Exception: If audio extraction fails.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_file:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(
            audio_file.name,
            verbose=False,
            logger=None,
            codec="pcm_s16le",
            fps=16000,
        )
        return audio_file.name


def classify_accent(audio_path: str) -> Dict[str, Any]:
    """
    Classify the English accent from an audio file.

    Args:
        audio_path (str): Path to the audio WAV file.

    Returns:
        dict: Result dictionary containing:
            - "summary" (str): Human-readable summary.
            - "accent" (str): Predicted accent label.
            - "confidence" (float): Confidence score between 0 and 1.
            - "probabilities" (dict): Mapping of accents to their probabilities.
    
    Raises:
        Exception: If classification fails.
    """
    classifier = get_classifier()

    # Run classification
    out_prob, score, _, text_lab = classifier.classify_file(audio_path)

    labels = classifier.hparams.label_encoder.labels

    # Map each label to its corresponding probability for first sample
    probabilities = {label: float(prob) for label, prob in zip(labels, out_prob[0])}

    accent = text_lab[0]
    confidence = float(score[0])

    summary = f"Detected accent: {accent.capitalize()} with confidence {confidence:.2%}"

    return {
        "summary": summary,
        "accent": accent,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def cleanup_files(*paths: Optional[str]) -> None:
    """
    Delete temporary files if they exist.

    Args:
        *paths (str): Paths to files to delete.
    """
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def detect_accent_from_url(url: str) -> Optional[Dict[str, Any]]:
    """
    Download a video from a URL, extract audio, classify accent, and cleanup temp files.

    Args:
        url (str): URL of the video.

    Returns:
        Optional[dict]: Classification result dict or None on failure.
    """
    tmp_video_path = None
    tmp_audio_path = None
    try:
        tmp_video_path = download_video(url)
        tmp_audio_path = extract_audio(tmp_video_path)
        return classify_accent(tmp_audio_path)
    except Exception as e:
        st.error(f"Failed to process the video URL: {e}")
        st.info(
            "Troubleshooting tips:\n"
            "- Ensure the URL is a direct link to a public video (e.g., MP4, MOV, MKV).\n"
            "- The video should be accessible and not private or restricted.\n"
            "- Try uploading a local file if the URL does not work.\n"
            "- Large or long videos may take longer to process or may fail.\n"
            "- Only English speech is supported for accent detection."
        )
        return None
    finally:
        cleanup_files(tmp_video_path, tmp_audio_path)
