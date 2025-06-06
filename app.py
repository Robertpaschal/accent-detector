#!/usr/bin/env python3
"""
app.py: Streamlit UI for the SpeechBrain‐based English accent classifier.
"""

import streamlit as st
import tempfile
import os
import torchaudio
from moviepy import VideoFileClip

from accent_detector import classify_accent, detect_accent_from_url

st.set_page_config(page_title="Accent Detector", layout="centered")

st.title("English Accent Detection App")
st.caption(
    "Model: [CommonAccent XLSR‐English](https://huggingface.co/Jzuluaga/accent-id-commonaccent_xlsr-en-english)"
)
st.write(
    "Upload a video/audio file, or paste a public video URL, and we'll detect the English accent."
)

# ————— File Upload / URL Input —————
uploaded_file = st.file_uploader(
    "Upload an audio/video file", type=["mp4", "mp3", "wav", "m4a", "mov", "mkv"]
)
url_input = st.text_input("Or paste a public video URL to analyze")

def process_audio_file(file_path: str, file_ext: str) -> str:
    """
    Given a local file_path and extension, either:
      - If it's a video (mp4, mkv, mov, m4a), extract audio to WAV → return WAV path.
      - If it's audio (mp3, wav), return the same path.
    """
    if file_ext in [".mp4", ".mkv", ".mov", ".m4a"]:
        st.info("Extracting audio from video…")
        clip = VideoFileClip(file_path)
        audio_path = file_path + ".wav"
        clip.audio.write_audiofile(
            audio_path, verbose=False, logger=None, codec="pcm_s16le", fps=16000
        )
        return audio_path

    elif file_ext in [".mp3", ".wav"]:
        st.info("Processing audio file…")
        return file_path

    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def cleanup_files(*file_paths: str) -> None:
    """Remove any temporary files that exist."""
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


# ————— Handle Local Upload —————
if uploaded_file is not None:
    try:
        # Save the uploaded file to a temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        file_ext = os.path.splitext(tmp_path)[1].lower()
        audio_path = process_audio_file(tmp_path, file_ext)

        # Show a built-in audio player
        st.audio(audio_path, format="audio/wav")
        waveform, sample_rate = torchaudio.load(audio_path)
        st.write(f"Sample Rate: {sample_rate}, Waveform shape: {tuple(waveform.shape)}")

        # Run the accent classifier
        st.subheader("Accent Detection Result")
        with st.spinner("Analyzing…"):
            result = classify_accent(audio_path)

        st.success(result["summary"])
        st.metric("Detected Accent", result["accent"])
        st.metric("Confidence", f"{result['confidence']:.2%}")

    except Exception as e:
        st.error(f"Error while processing the file: {e}")

    finally:
        cleanup_files(tmp_path, audio_path)

# ————— Handle Public Video URL —————
elif url_input:
    try:
        with st.spinner("Downloading and analyzing the video…"):
            result = detect_accent_from_url(url_input)

        st.success(result["summary"])
        st.metric("Detected Accent", result["accent"])
        st.metric("Confidence", f"{result['confidence']:.2%}")

    except Exception as e:
        st.error(f"Error while processing the URL: {e}")

else:
    st.info("Please upload a file or paste a video URL to analyze.")
