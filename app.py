#!/usr/bin/env python3
"""
app.py: Streamlit UI for English accent detection from video URLs or local files.
"""

import streamlit as st
import tempfile
import os
import torchaudio
from moviepy import VideoFileClip
from accent_detector import classify_accent, detect_accent_from_url


def process_audio_file(file_path: str, file_ext: str) -> str:
    """
    Given a local file path and extension, either:
    - If video, extract audio to WAV and return WAV path.
    - If audio, return the same path.
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
    """Remove temporary files if they exist."""
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


st.set_page_config(page_title="English Accent Detector", page_icon="🗣️", layout="centered")

st.title("🎯 English Accent Detection from Video or Audio")

st.markdown("""
Upload an audio/video file or paste a **public video URL**, and the app will:
1. Extract audio (if needed).
2. Detect the speaker's English accent using a deep learning model.
""")

uploaded_file = st.file_uploader(
    "Upload audio/video file", type=["mp4", "mp3", "wav", "m4a", "mov", "mkv"]
)
url_input = st.text_input("Or paste a public video URL to analyze")

# Process uploaded file
if uploaded_file is not None:
    tmp_path = audio_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        file_ext = os.path.splitext(tmp_path)[1].lower()
        audio_path = process_audio_file(tmp_path, file_ext)

        st.audio(audio_path, format="audio/wav")
        waveform, sample_rate = torchaudio.load(audio_path)
        st.write(f"Sample Rate: {sample_rate}, Waveform shape: {tuple(waveform.shape)}")

        st.subheader("Accent Detection Result")
        with st.spinner("Analyzing audio…"):
            result = classify_accent(audio_path)

        st.success(result["summary"])
        st.metric("Detected Accent", result["accent"].capitalize())
        st.metric("Confidence", f"{result['confidence']:.2%}")

    except Exception as e:
        st.error(f"Error processing file: {e}")

    finally:
        cleanup_files(tmp_path, audio_path)

# Process URL input
elif url_input:
    try:
        with st.spinner("Downloading and analyzing video…"):
            result = detect_accent_from_url(url_input)

        st.success(result["summary"])
        st.metric("Detected Accent", result["accent"].capitalize())
        st.metric("Confidence", f"{result['confidence']:.2%}")

    except Exception as e:
        st.error(f"Error processing URL: {e}")

else:
    st.info("Please upload a file or paste a video URL to analyze.")
