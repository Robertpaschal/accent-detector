# 🎙️ English Accent Detection App

A simple, smart tool to detect English language accents from uploaded audio/video files or public video URLs. Built using Streamlit and a fine-tuned Wav2Vec2 model.

---

## 🚀 Overview

This app helps identify English-speaking accents from candidates' speech. It accepts:
- 📂 Uploaded audio/video files (`.mp4`, `.mp3`, `.wav`, `.m4a`)
- 🔗 Public video URLs (e.g., Loom or direct MP4 links)

Using a pretrained deep learning model, the app extracts the audio, processes it, and classifies the speaker’s English accent with a confidence score.

---

## ✅ Features

- 🎧 Automatic audio extraction from video files
- 🌍 English accent classification (e.g., American, British, Australian)
- 📊 Confidence score output (e.g., 91.3%)
- 📝 Summary explanation
- 🧠 Based on the `CommonAccent XLSR‐English` model

---

## 🖥️ Demo

Try the live app here: [https://your-streamlit-app-link](https://your-streamlit-app-link)  
*(Replace with your deployed Streamlit Cloud or Hugging Face Spaces link)*

---

## 📦 Installation

### 🔧 Prerequisites

- Python 3.8+
- `ffmpeg` installed and available in your system PATH

### 🧪 Set up locally

1. **Clone the repo**
    ```bash
    git clone https://github.com/Robertpaschal/accent-detector.git
    cd accent-detector
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the app**
    ```bash
    streamlit run app.py
    ```

---

## 🧠 Model Details

### 🧠 Model Details

#### Model: [`CommonAccent XLSR‐English`](https://huggingface.co/Jzuluaga/accent-id-commonaccent_xlsr-en-english)

- **Architecture**: `speechbrain.pretrained`
- **Fine-tuned on six major English accents**:
    - 🇺🇸 American
    - 🇬🇧 British
    - 🇦🇺 Australian
    - 🇮🇳 Indian
    - 🇮🇪 Irish
    - 🇿🇦 South African

---

## 🔍 Example Output

```yaml
🎧 Uploaded: interview_clip.mp4

✅ Detected Accent: British
✅ Confidence: 87.4%
📘 Summary: Detected British accent with 87.4% confidence.
```

---

## 📤 Deployment

To deploy this app on Streamlit Cloud:

1. Push the project to a public GitHub repo.
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud).
3. Click “New App” > Connect GitHub > Select repo.
4. Set the main file to `app.py`.
5. Click Deploy.

---

## 📝 License

This project is for demonstration and interview purposes at REM Waste.  
Uses open-source model under Apache 2.0 License.

---

## 👋 Contact

Built by **Odinaka Robert Nnamani** – [nnamani.odinakarobert@gmail.com](mailto:nnamani.odinakarobert@gmail.com)  
Let’s connect on [LinkedIn](https://www.linkedin.com/in/odinaka-nnamani-fullstack-developer/)!