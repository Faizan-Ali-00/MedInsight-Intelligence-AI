# 🩺 MedInsight — AI-Powered Medicine & Lab Report Analyzer

Making medical information easier to understand. Upload an image of a medicine package or lab report, and get a clear, structured, and simple explanation powered by AI.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?logo=groq&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Project Overview

I developed an AI-powered application that helps users understand medical information by simply uploading an image of a medicine package or a laboratory report. The system uses artificial intelligence to extract, analyze, and present medical information in a clear, structured, and understandable format.

**The problem:** Medical information is everywhere — on medicine labels, prescription instructions, and laboratory reports — but it's written in complex terminology that most people can't easily understand.

**The solution:** MedInsight bridges that gap by using AI to translate medical jargon into plain, simple language anyone can understand.

## ✨ Features

- 📸 Image Upload — Upload a photo of a medicine package or lab report
- 🤖 AI-Powered Analysis — Uses Groq's blazing-fast LLM inference to extract and interpret medical data
- 💊 Medicine Info Extraction — Dosage, usage, side effects, warnings
- 🧪 Lab Report Breakdown — Explains each test, value, and what it means
- 📝 Plain Language Output — No confusing medical jargon
- 🌐 Web-Based Interface — Easy-to-use Streamlit app
- 📷 Camera Support — Take photos directly from your device

## 🚀 Demo

Upload a photo of a medicine box or lab report, and get:

- What the medicine is used for
- How and when to take it
- Common side effects and warnings
- Lab values explained in plain English

## 🛠️ Tech Stack

- Frontend: Streamlit
- AI Inference: Groq API (Llama / Mixtral models)
- Image Processing: Python
- Camera Input: streamlit-back-camera-input
- Language: Python 3.10+
- Styling: HTML / CSS

## 📂 Project Structure

    MedInsight-Intelligence-AI/
    ├── app.py                    # Streamlit web app (main entry point)
    ├── gemini_service.py         # AI service integration (Groq)
    ├── camera_component/         # Camera input component
    │   └── index.html            # Frontend for camera
    ├── assets/                   # Images, logos, static files
    ├── requirements.txt          # Python dependencies
    ├── .gitignore                # Git ignore rules
    └── README.md

## ⚙️ Installation

### 1. Clone the repository

    git clone https://github.com/Faizan-Ali-00/MedInsight-Intelligence-AI.git
    cd MedInsight-Intelligence-AI

### 2. Create a virtual environment

    # Windows
    python -m venv venv
    venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate

### 3. Install dependencies

    pip install -r requirements.txt

### 4. Set up your API key

Create a `.env` file in the root directory:

    GROQ_API_KEY=your_groq_api_key_here

Get your API key from https://console.groq.com/keys

## ▶️ Usage

Run the Streamlit app:

    streamlit run app.py

Then open your browser at http://localhost:8501

1. Upload an image of a medicine package or lab report (or use the camera)
2. Click Analyze
3. Read the AI-generated explanation in plain language

## 🔒 Security Notes

- Never commit your `.env` file — it contains your Groq API key
- Make sure `.env` is listed in `.gitignore`
- If you accidentally expose a key, revoke it immediately at https://console.groq.com/keys

## ⚠️ Medical Disclaimer

This application is for informational and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider regarding any medical concerns.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m "Add some AmazingFeature"`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

Faizan Ali

- GitHub: https://github.com/Faizan-Ali-00
- Repository: https://github.com/Faizan-Ali-00/MedInsight-Intelligence-AI

## ⭐ Show Your Support

If this project helped you, please give it a star on GitHub.

## 🙏 Acknowledgments

- Groq — https://groq.com
- Streamlit — https://streamlit.io
- Open-source contributors
