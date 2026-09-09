import base64
from io import BytesIO
from PIL import Image
import requests
import time
import streamlit as st
import os
import speech_recognition as sr
from pydub import AudioSegment
import io

# --- Helper Functions ---

def test_api_key(api_key: str):
    """Quick check to confirm a Groq key is valid."""
    key = api_key.strip()
    if not key.startswith("gsk_"):
        return False, "That doesn't look like a Groq key — it should start with 'gsk_'."
    try:
        resp = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=15
        )
        if resp.status_code == 200:
            return True, "Key is valid and working."
        elif resp.status_code == 401:
            return False, "Groq says this key is invalid or revoked. Generate a new one at console.groq.com/keys."
        else:
            return False, f"Unexpected response ({resp.status_code}): {resp.text[:200]}"
    except requests.exceptions.RequestException as e:
        return False, f"Network error reaching Groq: {e}"


def format_result(text: str) -> str:
    """Clean up the response and add disclaimer."""
    # Remove any thinking/analysis text if present
    lines = text.split('\n')
    cleaned_lines = []
    skip = False
    
    for line in lines:
        # Skip lines that look like thinking/analysis
        if any(skip_phrase in line.lower() for skip_phrase in [
            'thinking process',
            'analyze user input',
            'deconstruct requirements',
            "let's draft",
            'refine',
            'note:',
            'bullets:',
            'bullet point',
            'i need to',
            'this is not a diagnosis',
            'check constraints',
            'draft - section by section',
            'mental refinement',
            'header 1:',
            'header 2:',
            'header 3:',
            'header 4:',
            'header 5:',
            'header 6:',
            'header 7:',
            'adjust to match the constraint',
            "i'll make sure",
            "let's adjust"
        ]):
            skip = True
            continue
        # Stop skipping when we hit actual content with ###
        if '###' in line:
            skip = False
            cleaned_lines.append(line)
        elif not skip and line.strip():
            cleaned_lines.append(line)
    
    cleaned_text = '\n'.join(cleaned_lines)
    
    # If no ### headers found, try to extract just the final answer
    if '###' not in cleaned_text:
        lines = cleaned_text.split('\n')
        found_answer = False
        final_lines = []
        for line in lines:
            if line.strip().startswith('###') or line.strip().startswith('-'):
                found_answer = True
                final_lines.append(line)
            elif found_answer and line.strip():
                final_lines.append(line)
        if final_lines:
            cleaned_text = '\n'.join(final_lines)
    
    # Add disclaimer if not present
    disclaimer = "This is not a diagnosis, please discuss these results with your doctor."
    if disclaimer.lower() not in cleaned_text.lower():
        cleaned_text = cleaned_text.rstrip() + f"\n\n---\n**{disclaimer}**"
    
    return cleaned_text


def build_payload(img_b64: str) -> dict:
    prompt_text = (
        "You are a friendly medical assistant explaining results to someone with NO medical background.\n\n"
        "IMPORTANT: Do NOT include any thinking, analysis, planning, or reasoning in your response. "
        "ONLY provide the final formatted answer directly. Start immediately with the first header.\n\n"
        "Use simple, everyday language. Avoid jargon, or if you must use a medical term, immediately explain "
        "it in plain words. Use short bullet points. For each point add one brief clause explaining WHY "
        "it matters (not just WHAT it is). Keep the full answer within 1000 output tokens.\n\n"
        "If it is a MEDICINE, use this exact format with these headers:\n"
        "### 1. What it treats and how it works\n"
        "### 2. When it's prescribed\n"
        "### 3. How to take it (dosage basics)\n"
        "### 4. Side effects to watch for (common and serious)\n"
        "### 5. Who should NOT take it\n"
        "### 6. What to avoid while taking it\n"
        "### 7. How to store it\n\n"
        "If it is a LAB REPORT, use this format:\n"
        "### 1. Biomarker results (number, and whether Normal, High, or Low)\n"
        "### 2. What abnormal values mean, in one simple sentence each\n"
        "### 3. Overall summary in plain language, avoiding scary diagnostic labels\n\n"
        "Start your response immediately with '### 1.' Do not include any introductory text or analysis."
    )

    return {
        "model": "qwen/qwen3.6-27b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.1,
        "reasoning_effort": "none"
    }


def transcribe_voice_from_bytes(audio_bytes: bytes) -> str:
    """
    Transcribe voice from audio bytes using Google Speech Recognition.
    
    Args:
        audio_bytes: The audio data in bytes (any common audio format)
    
    Returns:
        str: Transcribed text from the audio
    """
    try:
        # Try to convert audio to WAV format using pydub
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)
        
        # Use Google Speech Recognition (free, no API key needed)
        text = recognizer.recognize_google(audio_data)
        return text
        
    except sr.UnknownValueError:
        raise Exception("Could not understand the audio. Please speak more clearly or type your question.")
    except sr.RequestError as e:
        raise Exception(f"Speech recognition service error: {e}")
    except Exception as e:
        raise Exception(f"Voice processing error: {e}")


# --- Main Analysis Functions ---

def analyze_health_document_openai(image, api_key: str = None) -> str:
    """Analyze a medical document image using Groq API."""
    key = (api_key or st.secrets.get("GROQ_API_KEY", "")).strip()
    if not key:
        raise Exception("No API key configured. Set GROQ_API_KEY in Streamlit secrets.")

    # Handle different image input types
    if hasattr(image, 'read'):
        img = Image.open(image)
    elif isinstance(image, Image.Image):
        img = image
    else:
        img = Image.open(image)

    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

    payload = build_payload(img_b64)

    last_error = None

    for attempt in range(3):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            time.sleep(2)
            continue

        if response.status_code == 200:
            data = response.json()
            result_text = data["choices"][0]["message"]["content"]
            return format_result(result_text)

        if response.status_code == 401:
            raise Exception("Invalid API Key. Go to console.groq.com/keys, generate a fresh key.")

        if response.status_code == 429:
            error_data = response.json().get("error", {})
            msg = error_data.get("message", "")
            last_error = msg
            if "try again in" in msg:
                try:
                    delay = float(msg.split("try again in")[1].split("s")[0].strip())
                except Exception:
                    delay = 30
                if attempt < 2:
                    time.sleep(delay + 1)
                    continue
            raise Exception(f"Rate limit hit: {msg}")

        if response.status_code == 503:
            last_error = "Model temporarily overloaded (503)."
            if attempt < 2:
                time.sleep((2 ** attempt) * 2)
                continue
            raise Exception(
                "The AI model is temporarily overloaded on Groq's end. "
                "This usually clears up within a minute — please try again shortly."
            )

        last_error = f"Groq Error {response.status_code}: {response.text}"
        raise Exception(last_error)

    raise Exception(f"Analysis failed after multiple retries. Last error: {last_error}")


def analyze_text_query(query: str, api_key: str = None) -> str:
    """
    Handle text-only queries for medicine information using Groq API.
    
    Args:
        query: The user's question (e.g., "paracetamol 2 tablets for what is used for")
        api_key: Optional API key (will use from secrets if not provided)
    
    Returns:
        str: The AI's response in markdown format
    """
    key = (api_key or st.secrets.get("GROQ_API_KEY", "")).strip()
    if not key:
        raise Exception("No API key configured. Set GROQ_API_KEY in Streamlit secrets.")

    prompt_text = (
        "You are a friendly medical assistant answering health questions for someone with NO medical background.\n\n"
        "IMPORTANT: Do NOT include any thinking, analysis, planning, or reasoning in your response. "
        "ONLY provide the final formatted answer directly. Start immediately with the first header.\n\n"
        "Use simple, everyday language. Avoid jargon, or if you must use a medical term, immediately explain "
        "it in plain words. Use short bullet points. For each point add one brief clause explaining WHY "
        "it matters (not just WHAT it is). Keep the full answer within 1000 output tokens.\n\n"
        "If the user asks about a MEDICINE, use this exact format with these headers:\n"
        "### 1. What it treats and how it works\n"
        "### 2. When it's prescribed\n"
        "### 3. How to take it (dosage basics)\n"
        "### 4. Side effects to watch for (common and serious)\n"
        "### 5. Who should NOT take it\n"
        "### 6. What to avoid while taking it\n"
        "### 7. How to store it\n\n"
        "If the user asks about a LAB REPORT or health condition, use this format:\n"
        "### 1. Key findings\n"
        "### 2. What this means for you\n"
        "### 3. Next steps or recommendations\n\n"
        "Start your response immediately with '### 1.' Do not include any introductory text or analysis.\n\n"
        f"User question: {query}"
    )

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "qwen/qwen3.6-27b",
        "messages": [
            {
                "role": "user",
                "content": prompt_text
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.1,
        "reasoning_effort": "none"
    }

    last_error = None

    for attempt in range(3):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            time.sleep(2)
            continue

        if response.status_code == 200:
            data = response.json()
            result_text = data["choices"][0]["message"]["content"]
            return format_result(result_text)

        if response.status_code == 401:
            raise Exception("Invalid API Key. Go to console.groq.com/keys, generate a fresh key.")

        if response.status_code == 429:
            error_data = response.json().get("error", {})
            msg = error_data.get("message", "")
            last_error = msg
            if "try again in" in msg:
                try:
                    delay = float(msg.split("try again in")[1].split("s")[0].strip())
                except Exception:
                    delay = 30
                if attempt < 2:
                    time.sleep(delay + 1)
                    continue
            raise Exception(f"Rate limit hit: {msg}")

        if response.status_code == 503:
            last_error = "Model temporarily overloaded (503)."
            if attempt < 2:
                time.sleep((2 ** attempt) * 2)
                continue
            raise Exception(
                "The AI model is temporarily overloaded on Groq's end. "
                "This usually clears up within a minute — please try again shortly."
            )

        last_error = f"Groq Error {response.status_code}: {response.text}"
        raise Exception(last_error)

    raise Exception(f"Analysis failed after multiple retries. Last error: {last_error}")


# --- Gemini Alternative (Optional) ---

def setup_gemini(api_key: str = None):
    """Setup Gemini with the provided API key."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise Exception("Google Generative AI package not installed. Run: pip install google-generativeai")
    
    key = api_key or st.secrets.get("GEMINI_API_KEY", "")
    if not key:
        raise Exception("No Gemini API key configured. Set GEMINI_API_KEY in Streamlit secrets.")
    
    genai.configure(api_key=key)
    return genai.GenerativeModel('gemini-pro')


def analyze_text_query_gemini(query: str, model=None) -> str:
    """Handle text-only queries using Gemini."""
    if model is None:
        model = setup_gemini()
    
    prompt = f"""You are a friendly medical assistant. Answer this question using simple, everyday language.

IMPORTANT: Do NOT include any thinking, analysis, planning, or reasoning. ONLY provide the final formatted answer.

Use this exact format:
### 1. What it treats and how it works
### 2. When it's prescribed
### 3. How to take it (dosage basics)
### 4. Side effects to watch for
### 5. Who should NOT take it
### 6. What to avoid while taking it

Start immediately with '### 1.' No introductory text.

Question: {query}"""
    
    response = model.generate_content(prompt)
    return format_result(response.text)


def analyze_health_document_gemini(image, model=None) -> str:
    """Analyze a medical document image using Gemini."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise Exception("Google Generative AI package not installed. Run: pip install google-generativeai")
    
    if model is None:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if not key:
            raise Exception("No Gemini API key configured. Set GEMINI_API_KEY in Streamlit secrets.")
        
        genai.configure(api_key=key)
        model = genai.GenerativeModel('gemini-1.5-flash')
    
    if hasattr(image, 'read'):
        img = Image.open(image)
    elif isinstance(image, Image.Image):
        img = image
    else:
        img = Image.open(image)
    
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    
    prompt = """You are a friendly medical assistant explaining results to someone with NO medical background.

IMPORTANT: Do NOT include any thinking, analysis, planning, or reasoning. ONLY provide the final formatted answer.

Use simple, everyday language. Avoid jargon, or if you must use a medical term, immediately explain 
it in plain words. Use short bullet points. For each point add one brief clause explaining WHY 
it matters (not just WHAT it is). Keep the full answer within 1000 output tokens.

If it is a MEDICINE, use this exact format:
### 1. What it treats and how it works
### 2. When it's prescribed
### 3. How to take it (dosage basics)
### 4. Side effects to watch for (common and serious)
### 5. Who should NOT take it
### 6. What to avoid while taking it (food and drugs)
### 7. How to store it

If it is a LAB REPORT, use this format:
### 1. Biomarker results (number, and whether Normal, High, or Low)
### 2. What abnormal values mean, in one simple sentence each
### 3. Overall summary in plain language, avoiding scary diagnostic labels

Start your response immediately with '### 1.' Do not include any introductory text or analysis."""
    
    image_part = {
        "mime_type": "image/jpeg",
        "data": img_bytes
    }
    
    response = model.generate_content([prompt, image_part])
    return format_result(response.text)


# --- Utility Functions ---

def get_api_key():
    """Get the API key from secrets or environment variable."""
    try:
        return st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")


# Export the main functions for easy import
__all__ = [
    'analyze_health_document_openai',
    'analyze_text_query',
    'format_result',
    'test_api_key',
    'analyze_health_document_gemini',
    'analyze_text_query_gemini',
    'setup_gemini',
    'transcribe_voice_from_bytes'
]