import streamlit as st
import markdown as md
from PIL import Image
import os
import tempfile
import base64
import json
import hashlib
from io import BytesIO
from pathlib import Path
from datetime import datetime
import requests
from bs4 import BeautifulSoup

# Back-camera-only capture widget (replaces st.camera_input, which has no
# facingMode / back-camera option in its Python API).
from streamlit_back_camera_input import back_camera_input

# Import from gemini_service
from gemini_service import (
    analyze_health_document_openai,
    analyze_text_query,
    format_result
)

# --- Page Config ---
st.set_page_config(
    page_title="MedInsight AI | Intelligent Medical Analysis System",
    page_icon="✚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Asset Paths ---
_asset_candidates = [
    Path(__file__).resolve().parent / "assets" / "doctor_portrait.jpg",
    Path.cwd() / "assets" / "doctor_portrait.jpg",
    Path(__file__).resolve().parent / "doctor_portrait.jpg",
]
DOCTOR_IMAGE = next((candidate for candidate in _asset_candidates if candidate.exists()), _asset_candidates[0])
_guidance_candidates = [
    Path(__file__).resolve().parent / "assets" / "doctor_guidance.jpg",
    Path.cwd() / "assets" / "doctor_guidance.jpg",
]
DOCTOR_GUIDANCE_IMAGE = next((candidate for candidate in _guidance_candidates if candidate.exists()), _guidance_candidates[0])
_intro_candidates = [
    Path(__file__).resolve().parent / "assets" / "doctor_intro.jpg",
    Path.cwd() / "assets" / "doctor_intro.jpg",
]
DOCTOR_INTRO_IMAGE = next((candidate for candidate in _intro_candidates if candidate.exists()), _intro_candidates[0])
DOCTOR_DATA_URI = ""
if DOCTOR_IMAGE.exists():
    with DOCTOR_IMAGE.open("rb") as doctor_file:
        DOCTOR_DATA_URI = "data:image/jpeg;base64," + base64.b64encode(doctor_file.read()).decode("ascii")

# --- WHO Topics ---
WHO_TOPICS = {
    "Digital health": "https://www.who.int/health-topics/digital-health",
    "Healthy diet": "https://www.who.int/health-topics/healthy-diet",
    "Mental health": "https://www.who.int/health-topics/mental-health",
    "Diabetes": "https://www.who.int/health-topics/diabetes",
    "Cardiovascular diseases": "https://www.who.int/health-topics/cardiovascular-diseases",
    "Antimicrobial resistance": "https://www.who.int/health-topics/antimicrobial-resistance",
    "Air pollution": "https://www.who.int/health-topics/air-pollution",
    "Vaccines and immunization": "https://www.who.int/health-topics/vaccines-and-immunization",
    "Emergency care": "https://www.who.int/health-topics/emergency-care",
    "Women's health": "https://www.who.int/health-topics/women-s-health",
}

# --- WHO Functions ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_who_updates():
    """Retrieve a small set of current official WHO newsroom links."""
    try:
        response = requests.get("https://www.who.int/news-room", timeout=12, headers={"User-Agent": "MedInsightAI/1.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        updates = []
        seen = set()
        for link in soup.select("a[href]"):
            href = link.get("href", "")
            title = " ".join(link.get_text(" ", strip=True).split())
            if not title or len(title) < 20 or "/news/item/" not in href:
                continue
            url = href if href.startswith("http") else "https://www.who.int" + href
            if url in seen:
                continue
            seen.add(url)
            updates.append({"title": title, "url": url})
            if len(updates) == 6:
                break
        return updates
    except Exception:
        return []

WHO_ARCHIVE_FILE = Path(__file__).resolve().parent / ".who_updates_archive.json"

def load_who_archive():
    try:
        if WHO_ARCHIVE_FILE.exists():
            return json.loads(WHO_ARCHIVE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        pass
    return []

def save_who_archive(archive):
    try:
        WHO_ARCHIVE_FILE.write_text(json.dumps(archive, indent=2), encoding="utf-8")
    except OSError:
        pass

def update_who_archive(latest_updates):
    archive = load_who_archive()
    known_urls = {item.get("url") for item in archive}
    changed = False
    for item in latest_updates:
        if item.get("url") not in known_urls:
            archive.insert(0, {"title": item.get("title"), "url": item.get("url"), "first_seen": datetime.now().strftime("%d %b %Y, %H:%M")})
            known_urls.add(item.get("url"))
            changed = True
    archive = archive[:60]
    if changed:
        save_who_archive(archive)
    return archive

# --- Usage Functions ---
USAGE_FILE = Path(__file__).resolve().parent / ".medinsight_usage.json"

def load_usage():
    defaults = {"documents_analyzed": 24, "saved_insights": 12, "reports_reviewed": 18, "analysis_ids": [], "history": []}
    try:
        if USAGE_FILE.exists():
            defaults.update(json.loads(USAGE_FILE.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        pass
    return defaults

def save_usage(usage):
    try:
        USAGE_FILE.write_text(json.dumps(usage, indent=2), encoding="utf-8")
    except OSError:
        pass

if "usage" not in st.session_state:
    st.session_state.usage = load_usage()

def register_analysis(image_source):
    try:
        file_bytes = image_source.getvalue()
    except AttributeError:
        file_bytes = bytes(image_source)
    analysis_id = hashlib.sha256(file_bytes).hexdigest()
    usage = st.session_state.usage
    if analysis_id in usage.get("analysis_ids", []):
        return False
    usage.setdefault("analysis_ids", []).append(analysis_id)
    usage["documents_analyzed"] = int(usage.get("documents_analyzed", 0)) + 1
    usage["reports_reviewed"] = int(usage.get("reports_reviewed", 0)) + 1
    filename = getattr(image_source, "name", "Medical document") or "Medical document"
    usage.setdefault("history", []).insert(0, {"name": filename, "time": datetime.now().strftime("%d %b %Y, %H:%M"), "status": "Reviewed"})
    usage["history"] = usage["history"][:20]
    save_usage(usage)
    return True

# --- Camera helper ---
#
# NOTE ON THE CAMERA FIX:
# st.camera_input has no facingMode / "back camera only" option in its
# Python API, so it can't be forced away from the front camera in a
# reliable, cross-device way. `back_camera_input` (from the
# streamlit-back-camera-input package) opens the camera stream directly
# with getUserMedia({facingMode: "environment"}), so it defaults to the
# rear camera on phones without exposing a flip control that can get
# stuck on the front camera.
#
# The component's return value isn't guaranteed to be a drop-in
# UploadedFile (it may be a PIL Image or a bytes-like object depending on
# version), so we normalize it here into something with .getvalue() and
# .name, which is what register_analysis() / analyze_health_document_openai()
# and the rest of this file expect.
def normalize_camera_image(raw, filename="camera_capture.jpg"):
    """Coerce whatever back_camera_input() returns into an UploadedFile-like object."""
    if raw is None:
        return None
    try:
        if hasattr(raw, "save"):
            # Already a PIL Image
            pil_img = raw
        else:
            data = raw.getvalue() if hasattr(raw, "getvalue") else raw
            pil_img = Image.open(BytesIO(data))
        buf = BytesIO()
        pil_img.convert("RGB").save(buf, format="JPEG")
        buf.seek(0)
        buf.name = filename
        return buf
    except Exception:
        # If normalization fails for any reason, fall back to the raw
        # value so the rest of the app can still try to use it.
        return raw

# --- CSS ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

    :root {
        --bg: #f0f7f2;
        --paper: #ffffff;
        --ink: #1a2e26;
        --muted: #5a7a6a;
        --line: #d4e8da;
        --forest: #0d3d2e;
        --forest2: #1a5a44;
        --green: #2e9b62;
        --green-light: #e8f5ec;
        --green-lighter: #f0faf4;
        --lime: #c9f1d3;
        --sage: #e8f5ec;
        --cream: #fbf7ed;
        --danger: #a44145;
    }

    /* Brand with Medical Cross */
    .brand {
        display: flex;
        gap: .7rem;
        align-items: center;
        padding: .5rem 0 1.5rem;
    }

    .brand-mark {
        display: grid;
        place-items: center;
        width: 42px;
        height: 42px;
        border-radius: 12px;
        background: white;
        color: #0d3d2e;
        font-weight: 800;
        font-size: 1.4rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 2px solid white;
    }

    .brand-name {
        font-family: 'Manrope';
        font-size: 1rem;
        font-weight: 800;
        letter-spacing: -.04em;
        color: white;
    }

    .brand-sub {
        opacity: .6;
        font-size: .65rem;
        margin-top: .12rem;
        color: #dceee2;
    }

    .side-section {
        font-size: .64rem;
        opacity: .5;
        text-transform: uppercase;
        letter-spacing: .15em;
        margin: .9rem 0 .5rem;
        color: #dceee2;
    }

    /* Rest of your CSS */
    html,body,[class*="css"]{font-family:'DM Sans',sans-serif}.stApp{background:var(--bg);color:var(--ink)}
    [data-testid="stHeader"]{background:transparent}
    .block-container{max-width:1350px;padding:1rem 2.4rem 2rem}
    [data-testid="stSidebar"]{background:var(--forest);border:0}
    [data-testid="stSidebar"] *{color:#dceee2!important}
    [data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.12)}
    [data-testid="stSidebar"] .stRadio label{padding:.45rem .55rem;border-radius:9px}
    [data-testid="stSidebar"] .stRadio label:hover{background:rgba(255,255,255,.08)}

    .topbar{display:flex;justify-content:space-between;align-items:center;padding:.35rem 0 1.3rem}
    .topmark{font-family:'Manrope';font-size:1.15rem;font-weight:800;color:var(--forest);letter-spacing:-.05em}
    .topmark span{color:var(--green)}
    .top-actions{display:flex;gap:.6rem;align-items:center}
    .status{display:inline-flex;align-items:center;gap:.4rem;background:#e7f6eb;color:#277449;border:1px solid #c9e8d0;border-radius:99px;padding:.43rem .65rem;font-size:.7rem;font-weight:700}
    .dot{width:7px;height:7px;background:#39a86f;border-radius:50%;animation:pulse-dot 2s infinite}
    @keyframes pulse-dot{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.3;transform:scale(.8)}}
    .avatar{display:grid;place-items:center;width:32px;height:32px;border-radius:50%;background:var(--forest);color:white;font-size:.7rem;font-weight:800}

    .hero-premium {
        background: linear-gradient(135deg, #0a3226 0%, #1a5a44 60%, #0d3d2e 100%);
        border-radius: 24px;
        padding: 2.5rem 3.5rem;
        display: flex;
        align-items: center;
        gap: 2.5rem;
        min-height: 320px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(13, 61, 46, 0.25);
        width: 100%;
    }
    .hero-premium::before {
        content: '';
        position: absolute;
        top: -30%;
        right: -5%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(46, 155, 98, 0.08) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-premium::after {
        content: '';
        position: absolute;
        bottom: -20%;
        left: 10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(46, 155, 98, 0.05) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-content { flex: 1.2; position: relative; z-index: 2; }
    .hero-visual { flex: 0.8; position: relative; z-index: 2; display: flex; justify-content: center; align-items: flex-end; min-height: 280px; }
    .hero-badge { display: inline-block; background: rgba(46, 155, 98, 0.2); backdrop-filter: blur(10px); border: 1px solid rgba(46, 155, 98, 0.25); padding: .35rem 1rem; border-radius: 99px; font-size: .6rem; font-weight: 700; color: #91deb0; letter-spacing: .05em; margin-bottom: .8rem; }
    .hero-title { font-family: 'Manrope'; font-size: clamp(2rem, 4.5vw, 3.5rem); font-weight: 800; line-height: 1.08; letter-spacing: -0.06em; color: white; margin: 0; max-width: 580px; }
    .hero-title .highlight { background: linear-gradient(135deg, #bdf0ca, #6bc98a); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .hero-desc { color: rgba(200, 223, 212, 0.9); font-size: .95rem; line-height: 1.6; max-width: 440px; margin: 1rem 0 1.5rem; }
    .hero-actions { display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; }
    .btn-primary-hero { display: inline-flex; align-items: center; gap: .5rem; background: linear-gradient(135deg, #bdf0ca, #7edba0); color: #0a3226 !important; padding: .75rem 2rem; border-radius: 12px; font-size: .8rem; font-weight: 800; text-decoration: none; transition: all .3s ease; box-shadow: 0 8px 25px rgba(46, 155, 98, 0.3); border: none; cursor: pointer; }
    .btn-primary-hero:hover { transform: translateY(-3px); box-shadow: 0 12px 35px rgba(46, 155, 98, 0.4); }
    .btn-secondary-hero { display: inline-flex; align-items: center; gap: .3rem; color: rgba(255, 255, 255, 0.85) !important; font-size: .8rem; font-weight: 600; text-decoration: none; padding: .75rem 1.5rem; border-radius: 12px; border: 2px solid rgba(255, 255, 255, 0.12); transition: all .3s ease; background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(10px); }
    .btn-secondary-hero:hover { background: rgba(255, 255, 255, 0.08); border-color: rgba(255, 255, 255, 0.25); transform: translateY(-2px); }

    .doctor-combo { display: flex; align-items: flex-end; gap: 0; position: relative; }
    .paperboard-premium { width: 200px; height: 150px; background: white; border-radius: 16px; padding: 1rem 1.2rem; box-shadow: 0 20px 50px rgba(0, 0, 0, 0.25); transform: rotate(-4deg) translateY(-10px); margin-right: -30px; flex-shrink: 0; position: relative; z-index: 3; transition: transform 0.3s ease; }
    .paperboard-premium:hover { transform: rotate(-2deg) translateY(-12px); }
    .paperboard-premium::before { content: ''; position: absolute; top: -8px; left: 50%; transform: translateX(-50%); width: 35px; height: 7px; background: #c0ccc0; border-radius: 3px 3px 0 0; box-shadow: 0 -2px 8px rgba(0, 0, 0, 0.08); }
    .paperboard-premium .pb-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: .4rem; }
    .paperboard-premium .pb-top-label { font-size: .45rem; font-weight: 800; color: #8a9e8a; text-transform: uppercase; letter-spacing: .08em; }
    .pb-status-dot { width: 7px; height: 7px; border-radius: 50%; background: #2e9b62; animation: pulse-dot 2s infinite; display: inline-block; }
    .paperboard-premium .pb-title { font-size: .7rem; font-weight: 700; color: #0a3226; margin-bottom: .2rem; }
    .paperboard-premium .pb-lines { margin: .2rem 0; }
    .paperboard-premium .pb-line { height: 4px; border-radius: 99px; background: #eef5ef; margin: .25rem 0; }
    .paperboard-premium .pb-line.short { width: 40%; }
    .paperboard-premium .pb-line.medium { width: 65%; }
    .paperboard-premium .pb-line.long { width: 85%; }
    .paperboard-premium .pb-footer { display: flex; align-items: center; gap: .4rem; margin-top: .4rem; padding-top: .4rem; border-top: 1.5px solid #eef5ef; }
    .pb-check-circle { width: 16px; height: 16px; background: #2e9b62; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 10px; font-weight: 800; flex-shrink: 0; }
    .pb-check-text { color: #2e9b62; font-size: .6rem; font-weight: 700; }
    .pb-version { margin-left: auto; font-size: .4rem; color: #8a9e8a; font-weight: 700; background: #f0f5f0; padding: .1rem .5rem; border-radius: 99px; }
    .doctor-portrait-premium { position: relative; z-index: 4; flex-shrink: 0; display: flex; align-items: flex-end; }
    .doctor-portrait-premium img { width: 160px; height: 225px; object-fit: cover; object-position: top center; border-radius: 80px 80px 0 0; box-shadow: 0 15px 40px rgba(0, 0, 0, 0.25); border: 4px solid rgba(255, 255, 255, 0.9); display: block; transition: transform 0.3s ease; }
    .doctor-portrait-premium img:hover { transform: scale(1.02); }
    .doctor-ai-badge { position: absolute; top: 5px; right: -5px; background: linear-gradient(135deg, #bdf0ca, #7edba0); color: #0a3226; padding: .25rem .6rem; border-radius: 99px; font-size: .5rem; font-weight: 800; box-shadow: 0 4px 15px rgba(46, 155, 98, 0.3); white-space: nowrap; display: flex; align-items: center; gap: .2rem; }
    .hero-stats-float { position: absolute; bottom: 15px; left: -15px; background: rgba(255, 255, 255, 0.06); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: .4rem .7rem; z-index: 5; display: flex; align-items: center; gap: .6rem; }
    .hero-stats-float .stat-item { text-align: center; }
    .hero-stats-float .stat-num { font-size: 1rem; font-weight: 800; color: #bdf0ca; display: block; }
    .hero-stats-float .stat-label { font-size: .45rem; color: rgba(255, 255, 255, 0.5); text-transform: uppercase; letter-spacing: .05em; }
    .hero-stats-float .stat-divider { width: 1px; height: 25px; background: rgba(255, 255, 255, 0.08); }

    @media(max-width:1024px) {
        .hero-premium { flex-direction: column; padding: 2rem 2rem; text-align: center; min-height: auto; }
        .hero-title { max-width: 100%; }
        .hero-desc { max-width: 100%; margin-left: auto; margin-right: auto; }
        .hero-actions { justify-content: center; }
        .hero-visual { min-height: 200px; width: 100%; }
        .paperboard-premium { width: 160px; height: 130px; padding: .7rem 1rem; transform: rotate(-3deg) translateY(-8px); margin-right: -20px; }
        .paperboard-premium .pb-title { font-size: .6rem; }
        .doctor-portrait-premium img { width: 130px; height: 185px; }
        .hero-stats-float { display: none; }
    }
    @media(max-width:768px) {
        .hero-premium { padding: 1.5rem 1.2rem; border-radius: 18px; }
        .hero-title { font-size: clamp(1.5rem, 4.5vw, 2rem); }
        .paperboard-premium { display: none; }
        .doctor-portrait-premium img { width: 100px; height: 140px; }
        .hero-visual { min-height: 150px; }
        .doctor-ai-badge { font-size: .4rem; padding: .15rem .4rem; top: -5px; right: -5px; }
        .hero-badge { font-size: .5rem; padding: .25rem .7rem; }
    }
    @media(max-width:480px) {
        .doctor-portrait-premium img { width: 80px; height: 115px; }
        .hero-visual { min-height: 120px; }
        .btn-primary-hero, .btn-secondary-hero { font-size: .65rem; padding: .5rem 1rem; }
    }

    .section-head{display:flex;justify-content:space-between;align-items:end;margin:2rem 0 .8rem}
    .section-title{font-family:'Manrope';font-size:1.3rem;font-weight:800;letter-spacing:-.045em;color:var(--forest)}
    .section-note{color:var(--muted);font-size:.75rem;margin-top:.2rem}
    .view-all{color:var(--forest2);font-size:.7rem;font-weight:800}

    .metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem}
    .metric{background:white;border:1px solid var(--line);border-radius:14px;padding:1rem;transition:all .3s ease}
    .metric:hover{transform:translateY(-3px);box-shadow:0 8px 20px rgba(0,0,0,.05)}
    .metric-label{color:var(--muted);font-size:.65rem;font-weight:700}
    .metric-value{font-family:'Manrope';font-size:1.5rem;color:var(--forest);font-weight:800;margin-top:.2rem}
    .metric-meta{color:#3d9665;font-size:.65rem;margin-top:.1rem}

    .cards{display:grid;grid-template-columns:repeat(3,1fr);gap:.8rem}
    .info-card{background:#fff;border:1px solid var(--line);border-radius:18px;padding:1.1rem;min-height:170px;position:relative;overflow:hidden;transition:all .3s ease}
    .info-card:hover{transform:translateY(-3px);box-shadow:0 8px 20px rgba(0,0,0,.05)}
    .info-card.green{background:var(--sage)}
    .info-card.cream{background:var(--cream)}
    .badge{display:inline-block;padding:.3rem .5rem;background:rgba(255,255,255,.7);border-radius:99px;color:var(--forest2);font-size:.6rem;font-weight:800}
    .info-title{font-family:'Manrope';font-size:1rem;line-height:1.1;color:var(--forest);max-width:180px;margin-top:2.2rem}
    .info-copy{font-size:.7rem;color:#668075;line-height:1.4;max-width:200px;margin-top:.4rem}
    .info-icon{position:absolute;right:.8rem;bottom:.8rem;width:48px;height:48px;border-radius:16px;background:var(--forest);color:#bdf0ca;display:grid;place-items:center;font-size:1.2rem}

    .who-container{background:#f8fcff;border:1px solid #c8e4ef;border-left:4px solid #0093d0;border-radius:18px;padding:1.1rem;margin-top:1.5rem;box-shadow:0 4px 15px rgba(0,91,126,.04)}
    .who-header{display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;margin-bottom:.6rem}
    .who-logo{background:linear-gradient(135deg,#0093d0,#007a7a);color:#fff;padding:.35rem .65rem;border-radius:8px;font-size:.65rem;font-weight:800;letter-spacing:.05em}
    .who-title{font-family:'Manrope';font-weight:800;color:var(--forest);font-size:.95rem}
    .who-live{background:#e3f4fa;color:#00749b;padding:.2rem .5rem;border-radius:99px;font-size:.55rem;font-weight:800}
    .who-note{color:#5d7480;font-size:.65rem;line-height:1.4;margin-bottom:.6rem}
    .who-post{display:flex;gap:.7rem;padding:.6rem 0;border-bottom:1px solid #e6f0f3}
    .who-post:last-child{border-bottom:0}
    .who-icon{display:grid;place-items:center;width:36px;height:36px;border-radius:10px;background:#e2f2f7;font-size:1rem;flex-shrink:0}
    .who-content{flex:1}
    .who-post-title{color:var(--forest);font-size:.75rem;font-weight:800}
    .who-post-meta{color:#80959d;font-size:.6rem;margin-top:.15rem}
    .who-link{color:#007a9f;font-size:.62rem;font-weight:800;text-decoration:none;white-space:nowrap}
    .who-link:hover{text-decoration:underline}

    .workspace{background:white;border-radius:18px;padding:1.1rem;box-shadow:0 4px 12px rgba(0,0,0,.03)}
    .workspace-top{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.7rem}
    .workspace-title{font-family:'Manrope';font-size:1rem;font-weight:800;color:var(--forest)}
    .workspace-copy{color:var(--muted);font-size:.7rem;margin-top:.15rem}
    .privacy-tag{color:var(--forest2);background:#d9f4df;border:1px solid #b5ddbd;border-radius:99px;padding:.3rem .5rem;font-size:.6rem;font-weight:800}
    .drop{background:#edf8ef;border:2px dashed #62ad78;border-radius:12px;padding:.8rem}
    [data-testid="stFileUploader"]{border:0;padding:0;background:transparent}
    [data-testid="stFileUploaderDropzone"]{border:0;background:transparent;min-height:50px}
    .hint{text-align:center;color:var(--muted);font-size:.65rem;margin-top:.2rem}

    .report-container {
        background: white;
        border-radius: 16px;
        overflow: hidden;
        margin-top: .8rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
        border: 1px solid var(--line);
    }
    .report-header {
        padding: 0.8rem 1.5rem;
        background: var(--green-lighter);
        border-bottom: 1px solid var(--line);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .report-header .report-title {
        font-family: 'Manrope';
        font-size: 1rem;
        font-weight: 800;
        color: var(--forest);
        letter-spacing: -0.02em;
    }
    .report-header .report-badge {
        background: #e5f6e8;
        color: #277449;
        border-radius: 99px;
        padding: .2rem .6rem;
        font-size: .6rem;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: .3rem;
    }
    .report-body-wrapper {
        padding: 0.5rem 0.5rem 0.5rem 0.5rem;
        background: var(--green-lighter);
        margin: 0.5rem 1.5rem 0.5rem 1.5rem;
        border-radius: 6px;
        position: relative;
    }
    .report-body-wrapper::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: var(--green);
        border-radius: 4px 0 0 4px;
    }
    .report-body {
        padding: 0.8rem 1rem 0.8rem 0.8rem;
        color: var(--ink);
        font-size: .85rem;
        line-height: 1.6;
        margin-left: 8px;
    }
    .report-body p { margin: 0.3rem 0; color: var(--ink); }
    .report-body h1 { font-family: 'Manrope'; font-size: 1.1rem; font-weight: 800; color: var(--forest); margin-top: 0.8rem; margin-bottom: 0.3rem; letter-spacing: -0.02em; }
    .report-body h1:first-child { margin-top: 0; }
    .report-body h2 { font-family: 'Manrope'; font-size: 0.9rem; font-weight: 700; color: var(--forest); margin-top: 0.6rem; margin-bottom: 0.2rem; letter-spacing: -0.01em; }
    .report-body h3 { font-family: 'Manrope'; font-size: 0.85rem; font-weight: 700; color: var(--forest); margin-top: 0.5rem; margin-bottom: 0.2rem; }
    .report-body ul, .report-body ol { margin: 0.2rem 0 0.2rem 1.2rem; padding-left: 0.3rem; }
    .report-body li { margin: 0.15rem 0; color: var(--ink); }
    .report-body blockquote { background: #fff6e9; border-left: 3px solid #e3aa62; padding: .3rem .6rem; color: #795d3a; margin: 0.3rem 0; border-radius: 0 4px 4px 0; font-size: 0.8rem; }
    .report-body strong { color: var(--forest); font-weight: 700; }
    .report-disclaimer { padding: 0.5rem 1.5rem 0.8rem 1.5rem; border-top: 1px solid var(--line); }
    .report-disclaimer .disclaimer-box { background: #fff8ef; border: 1px solid #f3dfc4; border-radius: 8px; padding: .5rem .7rem; color: #795d3a; font-size: .6rem; line-height: 1.4; }

    .history-item{display:flex;justify-content:space-between;align-items:center;padding:.7rem 0;border-bottom:1px solid #edf1ed}
    .history-item:last-child{border:0}
    .history-name{color:var(--forest);font-size:.75rem;font-weight:800}
    .history-meta{color:var(--muted);font-size:.62rem;margin-top:.1rem}
    .history-status{color:#317551;background:#edf8ef;padding:.25rem .5rem;border-radius:99px;font-size:.58rem;font-weight:800}

    .stButton>button{background:var(--forest2);color:white;border:0;border-radius:12px;min-height:2.4rem;font-weight:800;transition:all .3s ease}
    .stButton>button:hover{background:var(--forest);color:white;transform:translateY(-2px);box-shadow:0 8px 20px rgba(13,61,46,.15)}
    .stTabs [data-baseweb="tab-list"]{gap:.4rem;background:#c5e7cc;border-radius:12px;padding:.25rem}
    .stTabs [data-baseweb="tab"]{font-size:.7rem;font-weight:800;color:var(--forest2);border-radius:8px;padding:.45rem .7rem}
    .stTabs [aria-selected="true"]{background:var(--forest);color:white!important}

    .stTextArea textarea {
        font-size: 0.9rem !important;
        border-radius: 12px !important;
        border: 2px solid var(--line) !important;
        padding: 0.8rem !important;
        background: white !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--green) !important;
        box-shadow: 0 0 0 3px rgba(46, 155, 98, 0.1) !important;
    }

    .camera-selector {
        margin-bottom: 1rem;
    }

    .footer{text-align:center;color:#6f8b78;font-size:.62rem;padding:1.5rem 0 .5rem}
    @media(max-width:850px){.metrics{grid-template-columns:repeat(2,1fr)}.cards{grid-template-columns:1fr}.top-actions .status{display:none}.workspace-top{display:block}.privacy-tag{display:inline-block;margin-top:.5rem}}
    @media(max-width:600px){.report-body-wrapper{margin:0.3rem 0.8rem;padding:0.3rem}.report-body{padding:0.5rem 0.6rem 0.5rem 0.5rem;font-size:.8rem}}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    # Brand with Medical Cross (White ✚)
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">✚</div>
            <div>
                <div class="brand-name">MedInsight AI</div>
                <div class="brand-sub">Intelligent Medical Analysis System</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="side-section">Main menu</div>', unsafe_allow_html=True)
    page = st.radio("Main menu", ["Dashboard", "New analysis", "WHO knowledge", "History", "Saved insights"], label_visibility="collapsed")

    st.markdown('<div class="side-section">Quick capture</div>', unsafe_allow_html=True)
    st.caption("Capture here, then finish the analysis in the workspace. Uses your back camera by default.")
    sidebar_camera_raw = back_camera_input(key="sidebar_back_camera")
    sidebar_camera_file = normalize_camera_image(sidebar_camera_raw, "sidebar_capture.jpg")

    st.markdown('<div class="side-section">Account & help</div>', unsafe_allow_html=True)
    page2 = st.radio("Account and help", ["No extra page", "Safety & privacy", "Settings", "Help centre"], label_visibility="collapsed")

    st.markdown('<hr><div style="font-size:.7rem;opacity:.6;line-height:1.55">MedInsight AI is designed to support understanding—not diagnosis. Always involve a qualified healthcare professional in medical decisions.</div>', unsafe_allow_html=True)

# --- Topbar ---
st.markdown(
    """
    <div class="topbar">
        <div class="topmark">MedInsight <span>AI</span></div>
        <div class="top-actions">
            <div class="status"><span class="dot"></span> System operational</div>
            <div class="avatar">MI</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Dashboard Page ---
if page == "Dashboard":
    usage = st.session_state.usage

    if DOCTOR_DATA_URI:
        st.markdown(f'''
        <div class="hero-premium">
            <div class="hero-content">
                <div class="hero-badge">✦ AI-Powered Medical Analysis</div>
                <h1 class="hero-title">
                    Make every medical document <br><span class="highlight">easier to understand</span>
                </h1>
                <p class="hero-desc">Read medicine labels, prescriptions, and lab reports with a clear, structured summary built for better health conversations.</p>
                <div class="hero-actions">
                    <a class="btn-primary-hero" href="#new-analysis">✨ Start new analysis</a>
                    <a class="btn-secondary-hero" href="#how-it-works">Explore the system →</a>
                </div>
            </div>
            <div class="hero-visual">
                <div class="doctor-combo">
                    <div class="paperboard-premium">
                        <div class="pb-top">
                            <span class="pb-top-label">📋 AI Recommendation</span>
                            <span class="pb-status-dot"></span>
                        </div>
                        <div class="pb-title">Medication Review</div>
                        <div class="pb-lines">
                            <div class="pb-line long"></div>
                            <div class="pb-line medium"></div>
                            <div class="pb-line short"></div>
                        </div>
                        <div class="pb-footer">
                            <span class="pb-check-circle">✓</span>
                            <span class="pb-check-text">Analysis Complete</span>
                            <span class="pb-version">v2.4</span>
                        </div>
                    </div>
                    <div class="doctor-portrait-premium">
                        <img src="{DOCTOR_DATA_URI}" alt="Medical professional">
                        <span class="doctor-ai-badge">✦ AI Assistant</span>
                    </div>
                </div>
                <div class="hero-stats-float">
                    <div class="stat-item">
                        <span class="stat-num">{usage.get("documents_analyzed", 0)}</span>
                        <span class="stat-label">Analyses</span>
                    </div>
                    <div class="stat-divider"></div>
                    <div class="stat-item">
                        <span class="stat-num">98%</span>
                        <span class="stat-label">Accuracy</span>
                    </div>
                    <div class="stat-divider"></div>
                    <div class="stat-item">
                        <span class="stat-num">4.9⭐</span>
                        <span class="stat-label">Rating</span>
                    </div>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="hero-premium">
            <div class="hero-content">
                <div class="hero-badge">✦ AI-Powered Medical Analysis</div>
                <h1 class="hero-title">
                    Make every medical document <br><span class="highlight">easier to understand</span>
                </h1>
                <p class="hero-desc">Read medicine labels, prescriptions, and lab reports with a clear, structured summary built for better health conversations.</p>
                <div class="hero-actions">
                    <a class="btn-primary-hero" href="#new-analysis">✨ Start new analysis</a>
                    <a class="btn-secondary-hero" href="#how-it-works">Explore the system →</a>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        st.warning("⚠️ Doctor image not found. Please add 'doctor_portrait.jpg' to the 'assets' folder.")

    if DOCTOR_INTRO_IMAGE.exists():
        doctor_col, trust_col = st.columns([0.22, 0.78], gap="medium")
        with doctor_col:
            st.image(str(DOCTOR_INTRO_IMAGE), caption="Your AI health-information guide", use_container_width=True, output_format="JPEG")
        with trust_col:
            st.markdown('<div style="padding:.7rem 0 0 .2rem"><div class="eyebrow" style="color:var(--forest);">A calmer way to begin</div><div class="section-title">Bring a question. We\'ll help organize the information.</div><div class="section-note" style="max-width:600px;margin-top:.4rem">Upload a document, take a photo, or speak your question. MedInsight AI is designed to support conversations with your healthcare professional.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-head"><div><div class="section-title">Your workspace at a glance</div><div class="section-note">A simple record of the information you have reviewed.</div></div><div class="view-all">Updated just now</div></div>', unsafe_allow_html=True)

    usage = st.session_state.usage
    st.markdown(f'<div class="metrics"><div class="metric"><div class="metric-label">Documents analyzed</div><div class="metric-value">{usage.get("documents_analyzed", 0)}</div><div class="metric-meta">Updates after each review</div></div><div class="metric"><div class="metric-label">Saved insights</div><div class="metric-value">{usage.get("saved_insights", 0)}</div><div class="metric-meta">Across your categories</div></div><div class="metric"><div class="metric-label">Reports reviewed</div><div class="metric-value">{usage.get("reports_reviewed", 0)}</div><div class="metric-meta">One count per document</div></div><div class="metric"><div class="metric-label">System status</div><div class="metric-value">Ready</div><div class="metric-meta">Analysis service online</div></div></div>', unsafe_allow_html=True)

    who_updates = fetch_who_updates()
    who_archive = update_who_archive(who_updates)

    who_html = '<div class="who-container"><div class="who-header"><span class="who-logo">🌍 WHO</span><span class="who-title">Health updates & guidelines</span><span class="who-live">Official source links</span></div><div class="who-note">Current items are linked from the official World Health Organization newsroom. Open the source for the complete update, date, evidence, and guidance.</div>'
    if who_updates:
        for item in who_updates[:4]:
            who_html += f'<div class="who-post"><div class="who-icon">✦</div><div class="who-content"><div class="who-post-title">{item["title"]}</div><div class="who-post-meta">WHO Newsroom · Official update</div></div><a class="who-link" href="{item["url"]}" target="_blank">Read ↗</a></div>'
    else:
        fallback = [("WHO Health topics", "https://www.who.int/health-topics"), ("WHO Fact sheets", "https://www.who.int/news-room/fact-sheets"), ("Disease Outbreak News", "https://www.who.int/emergencies/disease-outbreak-news"), ("WHO Data", "https://www.who.int/data")]
        for title, url in fallback:
            who_html += f'<div class="who-post"><div class="who-icon">✦</div><div class="who-content"><div class="who-post-title">{title}</div><div class="who-post-meta">Official WHO resource</div></div><a class="who-link" href="{url}" target="_blank">Open ↗</a></div>'
    who_html += '</div>'
    st.markdown(who_html, unsafe_allow_html=True)

    current_urls = {item["url"] for item in who_updates}
    older_updates = [item for item in who_archive if item.get("url") not in current_urls][:4]
    if older_updates:
        older_html = '<div class="who-container" style="margin-top:.8rem;border-left-color:#8a9e8a"><div class="who-header"><span class="who-logo" style="background:linear-gradient(135deg,#8a9e8a,#5a7a6a)">🗂️ Archive</span><span class="who-title">Previously shown updates</span></div><div class="who-note">Saved locally so earlier WHO items stay visible even after they roll off the live newsroom page.</div>'
        for item in older_updates:
            older_html += f'<div class="who-post"><div class="who-icon">🕘</div><div class="who-content"><div class="who-post-title">{item["title"]}</div><div class="who-post-meta">First seen {item.get("first_seen", "")}</div></div><a class="who-link" href="{item["url"]}" target="_blank">Read ↗</a></div>'
        older_html += '</div>'
        st.markdown(older_html, unsafe_allow_html=True)

    st.markdown('<div class="section-head" id="how-it-works"><div><div class="section-title">How MedInsight AI helps</div><div class="section-note">A thoughtful workflow from document to discussion.</div></div></div><div class="cards"><div class="info-card"><span class="badge">01 · UPLOAD</span><div class="info-title">Bring your document as it is.</div><div class="info-copy">Use a photo of a prescription, medicine box, or lab report. No retyping required.</div><div class="info-icon">↥</div></div><div class="info-card green"><span class="badge">02 · ANALYZE</span><div class="info-title">Find the important parts.</div><div class="info-copy">The visible information is grouped into familiar sections and explained simply.</div><div class="info-icon">✦</div></div><div class="info-card cream"><span class="badge">03 · DISCUSS</span><div class="info-title">Ask better questions.</div><div class="info-copy">Use the report as a starting point for your next conversation with a care team.</div><div class="info-icon">↗</div></div></div>', unsafe_allow_html=True)

    if DOCTOR_GUIDANCE_IMAGE.exists():
        guidance_image_col, guidance_copy_col = st.columns([0.42, 0.58], gap="large")
        with guidance_image_col:
            st.image(str(DOCTOR_GUIDANCE_IMAGE), caption="Designed for better care conversations", use_container_width=True, output_format="JPEG")
        with guidance_copy_col:
            st.markdown('<div style="padding:1rem 0"><div class="eyebrow" style="color:var(--forest);">Built around understanding</div><div class="section-title">A digital companion for the questions you want to ask.</div><div class="section-note" style="max-width:520px;margin-top:.55rem;line-height:1.65">The doctor imagery is intentionally used near the guidance and conversation moments. MedInsight AI supports your understanding while keeping medical decisions with you and your qualified healthcare professional.</div></div>', unsafe_allow_html=True)

# --- New Analysis Page ---
if page in ["Dashboard", "New analysis"]:
    st.markdown('<div class="section-head" id="new-analysis"><div><div class="section-title">Start a new analysis</div><div class="section-note">Upload an image or take a photo of your document.</div></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="workspace"><div class="workspace-top"><div><div class="workspace-title">Upload or capture a document</div><div class="workspace-copy">Upload an image of your prescription, lab report, or medicine label.</div></div><div class="privacy-tag">✦ Information-first, not diagnosis</div></div><div class="drop">', unsafe_allow_html=True)

    # Only two tabs: Upload and Camera
    tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Take Photo"])

    uploaded_file = None
    camera_file = None

    # TAB 1: Upload Image
    with tab1:
        st.markdown("### 📤 Upload a document image")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="main_upload")
        if uploaded_file:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            image = Image.open(uploaded_file)
            st.image(image, caption="Document preview", use_container_width=True)
        st.markdown('<div class="hint">Drop an image here or browse your device · Use a well-lit, in-focus photo</div>', unsafe_allow_html=True)

    # TAB 2: Take Photo
    #
    # NOTE ON THE CAMERA FIX:
    # st.camera_input has no facingMode / back-camera-only option in its
    # Python API, so trying to rebuild a front/back switch on top of it
    # (as the previous version of this app did with a selectbox mounting
    # three separate camera_input widgets) causes browsers to fight over
    # the camera stream and produces a black screen.
    #
    # back_camera_input() opens the stream directly with
    # getUserMedia({facingMode: "environment"}), so it defaults to the
    # rear camera with no extra switching UI needed. Users simply tap the
    # video preview to capture.
    with tab2:
        st.markdown("### 📷 Take a photo")
        st.caption(
            "Point your camera at the document and tap the video preview "
            "to capture. This opens your back (rear) camera by default."
        )

        camera_raw = back_camera_input(key="main_back_camera")
        camera_file = normalize_camera_image(camera_raw, "captured_document.jpg")

        if camera_file:
            st.success("✅ Photo captured!")
            image = Image.open(camera_file)
            st.image(image, caption="Captured photo", use_container_width=True)
            camera_file.seek(0)  # rewind so downstream reads start from byte 0

    st.markdown('</div></div>', unsafe_allow_html=True)

    # Get the image source
    image_source = uploaded_file if uploaded_file is not None else camera_file
    if image_source is None:
        image_source = sidebar_camera_file

    # Show what's happening
    if image_source is not None:
        st.info("📷 Document uploaded. Click 'Analyze' to read and explain it.")
    else:
        st.info("💡 Upload an image or take a photo to get started.")

    # Analysis button
    if st.button("✨ Analyze", use_container_width=True):
        with st.spinner("Analyzing..."):
            try:
                if image_source is not None:
                    result = analyze_health_document_openai(image_source)
                else:
                    st.warning("Please upload an image or take a photo to analyze.")
                    st.stop()

                if image_source is not None:
                    is_new_analysis = register_analysis(image_source)
                    if is_new_analysis:
                        st.toast("✅ Review added to your workspace and history.")

                result_html = md.markdown(result, extensions=["extra", "sane_lists"])

                report_html = f'''
                <div class="report-container">
                    <div class="report-header">
                        <span class="report-title">📋 Analysis Report</span>
                        <span class="report-badge">✓ Ready to review</span>
                    </div>
                    <div class="report-body-wrapper">
                        <div class="report-body">
                            {result_html}
                        </div>
                    </div>
                    <div class="report-disclaimer">
                        <div class="disclaimer-box">
                            <strong>⚠️ Important:</strong> This is an informational summary, not a diagnosis or a substitute for advice from a qualified healthcare professional. For severe or worsening symptoms, seek urgent medical care.
                        </div>
                    </div>
                </div>
                '''
                st.session_state["last_report_html"] = report_html
                st.rerun()

            except Exception as exc:
                st.error(f"❌ We couldn't complete the analysis. Details: {exc}")

    if st.session_state.get("last_report_html"):
        st.markdown(st.session_state["last_report_html"], unsafe_allow_html=True)

# --- WHO Knowledge Page ---
if page == "WHO knowledge":
    st.markdown('<div class="section-head"><div><div class="section-title">🌍 WHO Knowledge & Updates</div><div class="section-note">A curated doorway to official World Health Organization information.</div></div><div class="view-all">Official sources</div></div>', unsafe_allow_html=True)
    st.info("ℹ️ MedInsight AI links to WHO materials but does not replace the original source. Always check the official WHO page for the latest wording, dates, and guidance.")

    updates = fetch_who_updates()
    who_archive = update_who_archive(updates)

    st.markdown('<div class="section-head"><div><div class="section-title">📰 Latest WHO newsroom updates</div><div class="section-note">Fetched from the official WHO newsroom when the service is available.</div></div></div>', unsafe_allow_html=True)
    if updates:
        for update in updates:
            st.markdown(f'<div class="history-item"><div><div class="history-name">{update["title"]}</div><div class="history-meta">WHO Newsroom · Official source</div></div><a class="view-all" href="{update["url"]}" target="_blank">Read on WHO ↗</a></div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ WHO newsroom updates could not be loaded right now. Use the official links below.")

    current_urls = {item["url"] for item in updates}
    older_updates = [item for item in who_archive if item.get("url") not in current_urls]
    if older_updates:
        st.markdown('<div class="section-head"><div><div class="section-title">🗂️ Previously shown updates</div><div class="section-note">Saved locally so nothing gets lost once WHO refreshes their live newsroom page.</div></div></div>', unsafe_allow_html=True)
        for item in older_updates[:10]:
            st.markdown(f'<div class="history-item"><div><div class="history-name">{item["title"]}</div><div class="history-meta">First seen {item.get("first_seen", "")}</div></div><a class="view-all" href="{item["url"]}" target="_blank">Read on WHO ↗</a></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-head"><div><div class="section-title">📚 Health topic library</div><div class="section-note">Browse official WHO topic pages across prevention, conditions, health systems, and emergencies.</div></div></div>', unsafe_allow_html=True)

    topic_search = st.text_input("🔍 Search WHO topics", placeholder="Try diabetes, mental health, vaccines…")
    filtered_topics = {name: url for name, url in WHO_TOPICS.items() if not topic_search or topic_search.lower() in name.lower()}
    topic_columns = st.columns(2)
    for index, (topic_name, topic_url) in enumerate(filtered_topics.items()):
        with topic_columns[index % 2]:
            st.markdown(f'<div class="history-item"><div><div class="history-name">{topic_name}</div><div class="history-meta">Official WHO health topic</div></div><a class="view-all" href="{topic_url}" target="_blank">Open ↗</a></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-head"><div><div class="section-title">🚨 Emergency and outbreak information</div><div class="section-note">Use WHO Disease Outbreak News for confirmed acute public-health events and potential events of concern.</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="workspace"><div class="history-item"><div><div class="history-name">WHO Disease Outbreak News</div><div class="history-meta">Official outbreak reports and public-health event updates.</div></div><a class="view-all" href="https://www.who.int/emergencies/disease-outbreak-news" target="_blank">Open WHO DON ↗</a></div><div class="history-item"><div><div class="history-name">WHO Fact Sheets</div><div class="history-meta">Evidence-based summaries across health conditions and interventions.</div></div><a class="view-all" href="https://www.who.int/news-room/fact-sheets" target="_blank">Browse fact sheets ↗</a></div><div class="history-item"><div><div class="history-name">WHO Data</div><div class="history-meta">Official data, indicators, and global health observatory resources.</div></div><a class="view-all" href="https://www.who.int/data" target="_blank">Open WHO Data ↗</a></div></div>', unsafe_allow_html=True)

# --- History Page ---
if page == "History":
    usage = st.session_state.usage
    st.markdown(f'<div class="section-head"><div><div class="section-title">📋 Analysis history</div><div class="section-note">Every document you have reviewed in MedInsight AI.</div></div><div class="view-all">{usage.get("documents_analyzed", 0)} total documents</div></div>', unsafe_allow_html=True)

    history_rows = usage.get("history", [])
    if history_rows:
        history_html = '<div class="workspace">'
        for row in history_rows:
            history_html += f'<div class="history-item"><div><div class="history-name">{row.get("name", "Medical document")}</div><div class="history-meta">{row.get("time", "")}</div></div><span class="history-status">{row.get("status", "Reviewed")}</span></div>'
        history_html += '</div>'
        st.markdown(history_html, unsafe_allow_html=True)
    else:
        st.info("📂 Your reviewed documents will appear here after your first successful analysis.")

# --- Saved Insights Page ---
if page == "Saved insights":
    st.markdown('<div class="section-head"><div><div class="section-title">💡 Saved insights</div><div class="section-note">Keep important explanations close for future care conversations.</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="cards"><div class="info-card green"><span class="badge">LAB REPORT</span><div class="info-title">What to ask about cholesterol results</div><div class="info-copy">Saved from your complete blood panel analysis.</div></div><div class="info-card cream"><span class="badge">MEDICINE</span><div class="info-title">Dosage and timing notes</div><div class="info-copy">Saved from your medicine label analysis.</div></div></div>', unsafe_allow_html=True)

# --- Other Pages ---
if page2 == "Safety & privacy":
    st.markdown('<div class="section-head"><div><div class="section-title">🔒 Safety & privacy</div><div class="section-note">Understand what MedInsight AI can—and cannot—do.</div></div></div><div class="workspace"><div class="history-item"><div><div class="history-name">Not a diagnostic system</div><div class="history-meta">MedInsight AI summarizes visible document information; it does not diagnose, prescribe, or replace a clinician.</div></div></div><div class="history-item"><div><div class="history-name">Transparent uncertainty</div><div class="history-meta">When an image is unclear or information is missing, the system should flag it rather than guess.</div></div></div><div class="history-item"><div><div class="history-name">Emergency care</div><div class="history-meta">For severe or worsening symptoms, contact local emergency services or seek urgent care.</div></div></div></div>', unsafe_allow_html=True)

if page2 == "Settings":
    st.markdown('<div class="section-head"><div><div class="section-title">⚙️ Settings</div><div class="section-note">Personalize your MedInsight AI workspace.</div></div></div><div class="workspace">', unsafe_allow_html=True)
    st.checkbox("Show safety reminder before every analysis", value=True)
    st.checkbox("Keep recent analysis history", value=True)
    st.selectbox("Preferred report language", ["English", "Hindi", "Spanish", "French"])
    st.markdown('</div>', unsafe_allow_html=True)

if page2 == "Help centre":
    st.markdown('<div class="section-head"><div><div class="section-title">❓ Help centre</div><div class="section-note">Quick guidance for getting the clearest results.</div></div></div><div class="workspace"><div class="history-item"><div><div class="history-name">How do I get a better result?</div><div class="history-meta">Use a sharp, well-lit image. Make sure all text is visible and avoid glare.</div></div></div><div class="history-item"><div><div class="history-name">What documents work best?</div><div class="history-meta">Medicine labels, prescriptions, lab reports, discharge summaries, and referral notes.</div></div></div></div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown('<div class="footer">MedInsight AI · Intelligent Medical Analysis System · Built for clearer health conversations · Not intended for emergency care or medical diagnosis.</div>', unsafe_allow_html=True)
