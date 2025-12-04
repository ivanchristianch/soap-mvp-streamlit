import json, io, os, re
from datetime import datetime
import streamlit as st
from fpdf import FPDF
import requests
from streamlit_mic_recorder import mic_recorder

# ============================================================
#                   STREAMLIT CONFIG
# ============================================================
st.set_page_config(page_title="SOAP MVP", page_icon="🩺")

st.title("🩺 SOAP Notation MVP")
st.caption("🎤 Voice → Text → SOAP → PDF (OpenAI + HuggingFace)")

# ============================================================
#                   LOAD SECRETS
# ============================================================
HF_TOKEN = st.secrets.get("HF_TOKEN", "")
HF_MODEL = st.secrets.get("HF_MODEL", "google/gemma-2-2b-it")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY belum dimasukkan ke Secrets.")
    st.stop()

# ============================================================
#                   INIT OPENAI CLIENT (FIXED)
# ============================================================
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
#               OPENAI SPEECH-TO-TEXT (FIXED)
# ============================================================
def speech_to_text_openai(audio_bytes):
    try:
        result = client.audio.transcriptions.create(
            model="gpt-4o-transcribe-1",
            file=("audio.wav", audio_bytes, "audio/wav")
        )
        return result.text.strip()
    except Exception as e:
        raise RuntimeError(f"Gagal transkripsi OpenAI: {e}")

# ============================================================
#         NORMALISASI OUTPUT mic_recorder → bytes
# ============================================================
def _as_bytes(a):
    if a is None:
        return None
    if isinstance(a, (bytes, bytearray)):
        return a
    if isinstance(a, dict):
        return a.get("bytes") or a.get("audio")
    if hasattr(a, "getvalue"):
        return a.getvalue()
    return None

# ============================================================
#        JSON Extractor (agar output SOAP selalu valid)
# ============================================================
def extract_json_block(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    txt = text.strip()

    txt = re.sub(r"```(?:json)?", "", txt)
    txt = txt.replace("```", "").strip()

    try:
        json.loads(txt)
        return txt
    except:
        pass

    start = txt.find("{")
    if start == -1:
        return None

    depth = 0
    end_idx = None

    for i, ch in enumerate(txt[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = i
                break

    if not end_idx:
        return None

    candidate = txt[start:end_idx+1]

    for cand in (candidate, candidate.replace("'", '"')):
        try:
            json.loads(cand)
            return cand
        except:
            pass

    return None

def parse_soap(s: str):
    block = extract_json_block(s)
    if block:
        try:
            d = json.loads(block)
            return (
                d.get("Subjective", ""),
                d.get("Objective", ""),
                d.get("Assessment", ""),
                d.get("Plan", ""),
                True,
            )
        except:
            pass
    return str(s), "", "", "", False

# ============================================================
#             🎤 VOICE RECORDING
# ============================================================
st.subheader("🎙 Rekam anamnesis pasien")

audio_obj = mic_recorder(
    start_prompt="🎤 Mulai rekam",
    stop_prompt="⏹ Stop",
    format="wav",
    just_once=True,
    key="mic1",
)

audio_bytes = _as_bytes(audio_obj)
voice_text = None

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

    with st.spinner("Mengubah suara menjadi teks (OpenAI)"):
