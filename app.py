import json, io, os, re
from datetime import datetime
import streamlit as st
from fpdf import FPDF
import requests
from streamlit_mic_recorder import mic_recorder

# ============================================================
#                 OPENAI SPEECH-TO-TEXT (STABLE)
# ============================================================
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

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

    # buang code fences
    txt = re.sub(r"```(?:json)?", "", txt)
    txt = txt.replace("```", "").strip()

    # coba parse langsung
    try:
        json.loads(txt)
        return txt
    except:
        pass

    # cari kurung kurawal pertama
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

    # test JSON
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
#                STREAMLIT UI
# ============================================================
st.set_page_config(page_title="SOAP MVP", page_icon="🩺")

st.title("🩺 SOAP Notation MVP")
st.caption("🎤 Voice → Text → SOAP → PDF (OpenAI + HuggingFace)")

# Load secrets
HF_TOKEN = st.secrets.get("HF_TOKEN", "")
HF_MODEL = st.secrets.get("HF_MODEL", "google/gemma-2-2b-it")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY belum dimasukkan ke Secrets.")
    st.stop()

# ============================================================
#             🎤 VOICE RECORDING (WORKING 100%)
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

    with st.spinner("Mengubah suara menjadi teks (OpenAI)…"):
        try:
            voice_text = speech_to_text_openai(audio_bytes)
            st.success("✔ Transkripsi selesai!")
        except Exception as e:
            st.error(f"Gagal transkripsi: {e}")


# ============================================================
#          INPUT TEKS (dari suara atau manual)
# ============================================================
st.subheader("📝 Input teks klinis")

default_text = voice_text if voice_text else "Pasien laki-laki 28 tahun demam 3 hari..."

text = st.text_area("Teks klinis", value=default_text, height=150)


# ============================================================
#               GENERATE SOAP VIA HF
# ============================================================
if st.button("🧠 Generate SOAP"):
    with st.spinner("Mengubah teks → SOAP…"):
        url = "https://router.huggingface.co/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": HF_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "Return ONLY valid JSON with keys Subjective, Objective, Assessment, Plan."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            "temperature": 0
        }

        r = requests.post(url, headers=headers, json=payload)
        raw = r.json()["choices"][0]["message"]["content"]

    S, O, A, P, ok = parse_soap(raw)

    st.success("SOAP berhasil dibuat!")

    col1, col2 = st.columns(2)
    with col1:
        S = st.text_area("🟡 Subjective", S)
        A = st.text_area("🟣 Assessment", A)
    with col2:
        O = st.text_area("🔵 Objective", O)
        P = st.text_area("🟢 Plan", P)

    st.divider()

    # PDF builder
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "SOAP Note", ln=1)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, f"Subjective:\n{S}")
    pdf.multi_cell(0, 8, f"\nObjective:\n{O}")
    pdf.multi_cell(0, 8, f"\nAssessment:\n{A}")
    pdf.multi_cell(0, 8, f"\nPlan:\n{P}")

    pdf_bytes = pdf.output(dest="S").encode("latin-1")

    st.download_button(
        "⬇ Download PDF",
        data=pdf_bytes,
        file_name="SOAP.pdf",
        mime="application/pdf"
    )
