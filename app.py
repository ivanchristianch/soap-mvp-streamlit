import json, io, os, re
from datetime import datetime
import streamlit as st
from fpdf import FPDF
import requests
from streamlit_mic_recorder import mic_recorder

# ============================================================
#                FIXED SPEECH RECOGNITION (WORKING)
# ============================================================
def speech_to_text(audio_bytes):
    """
    Konversi suara → teks pakai model Whisper (via HF Inference API)
    Endpoint stabil & gratis.
    """
    HF_ASR_MODEL = "openai/whisper-base"
    url = f"https://api-inference.huggingface.co/models/{HF_ASR_MODEL}"

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}

    r = requests.post(url, headers=headers, files=files, timeout=60)

    if r.status_code == 503:
        return "Model sedang loading, tunggu 10 detik dan coba lagi."

    if r.status_code >= 400:
        raise RuntimeError(f"HF ASR Error {r.status_code}: {r.text}")

    out = r.json()
    return out.get("text", "").strip()


# ============================================================
#         NORMALISASI OUTPUT mic_recorder → bytes
# ============================================================
def _as_bytes(a):
    if a is None:
        return None
    if isinstance(a, (bytes, bytearray)):
        return a
    if isinstance(a, dict):
        if "bytes" in a:
            return a["bytes"]
        if "audio" in a:
            return a["audio"]
    if hasattr(a, "getvalue"):
        return a.getvalue()
    return None


# ============================================================
#       JSON Extractor (agar output SOAP selalu valid)
# ============================================================
def extract_json_block(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    txt = text.strip()
    txt = re.sub(r"```(?:json)?", "", txt, flags=re.IGNORECASE)
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
            continue

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
#            Streamlit UI - Header
# ============================================================
st.set_page_config(page_title="SOAP Notation MVP", page_icon="🩺", layout="centered")

st.title("🩺 SOAP Notation MVP")
st.caption("Voice/Text → SOAP → PDF — gratis via HuggingFace")

# ===== Secrets =====
HF_TOKEN = st.secrets.get("HF_TOKEN", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
HF_MODEL = st.secrets.get("HF_MODEL", "google/gemma-2-2b-it")

if HF_TOKEN:
    st.caption(f"🤗 HF token loaded: hf-****{HF_TOKEN[-6:]}")
else:
    st.error("❌ Tidak ada HF_TOKEN. Tambahkan di Secrets.")
    st.stop()

st.divider()


# ============================================================
#             🎤 VOICE RECORDER (FIXED & WORKING)
# ============================================================
st.subheader("🎙️ Rekam anamnesis pasien (opsional)")

audio_obj = mic_recorder(
    start_prompt="🎤 Mulai rekam",
    stop_prompt="⏹️ Stop",
    just_once=True,
    format="wav",
    key="mic1",
)

audio_bytes = _as_bytes(audio_obj)

voice_text = None

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

    with st.spinner("Mengubah suara menjadi teks..."):
        try:
            voice_text = speech_to_text(audio_bytes)
            st.success("✔️ Transkripsi selesai")
        except Exception as e:
            st.error(f"Gagal memproses audio: {e}")


# ============================================================
#               TEXT INPUT (from voice OR manual)
# ============================================================
st.subheader("📝 Input teks klinis")

default_text = (
    voice_text
    if voice_text else
    "Pasien laki-laki 28 tahun demam 3 hari, pusing, mual. Tidak batuk..."
)

text = st.text_area("Teks klinis", value=default_text, height=150)


# ============================================================
#               GENERATE SOAP BUTTON
# ============================================================
if st.button("🧠 Generate SOAP"):

    with st.spinner("Mengubah teks → SOAP..."):
        try:
            # ============ HF LLM CALL ============
            url = "https://router.huggingface.co/v1/chat/completions"
            headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
            payload = {
                "model": HF_MODEL,
                "messages": [
                    {"role": "system", "content": "Return ONLY valid JSON with keys Subjective, Objective, Assessment, Plan."},
                    {"role": "user", "content": text}
                ],
                "temperature": 0
            }

            r = requests.post(url, headers=headers, json=payload)
            data = r.json()
            raw = data["choices"][0]["message"]["content"]

        except Exception as e:
            st.error(f"LLM error: {e}")
            st.stop()

    S, O, A, P, ok = parse_soap(raw)
    st.success("SOAP berhasil dibuat!")

    col1, col2 = st.columns(2)
    with col1:
        S = st.text_area("🟡 Subjective", S, height=140)
        A = st.text_area("🟣 Assessment", A, height=140)
    with col2:
        O = st.text_area("🔵 Objective", O, height=140)
        P = st.text_area("🟢 Plan", P, height=140)

    st.divider()

    # ========== PDF Builder ==========
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "SOAP Note", ln=1)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, f"Subjective:\n{S}")
    pdf.ln(2)
    pdf.multi_cell(0, 8, f"Objective:\n{O}")
    pdf.ln(2)
    pdf.multi_cell(0, 8, f"Assessment:\n{A}")
    pdf.ln(2)
    pdf.multi_cell(0, 8, f"Plan:\n{P}")

    pdf_bytes = pdf.output(dest="S").encode("latin-1")

    st.download_button(
        "⬇️ Download PDF",
        data=pdf_bytes,
        file_name="SOAP.pdf",
        mime="application/pdf"
    )
