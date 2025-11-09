import json, io, os, re
from datetime import datetime
import streamlit as st
from fpdf import FPDF
import requests
from streamlit_mic_recorder import mic_recorder
from huggingface_hub import InferenceClient

# ============================================================
#            Streamlit UI - Header
# ============================================================
st.set_page_config(page_title="SOAP Notation MVP", page_icon="🩺", layout="centered")

st.title("🩺 SOAP Notation MVP")
st.caption("Voice/Text → SOAP → PDF — gratis via Hugging Face")

# ===== Secrets =====
HF_TOKEN = st.secrets.get("HF_TOKEN", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
HF_MODEL = st.secrets.get("HF_MODEL", "google/gemma-2-2b-it")  # chat LLM via HF Router
ASR_MODEL = "openai/whisper-base"  # alternatif: "openai/whisper-small" / "distil-whisper/distil-small.en"

if HF_TOKEN:
    st.caption(f"🤗 HF token loaded: hf-****{HF_TOKEN[-6:]}")
else:
    st.error("❌ Tidak ada HF_TOKEN. Tambahkan di Settings → Secrets.")
    st.stop()

st.divider()

# ============================================================
#         NORMALISASI OUTPUT mic_recorder → bytes
# ============================================================
def _as_bytes(a):
    if a is None:
        return None
    if isinstance(a, (bytes, bytearray)):
        return a
    if isinstance(a, dict):
        # streamlit-mic-recorder: {'bytes': b'...','sample_rate':16000,...}
        if "bytes" in a and isinstance(a["bytes"], (bytes, bytearray)):
            return a["bytes"]
        if "audio" in a and isinstance(a["audio"], (bytes, bytearray)):
            return a["audio"]
    if hasattr(a, "getvalue"):
        return a.getvalue()
    return None

# ============================================================
#                SPEECH RECOGNITION (WORKING)
# ============================================================
def speech_to_text(audio_bytes: bytes) -> str:
    """
    Konversi suara → teks via HF InferenceClient (router baru, stabil).
    Batasi durasi rekaman ~<=30 detik untuk tier gratis.
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN belum diatur di Secrets.")
    client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)
    try:
        out = client.automatic_speech_recognition(audio_bytes, model=ASR_MODEL)
    except Exception as e:
        # tampilkan error jelas (cold start / rate limit / dsb.)
        raise RuntimeError(f"HF ASR failed: {e}")
    # Bentuk umum whisper: {'text': '...'}
    if isinstance(out, dict) and "text" in out:
        return out["text"].strip()
    # Kadang provider lain: [{'generated_text':'...'}]
    if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
        return out[0]["generated_text"].strip()
    return str(out)

# ============================================================
#       JSON Extractor (agar output SOAP selalu valid)
# ============================================================
def extract_json_block(text: str) -> str | None:
    """Ambil blok JSON { ... } pertama yang valid; buang code fences & noise."""
    if not isinstance(text, str):
        return None
    txt = text.strip()

    # Buang code fences ```json ... ``` bila ada
    txt = re.sub(r"```(?:json)?", "", txt, flags=re.IGNORECASE)
    txt = txt.replace("```", "").strip()

    # Coba parse langsung
    try:
        json.loads(txt)
        return txt
    except Exception:
        pass

    # Cari blok { ... } seimbang
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
    if end_idx is None:
        return None

    candidate = txt[start : end_idx + 1]

    # Coba parse apa adanya, lalu versi kutip tunggal → ganda
    for cand in (candidate, candidate.replace("'", '"')):
        try:
            json.loads(cand)
            return cand
        except Exception:
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
        except Exception:
            pass
    # fallback: tampilkan raw agar bisa diedit manual
    return str(s), "", "", "", False

# ============================================================
#             🎤 VOICE RECORDER (aktif)
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
            # ============ HF LLM CALL (OpenAI-compatible) ============
            url = "https://router.huggingface.co/v1/chat/completions"
            headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
            payload = {
                "model": HF_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Return ONLY valid JSON with keys: Subjective, Objective, Assessment, Plan. "
                            "No explanation, no code fences, single-line JSON."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                "temperature": 0,
                "max_tokens": 600,
            }
            r = requests.post(url, headers=headers, json=payload, timeout=120)
            if r.status_code >= 400:
                raise RuntimeError(f"HF chat error {r.status_code}: {r.text[:300]}")
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
        mime="application/pdf",
    )
