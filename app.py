import json, re
import streamlit as st
import requests
from fpdf import FPDF
from streamlit_mic_recorder import mic_recorder

# ============================================================
#                   LOAD SECRETS (WAJIB)
# ============================================================
HF_TOKEN = st.secrets.get("HF_TOKEN", "")
HF_MODEL = st.secrets.get("HF_MODEL", "google/gemma-2-2b-it")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY belum dimasukkan ke Secrets.")
    st.stop()

# ============================================================
#              INIT OPENAI CLIENT (SETELAH SECRETS)
# ============================================================
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
#               OPENAI SPEECH TO TEXT (STABIL)
# ============================================================
def speech_to_text_openai(audio_bytes):
    try:
        result = client.audio.transcriptions.create(
            file=("audio.wav", audio_bytes, "audio/wav"),
            model="gpt-4o-mini-transcribe"
        )
        return result.text.strip()
    except Exception as e:
        raise RuntimeError(f"Gagal transkripsi OpenAI: {e}")


# ============================================================
#        Normalisasi output mic_recorder → bytes
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
#      JSON Extractor untuk output SOAP dari HF
# ============================================================
def extract_json_block(text: str):
    if not isinstance(text, str):
        return None

    txt = text.strip()
    txt = re.sub(r"```(?:json)?", "", txt).replace("```", "").strip()

    try:
        json.loads(txt)
        return txt
    except:
        pass

    start = txt.find("{")
    if start == -1:
        return None

    depth = 0
    end = None

    for i, ch in enumerate(txt[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end is None:
        return None

    candidate = txt[start:end + 1]

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
                True
            )
        except:
            pass
    return s, "", "", "", False


# ============================================================
#                        UI
# ============================================================
st.set_page_config(page_title="SOAP MVP", page_icon="🩺")

st.title("🩺 SOAP Notation MVP")
st.caption("🎤 Voice → Text → SOAP → Diagnosis → PDF")

# init session state
if "clinical_text" not in st.session_state:
    st.session_state["clinical_text"] = "Masukkan keluhan pasien di sini..."

# ============================================================
#                 REKAM & TRANSKRIPSI SUARA
# ============================================================
st.subheader("🎙 Rekam anamnesis pasien")

audio_obj = mic_recorder(
    start_prompt="🎤 Mulai rekam",
    stop_prompt="⏹ Stop",
    format="wav",
    just_once=True,
    key="mic1"
)

audio_bytes = _as_bytes(audio_obj)

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

    with st.spinner("Mengubah suara menjadi teks (OpenAI)…"):
        try:
            voice_text = speech_to_text_openai(audio_bytes)
            st.session_state["clinical_text"] = voice_text
            st.success("✔ Transkripsi selesai!")
        except Exception as e:
            st.error(f"Gagal transkripsi: {e}")

# ============================================================
#                 INPUT TEKS KLINIS
# ============================================================
st.subheader("📝 Input teks klinis")

text = st.text_area(
    "Teks klinis",
    key="clinical_text",
    height=150
)


# ============================================================
#                 GENERATE SOAP
# ============================================================
if st.button("🧠 Generate SOAP"):

    text = st.session_state["clinical_text"]

    with st.spinner("Mengubah teks → SOAP + DIAGNOSIS…"):
        url = "https://router.huggingface.co/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }

        # =====================================================
        #             SYSTEM MESSAGE SUPER INTELLIGENT
        # =====================================================
        system_prompt = """
Kamu adalah asisten medis ahli.

Tugasmu membuat SOAP NOTE dalam format JSON (Subjective, Objective, Assessment, Plan).

RULES:
- Subjective = riwayat keluhan pasien.
- Objective = temuan pemeriksaan fisik/lab jika ada.
- Assessment = DIAGNOSIS UTAMA berdasarkan data pasien.
    • Jika diagnosis jelas: tulis diagnosis definitif (contoh: Appendicitis acuta).
    • Jika masih DD: pilih diagnosis PALING MUNGKIN.
    • Jangan masukkan detail pemeriksaan ke Assessment.
    • Jangan pakai <br>, bullet, list panjang. Hanya diagnosis singkat.
- Plan = rencana terapi / pemeriksaan yang sesuai dengan diagnosis.

Output HANYA JSON valid.
"""

        payload = {
            "model": HF_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            "temperature": 0.2
        }

        r = requests.post(url, headers=headers, json=payload)
        raw = r.json()["choices"][0]["message"]["content"]

    S, O, A, P, ok = parse_soap(raw)

    st.success("SOAP berhasil dibuat!")

    # ============================================================
    #                 Tampilkan SOAP
    # ============================================================
    col1, col2 = st.columns(2)
    with col1:
        S = st.text_area("🟡 Subjective", S)
        A = st.text_area("🟣 Assessment (DIAGNOSIS)", A)
    with col2:
        O = st.text_area("🔵 Objective", O)
        P = st.text_area("🟢 Plan", P)

    st.divider()

    # ============================================================
    #                      PDF BUILDER
    # ============================================================
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "SOAP Note", ln=1)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, f"Subjective:\n{S}")
    pdf.multi_cell(0, 8, f"\nObjective:\n{O}")
    pdf.multi_cell(0, 8, f"\nAssessment (Diagnosis):\n{A}")
    pdf.multi_cell(0, 8, f"\nPlan:\n{P}")

    pdf_bytes = pdf.output(dest="S").encode("latin-1")

    st.download_button(
        "⬇ Download PDF",
        data=pdf_bytes,
        file_name="SOAP.pdf",
        mime="application/pdf"
    )
