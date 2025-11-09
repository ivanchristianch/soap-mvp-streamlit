import json, re
from datetime import datetime
import requests
import streamlit as st
from fpdf import FPDF
from streamlit_mic_recorder import mic_recorder
from huggingface_hub import InferenceClient

# =========================
# App config
# =========================
st.set_page_config(page_title="SOAP Notation MVP", page_icon="🩺", layout="centered")
st.title("🩺 SOAP Notation MVP")
st.caption("Voice/Text → SOAP (S/O/A/P) → PDF — gratis via Hugging Face")

# =========================
# Secrets / config
# =========================
HF_TOKEN      = st.secrets.get("HF_TOKEN", "")
HF_MODEL      = st.secrets.get("HF_MODEL", "google/gemma-2-2b-it")  # chat LLM di HF Router
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
OPENAI_MODEL   = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
ASR_MODEL      = "openai/whisper-base"  # opsi lain: "openai/whisper-small" / "distil-whisper/distil-small.en"

if OPENAI_API_KEY:
    st.caption(f"🔑 OpenAI key loaded: sk-****{OPENAI_API_KEY[-6:]}")
elif HF_TOKEN:
    st.caption(f"🤗 HF token loaded: hf-****{HF_TOKEN[-6:]}")
else:
    st.error("❌ Secrets kosong. Set salah satu: HF_TOKEN (disarankan) atau OPENAI_API_KEY.")
    st.stop()

# simpan hasil transkrip agar mengisi textarea otomatis
if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""

# =========================
# Utils: JSON extractor
# =========================
def extract_json_block(text: str) -> str | None:
    if not isinstance(text, str): return None
    txt = text.strip()
    # buang code fences
    txt = re.sub(r"```(?:json)?", "", txt, flags=re.IGNORECASE).replace("```", "").strip()
    # coba parse langsung
    try:
        json.loads(txt); return txt
    except Exception:
        pass
    # cari blok { ... } seimbang
    start = txt.find("{")
    if start == -1: return None
    depth, end_idx = 0, None
    for i, ch in enumerate(txt[start:], start):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = i; break
    if end_idx is None: return None
    candidate = txt[start:end_idx+1]
    for cand in (candidate, candidate.replace("'", '"')):
        try:
            json.loads(cand); return cand
        except Exception:
            continue
    return None

def parse_soap(s: str):
    block = extract_json_block(s)
    if block:
        try:
            d = json.loads(block)
            return (
                d.get("Subjective",""),
                d.get("Objective",""),
                d.get("Assessment",""),
                d.get("Plan",""),
                True,
            )
        except Exception:
            pass
    return str(s), "", "", "", False

# =========================
# Speech → text (HF InferenceClient, router baru)
# =========================
def speech_to_text(audio_bytes: bytes) -> str:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN belum diatur di Secrets.")
    client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)
    try:
        out = client.automatic_speech_recognition(audio_bytes, model=ASR_MODEL)
    except Exception as e:
        # error dari provider akan tampil jelas di sini
        raise RuntimeError(f"HF ASR failed: {e}")
    if isinstance(out, dict) and "text" in out:
        return out["text"].strip()
    if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
        return out[0]["generated_text"].strip()
    return str(out)

def _as_bytes(a):
    if a is None: return None
    if isinstance(a, (bytes, bytearray)): return a
    if isinstance(a, dict):
        if "bytes" in a and isinstance(a["bytes"], (bytes, bytearray)): return a["bytes"]
        if "audio" in a and isinstance(a["audio"], (bytes, bytearray)): return a["audio"]
    if hasattr(a, "getvalue"): return a.getvalue()
    return None

# =========================
# LLM backends (Chat → JSON SOAP)
# =========================
def call_openai(raw_text: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    system = (
        "Kamu asisten dokumentasi medis. Kembalikan hasil HANYA JSON valid "
        "dengan keys persis: Subjective, Objective, Assessment, Plan. "
        "Tidak boleh ada teks lain di luar JSON; TIDAK BOLEH pakai ``` atau penjelasan. "
        "Balas dalam SATU baris JSON, tanpa komentar, tanpa trailing comma."
    )
    user = (
        f"Teks klinis:\n\"\"\"{raw_text.strip()}\"\"\"\n"
        "Ubah menjadi JSON valid (SATU BARIS) dengan keys: Subjective, Objective, Assessment, Plan."
    )
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
    )
    return resp.choices[0].message.content.strip()

def call_huggingface(raw_text: str) -> str:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN belum diatur di Secrets.")
    system = (
        "Kamu asisten dokumentasi medis. Kembalikan hasil HANYA JSON valid "
        "dengan keys persis: Subjective, Objective, Assessment, Plan. "
        "Tidak boleh ada teks lain di luar JSON; TIDAK BOLEH pakai ``` atau penjelasan. "
        "Balas dalam SATU baris JSON, tanpa komentar, tanpa trailing comma."
    )
    user = (
        f"Teks klinis:\n\"\"\"{raw_text.strip()}\"\"\"\n"
        "Ubah menjadi JSON valid (SATU BARIS) dengan keys: Subjective, Objective, Assessment, Plan."
    )
    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "temperature": 0.0,
        "max_tokens": 600
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    if r.status_code >= 400:
        raise RuntimeError(f"HF chat error {r.status_code}: {r.text[:300]}")
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(data)

def llm_to_soap(raw_text: str) -> str:
    if OPENAI_API_KEY:
        return call_openai(raw_text)
    return call_huggingface(raw_text)

# =========================
# VOICE RECORDER (opsional)
# =========================
st.subheader("🎙️ Rekam anamnesis pasien (opsional)")
audio_obj = mic_recorder(
    start_prompt="🎤 Mulai rekam",
    stop_prompt="⏹️ Berhenti rekam",
    just_once=True,
    format="wav",
    key="mic_recorder",
)
audio_bytes = _as_bytes(audio_obj)

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    with st.spinner("Mengubah suara menjadi teks..."):
        try:
            st.session_state.voice_text = speech_to_text(audio_bytes)
            st.success("✅ Transkripsi selesai! (isi otomatis ke textarea di bawah)")
        except Exception as e:
            st.error(f"Gagal memproses audio: {e}")

st.divider()

# =========================
# INPUT FORM (text)
# =========================
with st.form("soap_form"):
    st.write("Tempel teks klinis di bawah ini **atau** gunakan hasil transkrip di atas.")
    default_text = st.session_state.voice_text or (
        "Pasien laki-laki 28 tahun demam 3 hari, pusing, mual. "
        "Tidak batuk. Suhu 38.4°C, TD 118/76, N 96, RR 20. "
        "Pertimbangan: dengue vs infeksi virus. Rencana: CBC, NS1, hidrasi, parasetamol."
    )
    text = st.text_area("Teks klinis", value=default_text, height=160)
    colA, colB = st.columns(2)
    with colA:
        patient = st.text_input("Nama pasien (opsional)", value="Demo Patient")
    with colB:
        date_str = st.text_input("Tanggal (opsional)", value=datetime.now().strftime("%Y-%m-%d"))
    submitted = st.form_submit_button("🧠 Generate SOAP")

# =========================
# PDF builder
# =========================
def build_pdf(patient, date_str, S, O, A, P) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 16); pdf.cell(0, 10, "SOAP Note (MVP)", ln=1)
    pdf.set_font("Arial", "", 12); pdf.cell(0, 8, f"Patient: {patient or '-'}   |   Date: {date_str or '-'}", ln=1); pdf.ln(2)

    def sec(title, content):
        pdf.set_font("Arial", "B", 13); pdf.cell(0, 8, title, ln=1)
        pdf.set_font("Arial", "", 12); pdf.multi_cell(0, 7, (content or "-").strip()); pdf.ln(2)

    sec("Subjective", S); sec("Objective", O); sec("Assessment", A); sec("Plan", P)
    return pdf.output(dest="S").encode("latin-1")

# =========================
# RUN
# =========================
if submitted:
    if not text.strip():
        st.warning("Isi teks klinis dulu ya."); st.stop()

    with st.spinner("Mengubah teks → SOAP…"):
        try:
            raw = llm_to_soap(text)
        except Exception as e:
            st.error(f"❌ Gagal memanggil LLM: {e}")
            st.stop()

    S, O, A, P, ok = parse_soap(raw)
    st.success("Berhasil dibuat. Silakan review:")
    c1, c2 = st.columns(2)
    with c1:
        S = st.text_area("🟡 Subjective", value=S, height=150)
        A = st.text_area("🟣 Assessment", value=A, height=150)
    with c2:
        O = st.text_area("🔵 Objective", value=O, height=150)
        P = st.text_area("🟢 Plan", value=P, height=150)

    if not ok:
        with st.expander("Raw output (bukan JSON valid)"):
            st.code(raw, language="json")

    pdf_bytes = build_pdf(patient, date_str, S, O, A, P)
    fname = f"SOAP_{(patient or 'patient').replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    st.download_button("⬇️ Download PDF", data=pdf_bytes, file_name=fname, mime="application/pdf")
