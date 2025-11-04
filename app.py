import json, io, os, re
from datetime import datetime
import requests
import streamlit as st
from fpdf import FPDF
from streamlit_mic_recorder import mic_recorder

# ============== APP CONFIG ==============
st.set_page_config(page_title="SOAP Notation MVP", page_icon="🩺", layout="centered")
st.title("🩺 SOAP Notation MVP")
st.caption("Voice/Text → SOAP (S/O/A/P) → PDF — gratis via Hugging Face Router")

# ============== SECRETS / CONFIG ==============
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
OPENAI_MODEL   = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")

HF_TOKEN = st.secrets.get("HF_TOKEN", "")
HF_MODEL = st.secrets.get("HF_MODEL", "google/gemma-2-2b-it")  # aman & publik untuk chat
ASR_MODEL = "openai/whisper-small"  # model ASR (speech-to-text) di HF Router

if OPENAI_API_KEY:
    st.caption(f"🔑 OpenAI key loaded: sk-****{OPENAI_API_KEY[-6:]}")
elif HF_TOKEN:
    st.caption(f"🤗 HF token loaded: hf-****{HF_TOKEN[-6:]}")
else:
    st.warning("Belum ada API key. Set salah satu di Secrets: OPENAI_API_KEY **atau** HF_TOKEN.")
    st.info("Gratis: huggingface.co → Settings → Access Tokens → New token (Read) → paste sebagai HF_TOKEN.")
    st.stop()

# tempat menyimpan hasil transkrip (agar mengisi textarea otomatis)
if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""

# ============== UTIL: JSON PARSER TANGGUH ==============
def extract_json_block(text: str) -> str | None:
    """Ambil blok JSON { ... } pertama yang valid dari teks mentah."""
    if not isinstance(text, str):
        return None
    txt = text.strip()

    # hapus code fences ```json ... ```
    txt = re.sub(r"```(?:json)?", "", txt, flags=re.IGNORECASE).replace("```", "").strip()

    # coba parse langsung
    try:
        json.loads(txt)
        return txt
    except Exception:
        pass

    # cari kurung kurawal seimbang
    start = txt.find("{")
    if start == -1:
        return None
    depth, end_idx = 0, None
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

    candidate = txt[start:end_idx + 1]
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
                d.get("Subjective", ""),
                d.get("Objective", ""),
                d.get("Assessment", ""),
                d.get("Plan", ""),
                True,
            )
        except Exception:
            pass
    return str(s), "", "", "", False

# ============== SPEECH → TEXT (HF Router) ==============
def speech_to_text(audio_bytes: bytes) -> str:
    """Ubah audio (WAV/MP3) menjadi teks via Hugging Face Router (Whisper)."""
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN belum diatur di Secrets.")
    url = "https://router.huggingface.co/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    data = {"model": ASR_MODEL}
    r = requests.post(url, headers=headers, files=files, data=data, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"HF ASR error {r.status_code}: {r.text[:300]}")
    return r.json().get("text", "")

def _as_bytes(a):
    """Normalisasi keluaran mic_recorder menjadi bytes (WAV)."""
    if a is None: return None
    if isinstance(a, (bytes, bytearray)): return a
    if isinstance(a, dict):
        if "bytes" in a and isinstance(a["bytes"], (bytes, bytearray)): return a["bytes"]
        if "audio" in a and isinstance(a["audio"], (bytes, bytearray)): return a["audio"]
    if hasattr(a, "getvalue"): return a.getvalue()
    return None

# ============== LLM BACKENDS ==============
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
    st.caption(f"🔗 HF endpoint: {url} (model={HF_MODEL})")
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
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

# ============== VOICE RECORDER (opsional) ==============
st.subheader("🎙️ Rekam anamnesis pasien (opsional)")
audio_obj = mic_recorder(
    start_prompt="🎤 Mulai rekam",
    stop_prompt="⏹️ Berhenti rekam",
    just_once=True,
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

# ============== INPUT FORM (TEXT) ==============
with st.form("soap_form"):
    st.write("Tempel teks anamnesis/rawat jalan di bawah ini **atau** gunakan hasil transkrip dari atas.")
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

# ============== PDF BUILDER ==============
def build_pdf(patient, date_str, S, O, A, P) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "SOAP Note (MVP)", ln=1)

    pdf.set_font("Arial", "", 12)
    header = f"Patient: {patient or '-'}   |   Date: {date_str or '-'}"
    pdf.cell(0, 8, header, ln=1)
    pdf.ln(2)

    def sec(title, content):
        pdf.set_font("Arial", "B", 13); pdf.cell(0, 8, title, ln=1)
        pdf.set_font("Arial", "", 12); pdf.multi_cell(0, 7, (content or '-').strip()); pdf.ln(2)

    sec("Subjective", S); sec("Objective", O); sec("Assessment", A); sec("Plan", P)
    return pdf.output(dest="S").encode("latin-1")

# ============== RUN SUBMISSION ==============
if submitted:
    if not text.strip():
        st.warning("Isi teks klinis dulu ya."); st.stop()

    with st.spinner("Memproses ke format SOAP… (HF first-run bisa 10–30 dtk)"):
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
