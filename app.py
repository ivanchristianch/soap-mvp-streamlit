import json, io, os, re
from datetime import datetime
import streamlit as st
from fpdf import FPDF
import requests
from streamlit_mic_recorder import mic_recorder

st.divider()
st.subheader("🎙️ Rekam anamnesis pasien (opsional)")

audio_bytes = mic_recorder(
    start_prompt="🎤 Mulai rekam",
    stop_prompt="⏹️ Berhenti rekam",
    just_once=True,
    key="mic_recorder"
)

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    with st.spinner("Mengubah suara menjadi teks..."):
        try:
            voice_text = speech_to_text(audio_bytes)
            st.success("✅ Transkripsi selesai!")
            st.write(voice_text)

            # Optional: konversi langsung ke SOAP
            if st.button("🧠 Generate SOAP dari suara"):
                raw = llm_to_soap(voice_text)
                S, O, A, P, ok = parse_soap(raw)
                st.success("Berhasil dibuat dari rekaman suara!")
                st.write(f"Subjective: {S}\nObjective: {O}\nAssessment: {A}\nPlan: {P}")
        except Exception as e:
            st.error(f"Gagal memproses audio: {e}")
def speech_to_text(audio_bytes):
    """Ubah suara (WAV/MP3) jadi teks pakai model whisper kecil di Hugging Face"""
    url = "https://router.huggingface.co/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    data = {"model": "openai/whisper-small"}
    r = requests.post(url, headers=headers, files=files, data=data)
    r.raise_for_status()
    out = r.json()
    return out.get("text", "")
# ====== Parser JSON yang kuat ======
def extract_json_block(text: str) -> str | None:
    """Ambil blok JSON { ... } pertama yang valid dari teks mentah."""
    if not isinstance(text, str):
        return None
    txt = text.strip()

    # buang semua code fences ```json / ```
    txt = re.sub(r"```(?:json)?", "", txt, flags=re.IGNORECASE)
    txt = txt.replace("```", "").strip()

    # coba parse langsung
    try:
        json.loads(txt)
        return txt
    except Exception:
        pass

    # scan kurung kurawal seimbang dari { pertama
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

    # coba parse apa adanya, lalu versi kutip tunggal -> ganda
    for cand in (candidate, candidate.replace("'", '"')):
        try:
            json.loads(cand)
            return cand
        except Exception:
            continue
    return None

def parse_soap(s: str):
    """Parse JSON hasil model; fallback tetap bisa dipitch."""
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

# ====== UI dasar ======
st.set_page_config(page_title="SOAP Notation MVP", page_icon="🩺", layout="centered")
st.title("🩺 SOAP Notation MVP")
st.caption("Text → SOAP (S/O/A/P) → PDF — bisa jalan GRATIS via Hugging Face")

# ===== Config secrets =====
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
HF_TOKEN = st.secrets.get("HF_TOKEN", "")  # token gratis dari huggingface.co
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
HF_MODEL = st.secrets.get("HF_MODEL", "google/gemma-2-2b-it")  # aman & publik

# indikator kunci yang terbaca (masked)
if OPENAI_API_KEY:
    st.caption(f"🔑 OpenAI key loaded: sk-****{OPENAI_API_KEY[-6:]}")
elif HF_TOKEN:
    st.caption(f"🤗 HF token loaded: hf-****{HF_TOKEN[-6:]}")
else:
    st.warning("Belum ada API key. Set salah satu di Secrets: OPENAI_API_KEY **atau** HF_TOKEN.")
    st.info("Gratis: huggingface.co → Settings → Access Tokens → New token (Read) → paste sebagai HF_TOKEN.")
    st.stop()

# ===== Input form =====
with st.form("soap_form"):
    st.write("Tempel teks anamnesis/rawat jalan di bawah ini.")
    default_text = (
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

# ===== LLM backends =====
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
    import requests, json as _json
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

    # Router HF - OpenAI-compatible endpoint
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

    st.caption(f"🔗 HF endpoint: {url} (model={HF_MODEL})")  # debug
    r = requests.post(url, headers=headers, data=_json.dumps(payload), timeout=60)
    if r.status_code == 401:
        raise RuntimeError("HF 401 Unauthorized — token salah/expired. Periksa HF_TOKEN di Secrets, lalu Reboot app.")
    if r.status_code == 404:
        raise RuntimeError("HF 404 — model tidak ditemukan pada router. Cek HF_MODEL (contoh aman: google/gemma-2-2b-it).")
    if r.status_code >= 400:
        raise RuntimeError(f"HF error {r.status_code}: {r.text[:300]}")

    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return _json.dumps(data)

def llm_to_soap(raw_text: str) -> str:
    if OPENAI_API_KEY:
        return call_openai(raw_text)
    return call_huggingface(raw_text)

# ===== PDF builder =====
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
        pdf.set_font("Arial", "", 12); pdf.multi_cell(0, 7, (content or "-").strip()); pdf.ln(2)

    sec("Subjective", S); sec("Objective", O); sec("Assessment", A); sec("Plan", P)
    return pdf.output(dest="S").encode("latin-1")

# ===== Run =====
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
