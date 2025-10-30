import json, io
from datetime import datetime
import streamlit as st
from fpdf import FPDF
from openai import OpenAI

st.set_page_config(page_title="SOAP Notation MVP", page_icon="🩺", layout="centered")
st.title("🩺 SOAP Notation MVP")
st.caption("Text → SOAP (S/O/A/P) → PDF — ringan untuk pitching")

# --- OpenAI client (pakai Secrets di Streamlit Cloud) ---
api_key = st.secrets.get("OPENAI_API_KEY", "")
if not api_key:
    st.error("OPENAI_API_KEY belum diset di Settings → Secrets (Streamlit Cloud).")
    st.stop()

# tampilkan indikator aman (masked)
st.caption(f"🔑 OpenAI key loaded: sk-****{api_key[-6:]}")

from openai import OpenAI
client = OpenAI(api_key=api_key)

# --- Input form ---
with st.form("soap_form"):
    st.write("Tempel teks anamnesis/rawat jalan di bawah ini. (Audio bisa ditambah nanti)")
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

def llm_to_soap(raw_text: str):
    system = ("Kamu asisten dokumentasi medis. Kembalikan hasil HANYA JSON valid "
              "dengan keys persis: Subjective, Objective, Assessment, Plan. "
              "Ringkas & klinis, jangan menambah data fiktif.")
    user = f'Map teks klinis berikut menjadi SOAP. Balas HANYA JSON valid.\n\nTeks:\n""" {raw_text.strip()} """'
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        import openai as _openai
        if isinstance(e, _openai.AuthenticationError):
            st.error("Autentikasi ke OpenAI gagal. Cek kembali OPENAI_API_KEY di Settings → Secrets, lalu Restart app.")
        else:
            st.error(f"Gagal memanggil OpenAI: {e}")
        raise


def parse_soap(s: str):
    try:
        d = json.loads(s)
        return d.get("Subjective",""), d.get("Objective",""), d.get("Assessment",""), d.get("Plan",""), True
    except Exception:
        # fallback: tampilkan raw di Subjective agar tetap bisa dipitch
        return s, "", "", "", False

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

    def section(title, content):
        pdf.set_font("Arial", "B", 13); pdf.cell(0, 8, title, ln=1)
        pdf.set_font("Arial", "", 12); pdf.multi_cell(0, 7, content.strip() if content else "-"); pdf.ln(2)

    section("Subjective", S); section("Objective", O); section("Assessment", A); section("Plan", P)
    buf = io.BytesIO(); pdf.output(buf); return buf.getvalue()

if submitted:
    if not text.strip():
        st.warning("Isi teks klinis dulu ya.")
        st.stop()
    with st.spinner("Memproses ke format SOAP…"):
        raw = llm_to_soap(text)
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
