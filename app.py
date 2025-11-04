import json, io, os
from datetime import datetime
import streamlit as st
from fpdf import FPDF

import re  # tambahkan ini

# ===== PATCH 1: parser JSON =====
def extract_json_block(text: str) -> str | None:
    """Ambil blok JSON { ... } pertama yang valid dari teks mentah."""
    if not isinstance(text, str):
        return None
    txt = text.strip()
    # buang code fence ```json ... ```
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.IGNORECASE)

    # coba parse langsung
    try:
        json.loads(txt)
        return txt
    except Exception:
        pass

    # cari kurung kurawal terluar
    start = txt.find("{")
    end   = txt.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = txt[start:end+1]
        # perbaikan ringan: kutip tunggal -> ganda jika perlu
        if "'" in candidate and '"Subjective"' not in candidate:
            cand2 = candidate.replace("'", '"')
            try:
                json.loads(cand2); return cand2
            except Exception:
                pass
        try:
            json.loads(candidate); return candidate
        except Exception:
            return None
    return None

def parse_soap(s: str):
    """Parse JSON hasil model; jika gagal, fallback tetap bisa dipitch."""
    block = extract_json_block(s)
    if block:
        try:
            d = json.loads(block)
            return d.get("Subjective",""), d.get("Objective",""), d.get("Assessment",""), d.get("Plan",""), True
        except Exception:
            pass
    # fallback: taruh semua ke Subjective agar user tetap bisa edit
    return str(s), "", "", "", False


st.set_page_config(page_title="SOAP Notation MVP", page_icon="🩺", layout="centered")
st.title("🩺 SOAP Notation MVP")
st.caption("Text → SOAP (S/O/A/P) → PDF — bisa jalan GRATIS via Hugging Face")

# ===== Config secrets =====
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
HF_TOKEN = st.secrets.get("HF_TOKEN", "")  # token gratis dari huggingface.co
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
HF_MODEL = st.secrets.get("HF_MODEL", "google/gemma-2-2b-it")  # cepat & stabil utk demo

# indikator kunci yang terbaca (masked)
if OPENAI_API_KEY:
    st.caption(f"🔑 OpenAI key loaded: sk-****{OPENAI_API_KEY[-6:]}")
elif HF_TOKEN:
    st.caption(f"🤗 HF token loaded: hf-****{HF_TOKEN[-6:]}")
else:
    st.warning("Belum ada API key. Set salah satu di Secrets: OPENAI_API_KEY **atau** HF_TOKEN.")
    st.info("Rekomendasi gratis: buat akun di huggingface.co → Settings → Access Tokens → New token → paste sebagai HF_TOKEN.")
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
    system = ("Kamu asisten dokumentasi medis. Kembalikan hasil HANYA JSON valid "
              "dengan keys persis: Subjective, Objective, Assessment, Plan. "
              "Ringkas & klinis, jangan menambah data fiktif.")
    user = f'Map teks klinis berikut menjadi SOAP. Balas HANYA JSON valid.\n\nTeks:\n""" {raw_text.strip()} """'
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
    )
    return resp.choices[0].message.content.strip()

def call_huggingface(raw_text: str) -> str:
    import requests, json as _json
    system = ("Kamu asisten dokumentasi medis. Kembalikan hasil HANYA JSON valid "
              "dengan keys: Subjective, Objective, Assessment, Plan. "
              "Ringkas, klinis, dan jangan menambah data fiktif. "
              "Balas HANYA berupa JSON valid satu baris, tanpa komentar.")
    user = f'Map teks klinis berikut menjadi SOAP.\n\nTeks:\n""" {raw_text.strip()} """\n\nBalas HANYA JSON valid:'

    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "inputs": f"{system}\n\n{user}",
        "parameters": {"max_new_tokens": 500, "temperature": 0.0, "return_full_text": False},
        "options": {"use_cache": True, "wait_for_model": True}
    }
    r = requests.post(url, headers=headers, data=_json.dumps(payload), timeout=60)
    if r.status_code == 401:
        raise RuntimeError("HF 401 Unauthorized — token salah/expired. Periksa HF_TOKEN di Secrets, lalu Reboot app.")
    if r.status_code == 403:
        raise RuntimeError("HF 403 Forbidden — model butuh akses. Coba HF_MODEL = 'google/gemma-2-2b-it'.")
    if r.status_code >= 400:
        raise RuntimeError(f"HF error {r.status_code}: {r.text[:300]}")

    out = r.json()
    # Ekstrak teks generasi
    if isinstance(out, list) and out and "generated_text" in out[0]:
        return out[0]["generated_text"]
    if isinstance(out, dict) and "generated_text" in out:
        return out["generated_text"]
    # fallback: kirim apa adanya (nanti di-parse oleh extract_json_block)
    return _json.dumps(out)


def llm_to_soap(raw_text: str) -> str:
    if OPENAI_API_KEY:
        return call_openai(raw_text)
    return call_huggingface(raw_text)

def parse_soap(s: str):
    try:
        d = json.loads(s)
        return d.get("Subjective",""), d.get("Objective",""), d.get("Assessment",""), d.get("Plan",""), True
    except Exception:
        return s, "", "", "", False  # tetap bisa dipitch

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

    # KUNCI: kembalikan bytes via dest='S'
    return pdf.output(dest="S").encode("latin-1")


# ===== Run =====
if submitted:
    if not text.strip():
        st.warning("Isi teks klinis dulu ya."); st.stop()
    with st.spinner("Memproses ke format SOAP… (first use via HF bisa 10–30 dtk karena cold start)"):
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
