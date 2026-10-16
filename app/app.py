"""Streamlit chat UI for the SpongeBob LoRA model.  Run: streamlit run app/app.py"""
import base64, json, os, sys, threading
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
from persona import SYSTEM_PROMPT, FEW_SHOT

ADAPTER = os.environ.get("ADAPTER_PATH", str(ROOT / "models" / "spongebob-lora"))
HIST = ROOT / "data" / "chat_history.json"
BG = next(iter(sorted((ROOT / "assets").glob("background.*"))), None)  # supply your own image

def load_history():
    try:
        return json.loads(HIST.read_text(encoding="utf-8"))
    except Exception:
        return []

def save_history(msgs):
    HIST.parent.mkdir(parents=True, exist_ok=True)
    HIST.write_text(json.dumps(msgs, ensure_ascii=False), encoding="utf-8")

st.set_page_config(page_title="SpongeBob Chat (Parody)", page_icon="🧽", layout="centered")

def bg_css():
    if BG:
        b64 = base64.b64encode(BG.read_bytes()).decode()
        mime = "image/png" if BG.suffix == ".png" else "image/jpeg"
        return f"url(data:{mime};base64,{b64}) center/cover fixed"
    # themed fallback: sky-to-seabed gradient
    return "linear-gradient(180deg,#7fd6ff 0%,#2aa7e0 45%,#0d6fa8 80%,#e9d08a 100%) fixed"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;600&display=swap');
.stApp {{ background: {bg_css()}; font-family:'Fredoka',sans-serif; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.banner {{ background:#ffe14d; color:#3b2a00; border:3px solid #3b2a00; border-radius:14px;
  padding:6px 14px; text-align:center; font-size:.85rem; margin-bottom:10px; }}
h1.t {{ color:#fff; text-shadow:2px 2px 0 #0d4d75; text-align:center; margin:.2rem 0 .6rem; }}
div[data-testid="stChatMessage"] {{ border:2px solid #3b2a00; border-radius:18px; padding:10px 14px;
  background:rgba(255,255,255,.93); color:#1b1b1b; }}
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {{ background:#fff3a3; }}
div[data-testid="stChatMessage"] p {{ color:#1b1b1b; font-size:1.02rem; line-height:1.5; }}
</style>
<div class="banner">Educational AI Simulation / Parody Project — Not Affiliated with Nickelodeon.</div>
<h1 class="t">🧽 Bikini Bottom Chat</h1>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Waking up SpongeBob...")
def load_model():
    import torch
    if torch.cuda.is_available():
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(ADAPTER, max_seq_length=2048, load_in_4bit=True)
        FastLanguageModel.for_inference(model)
    else:  # slow CPU fallback
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
        model = AutoPeftModelForCausalLM.from_pretrained(ADAPTER, torch_dtype=torch.float32)
        tok = AutoTokenizer.from_pretrained(ADAPTER)
    return model, tok

def stream_reply(history, temperature, max_new):
    from transformers import TextIteratorStreamer
    model, tok = load_model()
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + FEW_SHOT + history
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kw = dict(**enc, streamer=streamer, pad_token_id=tok.pad_token_id or tok.eos_token_id, max_new_tokens=max_new, do_sample=True,
              temperature=temperature, top_p=0.9, repetition_penalty=1.25, no_repeat_ngram_size=3)
    threading.Thread(target=model.generate, kwargs=kw, daemon=True).start()
    for piece in streamer:
        yield piece

with st.sidebar:
    st.header("Settings")
    temperature = st.slider("Creativity", 0.2, 1.3, 0.5, 0.05)
    max_new = st.slider("Max reply length", 32, 400, 80, 16)
    if st.button("New chat"):
        st.session_state.messages = []; save_history([]); st.rerun()

if not Path(ADAPTER).exists():
    st.error(f"No LoRA adapter found at `{ADAPTER}`. Train one (see README) or set ADAPTER_PATH.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = load_history()
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🧽" if m["role"] == "assistant" else "🙂"):
        st.markdown(m["content"])

if prompt := st.chat_input("Say something to SpongeBob..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🙂"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="🧽"):
        reply = st.write_stream(stream_reply(st.session_state.messages[-12:], temperature, max_new))
    st.session_state.messages.append({"role": "assistant", "content": reply})
    save_history(st.session_state.messages)