import os
import time
from datetime import datetime
import streamlit as st
from duckduckgo_search import DDGS
import google.generativeai as genai

# --- Page Config ---
st.set_page_config(
    page_title="PyBot — AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# --- UI Custom Styling ---
st.markdown("""
<style>
    .stApp { background-color: #0f172a; }
    .chat-row { display: flex; align-items: flex-end; margin: 10px 0; gap: 10px; }
    .chat-row.user { flex-direction: row-reverse; }
    .avatar { width: 38px; height: 38px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 20px; flex-shrink: 0; }
    .avatar.bot  { background-color: #1e40af; }
    .avatar.user { background-color: #7c3aed; }
    .bubble { padding: 12px 16px; border-radius: 18px; max-width: 70%; word-wrap: break-word; }
    .bubble.bot  { background-color: #1e293b; color: #e2e8f0; border-bottom-left-radius: 4px; }
    .bubble.user { background-color: #1e40af; color: white; border-bottom-right-radius: 4px; text-align: right; }
    .timestamp, .source-badge { font-size: 10px; color: #475569; margin-top: 4px; }
    .typing { display: flex; align-items: center; gap: 4px; padding: 12px 16px; background-color: #1e293b; border-radius: 18px; border-bottom-left-radius: 4px; width: fit-content; }
    .dot { width: 8px; height: 8px; background-color: #3b82f6; border-radius: 50%; animation: bounce 1.2s infinite; }
    .dot:nth-child(2) { animation-delay: 0.2s; }
    .dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes bounce { 0%, 60%, 100% { transform: translateY(0); } 30% { transform: translateY(-8px); } }
    h1 { color: #3b82f6 !important; }
    .online-badge { display: inline-block; width: 10px; height: 10px; background-color: #22c55e; border-radius: 50%; margin-right: 6px; }
</style>
""", unsafe_allow_html=True)

# --- Fetch API Key safely from Streamlit Secrets ---
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    ai_available = True
else:
    ai_available = False

# --- Web Search Utility ---
def run_web_search(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                return "\n".join([f"- {r['body']}" for r in results])
    except Exception:
        pass
    return None

def process_chat_interaction(user_input):
    timestamp_now = datetime.now().strftime("%I:%M %p")
    st.session_state.messages.append({
        "role": "user", "content": user_input, "timestamp": timestamp_now
    })

    typing_box = st.empty()
    with typing_box:
        st.markdown("""
        <div class='chat-row bot'>
            <div class='avatar bot'>🤖</div>
            <div class='typing'>
                <div class='dot'></div><div class='dot'></div><div class='dot'></div>
            </div>
        </div>""", unsafe_allow_html=True)
    time.sleep(0.6)
    typing_box.empty()

    source_label = "🧠 Core Gemini Model"
    
    if ai_available:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            search_context = run_web_search(user_input)
            
            if search_context:
                prompt = f"Context:\n{search_context}\n\nUser: {user_input}"
                source_label = "🌐 Web Integration Engine"
            else:
                prompt = user_input

            response = model.generate_content(prompt).text
        except Exception as err:
            response = f"API Error: {str(err)}"
            source_label = "❌ Engine Error"
    else:
        response = "API Key configuration missing! Please add your key to Streamlit Advanced Settings -> Secrets."
        source_label = "⚠️ System Warning"

    st.session_state.messages.append({
        "role": "bot",
        "content": response,
        "timestamp": datetime.now().strftime("%I:%M %p"),
        "source": source_label
    })

def render_ui_message(role, content, timestamp, source=""):
    if role == "user":
        st.markdown(f"""
        <div class='chat-row user'>
            <div class='avatar user'>👤</div>
            <div>
                <div class='bubble user'>{content}</div>
                <div class='timestamp' style='text-align:right'>{timestamp}</div>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='chat-row bot'>
            <div class='avatar bot'>🤖</div>
            <div>
                <div class='bubble bot'>{content}</div>
                <div class='source-badge'>{source}</div>
                <div class='timestamp'>{timestamp}</div>
            </div>
        </div>""", unsafe_allow_html=True)

# --- View Layout Structure ---
st.title("PyBot Ecosystem")
app_status = "✅ Gemini Engine Online" if ai_available else "⚠️ API Secret Key Missing"
st.markdown(f"<span class='online-badge'></span>{app_status}", unsafe_allow_html=True)
st.divider()

if "messages" not in st.session_state: st.session_state.messages = []

# --- Main Message Render Feed ---
for msg in st.session_state.messages:
    render_ui_message(msg["role"], msg["content"], msg["timestamp"], msg.get("source", ""))

# --- Standard Chat Input ---
chat_text = st.chat_input("Type your message here...")
if chat_text:
    process_chat_interaction(chat_text)
    st.rerun()

with st.sidebar:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()