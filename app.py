import os
import time
from datetime import datetime
from PIL import Image
import streamlit as st
import speech_recognition as sr
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

# --- Localization & Translation Mappings ---
LANGUAGES = {
    "English 🇬🇧": {
        "code": "en-US",
        "placeholder": "Type your message here...",
        "instruction": "Always respond in English.",
        "clear": "🗑️ Clear Chat",
        "upload_label": "📷 Upload an image",
        "analyze_btn": "🔍 Analyze Image",
        "ask_image": "Ask something about this image...",
        "voice_label": "🎤 Click and speak in English",
    },
    "Hindi 🇮🇳": {
        "code": "hi-IN",
        "placeholder": "अपना संदेश यहाँ लिखें...",
        "instruction": "Always respond in Hindi (Devanagari script).",
        "clear": "🗑️ चैट साफ करें",
        "upload_label": "📷 छवि अपलोड करें",
        "analyze_btn": "🔍 छवि विश्लेषण करें",
        "ask_image": "इस छवि के बारे में कुछ पूछें...",
        "voice_label": "🎤 हिंदी में बोलें",
    },
    "Tamil 🇮🇳": {
        "code": "ta-IN",
        "placeholder": "உங்கள் செய்தியை இங்கே தட்டச்சு செய்யுங்கள்...",
        "instruction": "Always respond in Tamil script.",
        "clear": "🗑️ அரட்டையை அழி",
        "upload_label": "📷 படத்தை பதிவேற்றவும்",
        "analyze_btn": "🔍 படத்தை பகுப்பாய்வு செய்",
        "ask_image": "இந்த படத்தைப் பற்றி கேளுங்கள்...",
        "voice_label": "🎤 தமிழில் பேசுங்கள்",
    },
    "Telugu 🇮🇳": {
        "code": "te-IN",
        "placeholder": "మీ సందేశాన్ని ఇక్కడ టైప్ చేయండి...",
        "instruction": "Always respond in Telugu script.",
        "clear": "🗑️ చాట్ క్లియర్ చేయండి",
        "upload_label": "📷 చిత్రాన్ని అప్‌లోడ్ చేయండి",
        "analyze_btn": "🔍 చిత్రాన్ని విశ్లేషించండి",
        "ask_image": "ఈ చిత్రం గురించి అడగండి...",
        "voice_label": "🎤 తెలుగులో మాట్లాడండి",
    }
}

# --- Audio Handler ---
def record_voice(lang_code="en-US"):
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        text = recognizer.recognize_google(audio, language=lang_code)
        return text, None
    except sr.WaitTimeoutError:
        return None, "No speech detected. Please speak closer to your microphone."
    except sr.UnknownValueError:
        return None, "Audio was unclear. Please repeat or speak slower."
    except Exception as e:
        return None, f"Hardware/Service error: {str(e)}"

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

def process_chat_interaction(user_input, language_config):
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

    source_label = "🧠 Core Gemini Engine"
    
    if ai_available:
        try:
            # Using universally resolved fallback string mapping to guarantee initialization compatibility
            model = genai.GenerativeModel("gemini-pro")
            search_context = run_web_search(user_input)
            
            if search_context:
                prompt = f"Context:\n{search_context}\n\nInstruction: {language_config['instruction']}\nUser: {user_input}"
                source_label = "🌐 Web Integration Engine"
            else:
                prompt = f"Instruction: {language_config['instruction']}\nUser: {user_input}"

            response = model.generate_content(prompt).text
        except Exception as err:
            response = f"Engine Gateway Timeout: {str(err)}"
            source_label = "❌ Stack Error"
    else:
        response = "API Key configuration missing. Please add GEMINI_API_KEY into Advanced Settings -> Secrets."
        source_label = "⚠️ System Warning"

    st.session_state.messages.append({
        "role": "bot",
        "content": response,
        "timestamp": datetime.now().strftime("%I:%M %p"),
        "source": source_label
    })

def render_ui_message(role, content, timestamp, source="", uploaded_img=None):
    if role == "user":
        st.markdown(f"""
        <div class='chat-row user'>
            <div class='avatar user'>👤</div>
            <div>
                <div class='bubble user'>{content}</div>
                <div class='timestamp' style='text-align:right'>{timestamp}</div>
            </div>
        </div>""", unsafe_allow_html=True)
        if uploaded_img:
            layout_col = st.columns([3, 2])[1]
            with layout_col:
                st.image(uploaded_img, width=200)
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
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown("<div style='font-size:52px; text-align:center; padding-top:10px;'>🤖</div>", unsafe_allow_html=True)
with col_title:
    st.title("PyBot Ecosystem")
    app_status = "✅ Gemini Engine Online" if ai_available else "⚠️ API Secret Key Missing"
    st.markdown(f"<span class='online-badge'></span>{app_status}", unsafe_allow_html=True)

st.divider()

# --- State Instantiation ---
if "messages" not in st.session_state: st.session_state.messages = []
if "language" not in st.session_state: st.session_state.language = "English 🇬🇧"

# --- Control Panel Setup ---
selected_lang = st.selectbox(
    "🌐 Localization Configuration",
    list(LANGUAGES.keys()),
    index=list(LANGUAGES.keys()).index(st.session_state.language)
)
if selected_lang != st.session_state.language:
    st.session_state.language = selected_lang
    st.session_state.messages = []
    st.rerun()

lang_config = LANGUAGES[st.session_state.language]

# --- Panels (Voice & Vision Layout Panels) ---
with st.expander("🎤 Voice Synthesizer Capture"):
    st.info(lang_config["voice_label"])
    if st.button("🔴 Initialize Microphone Stream", use_container_width=True):
        with st.spinner("Streaming incoming audio buffer..."):
            captured_speech, recording_error = record_voice(lang_config["code"])
        if captured_speech:
            st.success(f"Parsed Stream String: **{captured_speech}**")
            process_chat_interaction(captured_speech, lang_config)
            st.rerun()
        else:
            st.error(f"Capture Fault: {recording_error}")

with st.expander("📷 Vision Object Analytics"):
    image_asset = st.file_uploader(
        lang_config["upload_label"],
        type=["jpg", "jpeg", "png", "webp"]
    )
    if image_asset:
        raw_img = Image.open(image_asset)
        st.image(raw_img, caption="Staged Vision Source Object", use_container_width=True)
        query_input = st.text_input(lang_config["ask_image"])
        
        if st.button(lang_config["analyze_btn"]) and ai_available:
            time_stamp = datetime.now().strftime("%I:%M %p")
            final_query = query_input if query_input else "Describe this image in detail."
            
            st.session_state.messages.append({
                "role": "user", "content": f"[Vision Query] {final_query}",
                "timestamp": time_stamp, "image": image_asset.getvalue()
            })
            try:
                v_model = genai.GenerativeModel("gemini-pro-vision")
                ai_response = v_model.generate_content([final_query, raw_img]).text
                vision_source = "🖼️ Gemini Vision Engine"
            except Exception as ex:
                ai_response = f"Vision Stack Fault: {str(ex)}"
                vision_source = "❌ Execution Error"
                
            st.session_state.messages.append({
                "role": "bot", "content": ai_response,
                "timestamp": datetime.now().strftime("%I:%M %p"),
                "source": vision_source
            })
            st.rerun()

st.divider()

# --- Main Message Render Feed ---
for msg in st.session_state.messages:
    render_ui_message(
        msg["role"], msg["content"],
        msg["timestamp"], msg.get("source", ""),
        msg.get("image", None)
    )

# --- Standard Chat Input ---
chat_text = st.chat_input(lang_config["placeholder"])
if chat_text:
    process_chat_interaction(chat_text, lang_config)
    st.rerun()

# --- Admin Sidebar Panels ---
with st.sidebar:
    st.markdown("### ⚙️ Engine Diagnostics")
    if st.button(lang_config["clear"], use_container_width=True):
        st.session_state.messages = []
        st.rerun()