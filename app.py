import streamlit as st
import json, os, pickle, random, time
from datetime import datetime
from PIL import Image
import speech_recognition as sr
import io
from src.gemini_client import init_gemini, ask_gemini, ask_with_image
from ddgs import DDGS

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INTENTS     = os.path.join(BASE_DIR, "data", "intents.json")
MODEL_PATH  = os.path.join(BASE_DIR, "model", "chatbot_model.pkl")
VECTOR_PATH = os.path.join(BASE_DIR, "model", "vectorizer.pkl")
CONFIDENCE_THRESHOLD = 0.55

st.set_page_config(
    page_title="PyBot — AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

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
    .timestamp    { font-size: 10px; color: #475569; margin-top: 4px; }
    .source-badge { font-size: 10px; color: #475569; margin-top: 4px; }
    .typing { display: flex; align-items: center; gap: 4px; padding: 12px 16px; background-color: #1e293b; border-radius: 18px; border-bottom-left-radius: 4px; width: fit-content; }
    .dot { width: 8px; height: 8px; background-color: #3b82f6; border-radius: 50%; animation: bounce 1.2s infinite; }
    .dot:nth-child(2) { animation-delay: 0.2s; }
    .dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes bounce { 0%, 60%, 100% { transform: translateY(0); } 30% { transform: translateY(-8px); } }
    h1 { color: #3b82f6 !important; }
    .online-badge { display: inline-block; width: 10px; height: 10px; background-color: #22c55e; border-radius: 50%; margin-right: 6px; }
    .mic-btn { background-color: #1e40af; color: white; border: none; border-radius: 50%; width: 50px; height: 50px; font-size: 24px; cursor: pointer; }
    .mic-btn:hover { background-color: #2563eb; }
</style>
""", unsafe_allow_html=True)

# ── Language config ───────────────────────────────────────────────
LANGUAGES = {
    "English 🇬🇧": {
        "code"        : "en-US",
        "placeholder" : "Type your message here...",
        "instruction" : "Always respond in English.",
        "clear"       : "🗑️ Clear Chat",
        "upload_label": "📷 Upload an image",
        "analyze_btn" : "🔍 Analyze Image",
        "ask_image"   : "Ask something about this image...",
        "voice_label" : "🎤 Click and speak in English",
    },
    "Hindi 🇮🇳": {
        "code"        : "hi-IN",
        "placeholder" : "अपना संदेश यहाँ लिखें...",
        "instruction" : "Always respond in Hindi (Devanagari script).",
        "clear"       : "🗑️ चैट साफ करें",
        "upload_label": "📷 छवि अपलोड करें",
        "analyze_btn" : "🔍 छवि विश्लेषण करें",
        "ask_image"   : "इस छवि के बारे में कुछ पूछें...",
        "voice_label" : "🎤 हिंदी में बोलें",
    },
    "Tamil 🇮🇳": {
        "code"        : "ta-IN",
        "placeholder" : "உங்கள் செய்தியை இங்கே தட்டச்சு செய்யுங்கள்...",
        "instruction" : "Always respond in Tamil script.",
        "clear"       : "🗑️ அரட்டையை அழி",
        "upload_label": "📷 படத்தை பதிவேற்றவும்",
        "analyze_btn" : "🔍 படத்தை பகுப்பாய்வு செய்",
        "ask_image"   : "இந்த படத்தைப் பற்றி கேளுங்கள்...",
        "voice_label" : "🎤 தமிழில் பேசுங்கள்",
    },
    "Telugu 🇮🇳": {
        "code"        : "te-IN",
        "placeholder" : "మీ సందేశాన్ని ఇక్కడ టైప్ చేయండి...",
        "instruction" : "Always respond in Telugu script.",
        "clear"       : "🗑️ చాట్ క్లియర్ చేయండి",
        "upload_label": "📷 చిత్రాన్ని అప్‌లోడ్ చేయండి",
        "analyze_btn" : "🔍 చిత్రాన్ని విశ్లేషించండి",
        "ask_image"   : "ఈ చిత్రం గురించి అడగండి...",
        "voice_label" : "🎤 తెలుగులో మాట్లాడండి",
    },
}

# ── Voice input ───────────────────────────────────────────────────
def record_voice(lang_code="en-US"):
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
        text = r.recognize_google(audio, language=lang_code)
        return text, None
    except sr.WaitTimeoutError:
        return None, "No speech detected. Try again."
    except sr.UnknownValueError:
        return None, "Could not understand. Speak clearly."
    except sr.RequestError as e:
        return None, f"Speech service error: {e}"
    except Exception as e:
        return None, f"Microphone error: {e}"

# ── Web search ────────────────────────────────────────────────────
def web_search(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if results:
                return " ".join([r['body'] for r in results])
    except:
        pass
    return None

def needs_web_search(text):
    keywords = [
        "who is", "what is the", "current", "latest", "today",
        "2024", "2025", "2026", "cm", "chief minister",
        "prime minister", "president", "ceo", "score", "news",
        "recent", "now", "winner", "election", "match", "price",
        "rate", "minister", "governor", "mayor", "chairman",
        "director", "head of", "leader", "party", "government",
        "india", "state", "country", "city", "capital", "when did",
        "how many", "population", "age of", "born", "died",
        "कौन", "क्या", "कब", "அவர்", "யார்", "ఎవరు", "ఇప్పుడు"
    ]
    return any(k in text.lower() for k in keywords)

# ── Load models ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model      = pickle.load(open(MODEL_PATH, "rb"))
    vectorizer = pickle.load(open(VECTOR_PATH, "rb"))
    with open(INTENTS, encoding="utf-8") as f:
        intents = json.load(f)
    try:
        ai_client    = init_gemini()
        ai_available = True
    except Exception as e:
        ai_client    = None
        ai_available = False
    return model, vectorizer, intents, ai_client, ai_available

model, vectorizer, intents, ai_client, ai_available = load_models()

def get_local_response(user_input):
    X          = vectorizer.transform([user_input.lower()])
    tag        = model.predict(X)[0]
    confidence = float(model.predict_proba(X).max())
    if confidence < CONFIDENCE_THRESHOLD:
        return None, tag, confidence
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"]), tag, confidence
    return None, tag, confidence

def process_input(user_input, lang):
    now = datetime.now().strftime("%I:%M %p")
    st.session_state.messages.append({
        "role": "user", "content": user_input, "timestamp": now
    })

    typing_placeholder = st.empty()
    with typing_placeholder:
        st.markdown("""
        <div class='chat-row bot'>
            <div class='avatar bot'>🤖</div>
            <div class='typing'>
                <div class='dot'></div><div class='dot'></div><div class='dot'></div>
            </div>
        </div>""", unsafe_allow_html=True)
    time.sleep(1.2)
    typing_placeholder.empty()

    response, tag, confidence = get_local_response(user_input)
    source = "📦 local NLP"

    if response is None and ai_available:
        try:
            if needs_web_search(user_input):
                web_context = web_search(user_input)
                if web_context:
                    prompt = f"""You are a helpful assistant with access to real-time web search results.
IMPORTANT: Use ONLY the following web search results to answer. Do NOT use your training data.
Do NOT say you don't have real-time access. Answer directly and confidently.
{lang["instruction"]}
Web Search Results:
{web_context}
Question: {user_input}
Answer based only on the web results above:"""
                    source = "🌐 web search + Llama AI"
                else:
                    prompt = f"{lang['instruction']}\n\n{user_input}"
                    source = "🤖 Llama AI"
            else:
                prompt = f"{lang['instruction']}\n\n{user_input}"
                source = "🤖 Llama AI"

            response = ask_gemini(ai_client, prompt, st.session_state.chat_history)
            st.session_state.chat_history.append({"role": "user",  "parts": [user_input]})
            st.session_state.chat_history.append({"role": "model", "parts": [response]})
            if len(st.session_state.chat_history) > 20:
                st.session_state.chat_history = st.session_state.chat_history[-20:]
        except Exception as e:
            response = f"AI error: {e}"
            source   = "❌ error"
    elif response is None:
        response = "I'm not sure about that. Could you rephrase?"
        source   = "❓ fallback"

    st.session_state.messages.append({
        "role"     : "bot",
        "content"  : response,
        "timestamp": datetime.now().strftime("%I:%M %p"),
        "source"   : f"{source}  |  intent: {tag}  |  confidence: {confidence:.2f}"
    })

def render_message(role, content, timestamp, source="", image=None):
    if role == "user":
        st.markdown(f"""
        <div class='chat-row user'>
            <div class='avatar user'>👤</div>
            <div>
                <div class='bubble user'>{content}</div>
                <div class='timestamp' style='text-align:right'>{timestamp}</div>
            </div>
        </div>""", unsafe_allow_html=True)
        if image:
            col = st.columns([3, 2])[1]
            with col:
                st.image(image, width=200)
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

# ── Header ────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 6])
with col1:
    st.markdown("<div style='font-size:52px; text-align:center'>🤖</div>", unsafe_allow_html=True)
with col2:
    st.title("PyBot")
    status = "✅ Llama AI + Web Search + Vision + Voice" if ai_available else "⚠️ Local NLP only"
    st.markdown(f"<span class='online-badge'></span>{status}", unsafe_allow_html=True)

st.divider()

# ── Session state ─────────────────────────────────────────────────
if "messages"     not in st.session_state: st.session_state.messages     = []
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "language"     not in st.session_state: st.session_state.language     = "English 🇬🇧"

# ── Language selector ─────────────────────────────────────────────
selected_lang = st.selectbox(
    "🌐 Select Language",
    list(LANGUAGES.keys()),
    index=list(LANGUAGES.keys()).index(st.session_state.language)
)
if selected_lang != st.session_state.language:
    st.session_state.language     = selected_lang
    st.session_state.messages     = []
    st.session_state.chat_history = []
    st.rerun()

lang = LANGUAGES[st.session_state.language]

# ── Voice input section ───────────────────────────────────────────
with st.expander("🎤 Voice Input — Speak to PyBot"):
    st.info(lang["voice_label"])
    if st.button("🎤 Start Listening", use_container_width=True):
        with st.spinner("🎤 Listening... Speak now!"):
            text, error = record_voice(lang["code"])
        if text:
            st.success(f"You said: **{text}**")
            process_input(text, lang)
            st.rerun()
        else:
            st.error(f"❌ {error}")

# ── Image upload section ──────────────────────────────────────────
with st.expander("📷 Upload Image & Ask Questions"):
    uploaded_file = st.file_uploader(
        lang["upload_label"],
        type=["jpg", "jpeg", "png", "webp"],
        key="image_uploader"
    )
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image_question = st.text_input(lang["ask_image"], key="image_question")
        if st.button(lang["analyze_btn"]) and ai_available:
            img_bytes = uploaded_file.getvalue()
            now = datetime.now().strftime("%I:%M %p")
            q = image_question if image_question else "Describe this image in detail."
            st.session_state.messages.append({
                "role": "user", "content": f"[Image] {q}",
                "timestamp": now, "image": img_bytes
            })
            try:
                response = ask_with_image(ai_client, q, img_bytes, lang["instruction"])
                source   = "🖼️ Vision AI"
            except Exception as e:
                response = f"Image analysis error: {e}"
                source   = "❌ error"
            st.session_state.messages.append({
                "role": "bot", "content": response,
                "timestamp": datetime.now().strftime("%I:%M %p"),
                "source": source
            })
            st.rerun()

st.divider()

# ── Display messages ──────────────────────────────────────────────
for msg in st.session_state.messages:
    render_message(
        msg["role"], msg["content"],
        msg["timestamp"], msg.get("source", ""),
        msg.get("image", None)
    )

# ── Text input ────────────────────────────────────────────────────
user_input = st.chat_input(lang["placeholder"])
if user_input:
    process_input(user_input, lang)
    st.rerun()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    if st.button(lang["clear"]):
        st.session_state.messages     = []
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.markdown("**🌐 Language**")
    st.write(f"Current: {st.session_state.language}")

    st.divider()
    st.markdown("**📊 Stats**")
    total    = len(st.session_state.messages)
    user_msg = len([m for m in st.session_state.messages if m["role"] == "user"])
    bot_msg  = len([m for m in st.session_state.messages if m["role"] == "bot"])
    st.write(f"Total : {total}")
    st.write(f"You   : {user_msg}")
    st.write(f"Bot   : {bot_msg}")

    st.divider()
    st.markdown("**ℹ️ How it works**")
    st.markdown("- 📦 Known topics → local NLP")
    st.markdown("- 🌐 Current events → web search")
    st.markdown("- 🤖 Everything else → Llama AI")
    st.markdown("- 🖼️ Images → Vision AI")
    st.markdown("- 🎤 Voice → Speech Recognition")
    st.markdown("- 🌐 4 languages supported")