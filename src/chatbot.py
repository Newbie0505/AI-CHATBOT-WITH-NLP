import json, os, pickle, random

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTENTS     = os.path.join(BASE_DIR, "data", "intents.json")
MODEL_PATH  = os.path.join(BASE_DIR, "model", "chatbot_model.pkl")
VECTOR_PATH = os.path.join(BASE_DIR, "model", "vectorizer.pkl")
CONFIDENCE_THRESHOLD = 0.55

def _load_local_model():
    model      = pickle.load(open(MODEL_PATH, "rb"))
    vectorizer = pickle.load(open(VECTOR_PATH, "rb"))
    with open(INTENTS, encoding="utf-8") as f:
        intents = json.load(f)
    return model, vectorizer, intents

def _local_response(user_input, model, vectorizer, intents):
    X          = vectorizer.transform([user_input.lower()])
    tag        = model.predict(X)[0]
    confidence = float(model.predict_proba(X).max())
    if confidence < CONFIDENCE_THRESHOLD:
        return None, tag, confidence
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"]), tag, confidence
    return None, tag, confidence

def run_chatbot():
    from src.gemini_client import init_gemini, ask_gemini

    print("\n" + "=" * 55)
    print("   PyBot — NLP + Llama AI")
    print("=" * 55)
    print("  Type  'quit' or 'exit' to stop")
    print("  Type  'debug' to toggle source info")
    print("  Type  'clear' to reset memory")
    print("=" * 55)

    local_model, vectorizer, intents = _load_local_model()
    print("  [OK] Local NLP model loaded")

    try:
        ai_client        = init_gemini()
        ai_available     = True
        print("  [OK] Llama AI connected ✓")
    except Exception as e:
        ai_available = False
        print(f"  [WARN] AI not available: {e}")

    print()
    debug_mode   = False
    chat_history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBot: Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "debug":
            debug_mode = not debug_mode
            print(f"Bot: Debug mode {'ON' if debug_mode else 'OFF'}.\n")
            continue

        if user_input.lower() == "clear":
            chat_history = []
            print("Bot: Memory cleared!\n")
            continue

        if user_input.lower() in ("quit", "exit", "bye"):
            print("Bot: Goodbye! 👋\n")
            break

        response, tag, confidence = _local_response(user_input, local_model, vectorizer, intents)
        source = "local"

        if response is None and ai_available:
            try:
                response = ask_gemini(ai_client, user_input, chat_history)
                source   = "llama"
                chat_history.append({"role": "user",  "parts": [user_input]})
                chat_history.append({"role": "model", "parts": [response]})
                if len(chat_history) > 20:
                    chat_history = chat_history[-20:]
            except Exception as e:
                response = f"AI error: {e}"
                source   = "error"
        elif response is None:
            response = "I'm not sure about that. Could you rephrase?"
            source   = "fallback"

        if debug_mode:
            icon = "🤖" if source == "llama" else "📦"
            print(f"     [{icon} source={source}  intent={tag}  confidence={confidence:.2f}]")

        print(f"Bot: {response}\n")