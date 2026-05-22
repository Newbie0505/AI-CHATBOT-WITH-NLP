import os
import google.generativeai as genai

def init_gemini():
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    api_key = None
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("GEMINI_API_KEY"):
                    api_key = line.strip().split("=", 1)[1].strip()
    if not api_key:
        raise ValueError("No GEMINI_API_KEY found in .env file")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction="You are PyBot, a friendly AI assistant built with Python. Keep answers short and conversational. Never say you are Google Gemini."
    )
    return model

def ask_gemini(model, user_input, history):
    chat = model.start_chat(history=history)
    response = chat.send_message(user_input)
    return response.text.strip()