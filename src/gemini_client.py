import os
import base64
from groq import Groq

def init_gemini():
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    api_key = None
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("GROQ_API_KEY"):
                    api_key = line.strip().split("=", 1)[1].strip()
    if not api_key:
        raise ValueError("No GROQ_API_KEY found in .env file")
    return Groq(api_key=api_key)

def ask_gemini(client, user_input, history):
    messages = [{"role": "system", "content": "You are PyBot, a friendly AI assistant built with Python. Keep answers short and conversational."}]
    for h in history:
        role = "assistant" if h["role"] == "model" else "user"
        messages.append({"role": role, "content": h["parts"][0]})
    messages.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

def ask_with_image(client, user_input, image_bytes, lang_instruction=""):
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }
                },
                {
                    "type": "text",
                    "text": f"{lang_instruction}\n\n{user_input}" if lang_instruction else user_input
                }
            ]
        }
    ]
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()