AI Chatbot with NLP
A lightweight, terminal-based chatbot built in Python that actually understands what you're asking. It uses NLTK for text processing and scikit-learn’s Multinomial Naïve Bayes classifier to detect intent and respond appropriately.
I created this project to experiment with traditional NLP techniques and build something simple yet functional without depending on heavy LLMs.

🚀 Quick Start
Bashgit clone https://github.com/YourUsername/ai-chatbot-nlp.git
cd ai-chatbot-nlp

python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
# source venv/bin/activate

pip install -r requirements.txt

python main.py
The chatbot will train itself the first time you run it.

📁 Project Structure
plaintextai-chatbot-nlp/
├── data/
│   └── intents.json
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── chatbot.py
├── model/
│   ├── chatbot_model.pkl
│   └── vectorizer.pkl
├── main.py
├── requirements.txt
└── README.md

✨ Features

Fast and fully offline
Easy to customize and extend
Debug mode showing intent and confidence
Clean, responsive command-line interface


🛠 Commands





























CommandDescriptionpython main.pyStart the chatbot (auto-trains if needed)python main.py --trainForce retrain the modelpython main.py --chatSkip training and jump straight to chatdebugToggle intent + confidence displayquit or exitExit the program

How It Works
The chatbot follows a simple but effective pipeline:

Preprocessing — Text is tokenized, stemmed, and cleaned using NLTK.
Vectorization — Converted into numerical features using CountVectorizer.
Classification — A Multinomial Naïve Bayes model predicts the best matching intent.
Response — A random response from the matched intent is returned.

All trained models are saved in the model/ folder for fast loading on future runs.

➕ Adding New Intents
Open data/intents.json and add new blocks like this:
JSON{
  "tag": "weather",
  "patterns": ["What's the weather like?", "Is it raining today?"],
  "responses": ["I'm a text bot, so I can't check live weather yet!", "Wish I could tell you — but I'm not connected to the internet."]
}
Then retrain:
Bashpython main.py --train

🛠 Tech Stack

Python 3.10+
NLTK
scikit-learn
Pickle


📌 Future Ideas

Add conversation memory
Improve fallback responses
Build a simple web interface


📄 License
MIT License — feel free to use and modify as you like.
