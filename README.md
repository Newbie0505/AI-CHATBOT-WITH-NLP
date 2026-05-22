# 🤖 AI Chatbot with NLP

A terminal-based AI chatbot built with Python, NLTK, and scikit-learn.
Uses a Multinomial Naïve Bayes classifier trained on custom intents to understand and respond to user queries.

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YourUsername/ai-chatbot-nlp.git
cd ai-chatbot-nlp

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run (auto-trains on first launch)
python main.py
```

## 📁 Project Structure

```
ai-chatbot-nlp/
├── data/
│   └── intents.json        # Training data: patterns & responses
├── src/
│   ├── preprocess.py       # Tokenization & stemming (NLTK)
│   ├── train.py            # Model training (scikit-learn)
│   └── chatbot.py          # Interactive chatbot loop
├── model/                  # Auto-generated after training
│   ├── chatbot_model.pkl
│   └── vectorizer.pkl
├── main.py                 # Entry point
├── requirements.txt
└── README.md
```

## 💬 Commands

| Command | Action |
|---|---|
| `python main.py` | Auto-train (if needed) then start chat |
| `python main.py --train` | Force retrain the model |
| `python main.py --chat` | Skip training, go straight to chat |
| Type `debug` in chat | Toggle intent + confidence display |
| Type `quit` or `exit` | Stop the chatbot |

## ➕ Adding New Topics

Edit `data/intents.json` and add a new intent block:

```json
{
  "tag": "your_topic",
  "patterns": ["How do I...", "Tell me about..."],
  "responses": ["Here's what I know...", "Great question!"]
}
```

Then retrain: `python main.py --train`

## 🛠 Tech Stack

- **Python 3.10+**
- **NLTK** — tokenization & stemming
- **scikit-learn** — CountVectorizer + MultinomialNB
- **pickle** — model persistence

## 📄 License

MIT
