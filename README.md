# AI Chatbot with NLP

A lightweight terminal-based chatbot built in Python using **NLTK** and **scikit-learn**. It uses a Multinomial Naïve Bayes classifier to understand user intents and respond intelligently.

Built as a practical project to explore traditional NLP techniques without relying on large language models.

---

## How To Start

```bash
# 1. Clone the repository
git clone https://github.com/Newbie0505/ai-chatbot-nlp.git
cd ai-chatbot-nlp

# 2. Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS / Linux
# source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the chatbot
python main.py

##Project Structure
plaintextai-chatbot-nlp/
├── data/
│   └── intents.json                 # Training patterns and responses
├── src/
│   ├── preprocess.py                # Text preprocessing (NLTK)
│   ├── train.py                     # Model training
│   └── chatbot.py                   # Chat interface and inference
├── model/                           # Auto-generated model files
│   ├── chatbot_model.pkl
│   └── vectorizer.pkl
├── main.py                          # Main entry point
├── requirements.txt
└── README.md

##Features

Fast and fully offline
Simple intent-based response system
Easy to extend with new topics
Debug mode to inspect intent and confidence
Clean command-line interface

##Commands
Command,Description
python main.py,Start chatbot (auto-trains if model not found)
python main.py --train,Force retrain the model
python main.py --chat,Launch chat mode directly (skip training)
debug (in chat),Toggle display of predicted intent + confidence score
quit or exit,Close the chatbot

How It Works

Preprocessing: User input is tokenized, stemmed, and cleaned using NLTK.
Vectorization: Text is converted to bag-of-words features using CountVectorizer.
Classification: Multinomial Naïve Bayes model predicts the best matching intent.
Response: A random response from the matched intent is displayed.

Trained models are saved in the model/ folder for quick loading on subsequent runs.

## Adding New Intents
To add new topics, edit data/intents.json:
{
  "tag": "your_topic",
  "patterns": [
    "How do I ...",
    "Tell me about ..."
  ],
  "responses": [
    "Here's what I know...",
    "Great question!"
  ]
}
After adding new intents, retrain the model:
Bashpython main.py --train

## Tech Stack

Python 3.10+
NLTK – Tokenization and stemming
scikit-learn – CountVectorizer + MultinomialNB
Pickle – Model persistence

## Future Improvements

Conversation context / memory
Better fallback handling for unknown queries
Web interface (Flask or Streamlit)
Multi-language support


 License
This project is licensed under the MIT License.
