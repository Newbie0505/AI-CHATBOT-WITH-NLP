"""
main.py  ─  Entry point for the AI Chatbot with NLP
─────────────────────────────────────────────────────
Usage:
    python main.py          → train if needed, then run chatbot
    python main.py --train  → force re-train the model
    python main.py --chat   → skip training, go straight to chat
"""

import argparse
import os
import sys

# Make sure src/ is importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train   import train
from src.chatbot import run_chatbot

MODEL_PATH = os.path.join("model", "chatbot_model.pkl")


def main():
    parser = argparse.ArgumentParser(description="AI Chatbot with NLP")
    parser.add_argument("--train", action="store_true", help="Force re-train the model")
    parser.add_argument("--chat",  action="store_true", help="Skip training, start chatting")
    args = parser.parse_args()

    # ── Decide whether to train ──────────────────────────────────────
    if args.train:
        print("\n[INFO] Force re-training model...")
        train()
    elif args.chat:
        print("[INFO] Skipping training — loading existing model.")
    else:
        # Auto-train if model doesn't exist yet
        if not os.path.exists(MODEL_PATH):
            print("[INFO] No model found. Training now...\n")
            train()
        else:
            print("[INFO] Model found. Skipping training.")
            print("       (Run  python main.py --train  to retrain)\n")

    # ── Start the chatbot ─────────────────────────────────────────────
    run_chatbot()


if __name__ == "__main__":
    main()
