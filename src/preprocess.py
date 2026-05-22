"""
preprocess.py
NLP text preprocessing: tokenization and stemming using NLTK.
"""

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK data silently
nltk.download("punkt",           quiet=True)
nltk.download("punkt_tab",       quiet=True)
nltk.download("stopwords",       quiet=True)

stemmer = PorterStemmer()


def tokenize(sentence: str) -> list[str]:
    """Split a sentence into individual word tokens."""
    return word_tokenize(sentence)


def stem(word: str) -> str:
    """Reduce a word to its root form (e.g. 'running' → 'run')."""
    return stemmer.stem(word.lower())


def preprocess(sentence: str) -> list[str]:
    """Tokenize then stem every word in a sentence."""
    return [stem(w) for w in tokenize(sentence)]
