from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import pandas as pd


CHAT_REPLACEMENTS = {
    "pls": "please",
    "plz": "please",
    "cant": "cannot",
    "wont": "will not",
    "dont": "do not",
    "didnt": "did not",
    "isnt": "is not",
    "havent": "have not",
    "hasnt": "has not",
    "nahin": "nahi",
    "nai": "nahi",
}

# Only strong Roman-Hindi markers belong here.
# Generic words such as "issue", "problem", "order", "payment", "refund"
# are deliberately NOT included because they are also common in English.
HINDI_MARKERS = {
    "mera", "meri", "mere", "mujhe", "mujhko", "hamara", "humara",
    "aap", "aapka", "aapki", "aapke", "hai", "hain", "ho", "hua",
    "huy", "huyi", "gaya", "gayi", "gaye", "raha", "rahi", "rahe",
    "nahi", "kaise", "kaha", "kahan", "kab", "kyun", "kyu", "chahiye",
    "karna", "karni", "karo", "karu", "kiya", "kiye", "mila", "mili",
    "mil", "abhi", "bahut", "paisa", "paise", "ke", "ka", "ki", "ko",
    "se", "par", "mein", "me", "ye", "yeh", "woh", "do", "diya",
    "liye", "liya", "sakta", "sakti", "sakte", "hoon", "hun", "mera",
    "mujhse", "mujhe", "kuch", "koi", "baar", "din", "tak", "wale",
    "wali", "wala", "andar", "bahar", "pehle", "baad"
}


def clean_text(text: str) -> str:
    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.replace("’", "'")
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = []
    for token in text.split():
        tokens.extend(CHAT_REPLACEMENTS.get(token, token).split())

    return " ".join(tokens)


def _tokens(text: str) -> list[str]:
    return clean_text(text).split()


def detect_language(text: str) -> str:
    """Explainable English/Hindi/Hinglish detector.

    Script detection is used first. For Roman text, only strong Hindi markers
    are counted. This prevents ordinary English words such as 'issue', 'order'
    or 'problem' from incorrectly turning English into Hinglish.
    """
    if text is None:
        return "English"

    raw = str(text)
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", raw))
    has_latin = bool(re.search(r"[A-Za-z]", raw))

    if has_devanagari and has_latin:
        return "Hinglish"
    if has_devanagari:
        return "Hindi"

    tokens = _tokens(raw)
    if not tokens:
        return "English"

    hindi_count = sum(1 for t in tokens if t in HINDI_MARKERS)
    english_like_count = sum(1 for t in tokens if re.fullmatch(r"[a-z]+", t))

    # One strong Hindi token is enough only for a clearly Hindi sentence.
    if hindi_count >= 3:
        if english_like_count - hindi_count >= 3:
            return "Hinglish"
        return "Hindi"

    if hindi_count == 2:
        # Two Hindi markers inside a mostly English sentence => Hinglish.
        if len(tokens) >= 6:
            return "Hinglish"
        return "Hindi"

    if hindi_count == 1 and len(tokens) <= 4:
        return "Hindi"

    return "English"


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["customer_message"] = out["customer_message"].fillna("").astype(str)
    out["clean_message"] = out["customer_message"].apply(clean_text)
    out["word_count"] = out["clean_message"].str.split().str.len()
    out["message_length"] = out["clean_message"].str.len()
    out["question_count"] = out["customer_message"].str.count(r"\?")
    out["exclamation_count"] = out["customer_message"].str.count(r"!")
    return out


def contains_any(text: str, phrases: Iterable[str]) -> bool:
    text = clean_text(text)
    return any(phrase in text for phrase in phrases)
