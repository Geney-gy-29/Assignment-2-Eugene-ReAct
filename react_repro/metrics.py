"""Evaluation metrics matching the paper's normalization conventions
(SQuAD-style: lowercase, strip articles/punctuation/extra whitespace)."""

import re
import string


def _normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em(prediction: str, gold: str) -> int:
    """Exact match after normalization, as used for HotpotQA in the paper."""
    return int(_normalize_answer(prediction) == _normalize_answer(gold))


def fever_acc(prediction: str, gold: str) -> int:
    """FEVER label accuracy: exact match on {SUPPORTS, REFUTES, NOT ENOUGH INFO}."""
    return int(prediction.strip().upper() == gold.strip().upper())
