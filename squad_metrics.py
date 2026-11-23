"""Official SQuAD v2 answer normalisation, EM and token F1 (re-implemented, no extra deps)."""
import collections
import re
import string


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def get_tokens(s: str):
    return normalize_answer(s).split() if s else []


def compute_exact(gold: str, pred: str) -> int:
    return int(normalize_answer(gold) == normalize_answer(pred))


def compute_f1(gold: str, pred: str) -> float:
    g, p = get_tokens(gold), get_tokens(pred)
    if len(g) == 0 or len(p) == 0:          # unanswerable: F1 is 1 only if both are empty
        return float(g == p)
    common = collections.Counter(g) & collections.Counter(p)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def best_over_golds(golds, pred, fn):
    """SQuAD scores a prediction against every reference answer and keeps the best."""
    golds = [g for g in golds if normalize_answer(g)] or [""]
    return max(fn(g, pred) for g in golds)
