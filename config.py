"""Shared settings for the Champion-Challenger mini project (USN 1MS25CI055)."""
import os

STUDENT_USN = "1MS25CI055"
STUDENT_NAME = "I.V. HRITHIK KUMAR"

# Tracking server (start it with: python -m mlflow server --host 127.0.0.1 --port 5000)
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "hf-qa-champion-challenger"
REGISTERED_MODEL_NAME = "qa-model"

TASK = "qa"
DATASET_ID = "rajpurkar/squad_v2"
DATASET_SPLIT = "validation"
DATASET_NAME = "rajpurkar/squad_v2 (validation split)"
EVAL_SAMPLES = 200
SEED = 55                 # last digits of my USN, fixed for every run
MAX_SEQ_LEN = 384         # tokens per question+context window
DOC_STRIDE = 128          # overlap between windows for long contexts
MAX_ANSWER_LEN = 30       # longest answer span the pipeline may return
HANDLE_IMPOSSIBLE = True  # SQuAD v2 has unanswerable questions -> model may return ""

# Model A = current Champion, Model B = Challenger (pairing "7 vs 6" from the brief)
MODELS = {
    "A": {
        "hf_model_id": "deepset/minilm-uncased-squad2",
        "short_name": "minilm-squad2",
        "role": "champion",
    },
    "B": {
        "hf_model_id": "deepset/roberta-base-squad2",
        "short_name": "roberta-base-squad2",
        "role": "challenger",
    },
}

# ---------------------------------------------------------------------------
# Champion rule (written BEFORE the final run - see README section 3)
# ---------------------------------------------------------------------------
PRIMARY_METRIC = "f1"          # SQuAD v2 token-level F1 (0-100), higher is better
MARGIN = 2.0                   # Challenger must beat Champion by >= 2.0 F1 points
MAX_LATENCY_RATIO = 5.0        # guardrail: reject Challenger if avg latency > 5x Champion
TIE_BREAKER = "lower avg_latency_ms"

# Direction of every metric (True = higher is better)
METRIC_DIRECTIONS = {
    "exact_match": True, "f1": True,
    "has_ans_exact": True, "has_ans_f1": True,
    "no_ans_accuracy": True, "abstain_rate": None,
    "n_zero_f1": False, "avg_latency_ms": False, "total_inference_s": False,
}

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
