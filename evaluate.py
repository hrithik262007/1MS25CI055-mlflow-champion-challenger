"""Run Champion (A) and Challenger (B) on the same SQuAD v2 questions and log everything to MLflow.

Usage:
    python evaluate.py              # full run (200 questions)
    python evaluate.py --samples 10 # quick debug run
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import load_dataset
from transformers import pipeline

import config as C
from hf_qa_model import HFQuestionAnsweringModel
from squad_metrics import best_over_golds, compute_exact, compute_f1, normalize_answer


# --------------------------------------------------------------------------- data
def build_eval_set(n: int) -> pd.DataFrame:
    """Shuffle the SQuAD v2 validation split with a fixed seed and keep the first n rows."""
    ds = load_dataset(C.DATASET_ID, split=C.DATASET_SPLIT).shuffle(seed=C.SEED).select(range(n))
    rows = []
    for i, ex in enumerate(ds):
        golds = ex["answers"]["text"]
        rows.append({
            "id": i, "squad_id": ex["id"], "title": ex["title"],
            "question": ex["question"], "context": ex["context"],
            "is_impossible": int(len(golds) == 0),
            "reference_answers": json.dumps(golds, ensure_ascii=False),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- inference
def run_model(hf_model_id: str, df: pd.DataFrame):
    qa = pipeline("question-answering", model=hf_model_id, tokenizer=hf_model_id, device=-1)
    kw = dict(max_seq_len=C.MAX_SEQ_LEN, doc_stride=C.DOC_STRIDE,
              max_answer_len=C.MAX_ANSWER_LEN, handle_impossible_answer=C.HANDLE_IMPOSSIBLE)
    qa(question="warm up?", context="This call warms up the model.", **kw)  # not timed
    answers, scores, lat = [], [], []
    for q, c in zip(df["question"], df["context"]):
        t0 = time.perf_counter()
        out = qa(question=q, context=c, **kw)
        lat.append(time.perf_counter() - t0)
        answers.append(out["answer"].strip())
        scores.append(float(out["score"]))
    return answers, scores, lat


def score_predictions(df, answers, scores, lat):
    p = df[["id", "question", "is_impossible", "reference_answers"]].copy()
    p["prediction"] = answers
    p["abstained"] = [int(normalize_answer(a) == "") for a in answers]
    p["score"] = np.round(scores, 4)
    golds = [json.loads(g) for g in df["reference_answers"]]
    p["exact"] = [best_over_golds(g, a, compute_exact) for g, a in zip(golds, answers)]
    p["f1"] = np.round([best_over_golds(g, a, compute_f1) for g, a in zip(golds, answers)], 4)
    p["latency_ms"] = np.round(np.array(lat) * 1000, 2)

    has, no = p[p.is_impossible == 0], p[p.is_impossible == 1]
    m = {
        "exact_match": 100 * p.exact.mean(),
        "f1": 100 * p.f1.mean(),
        "has_ans_exact": 100 * has.exact.mean(),
        "has_ans_f1": 100 * has.f1.mean(),
        "no_ans_accuracy": 100 * no.abstained.mean(),     # said "no answer" when there was none
        "abstain_rate": 100 * p.abstained.mean(),
        "n_zero_f1": int((p.f1 == 0).sum()),
        "avg_latency_ms": float(np.mean(lat) * 1000),
        "total_inference_s": float(np.sum(lat)),
    }
    return p, {k: round(float(v), 4) for k, v in m.items()}


# --------------------------------------------------------------------------- plots
def plot_metric_bars(mA, mB, path):
    keys = ["exact_match", "f1", "has_ans_exact", "has_ans_f1", "no_ans_accuracy"]
    x = np.arange(len(keys)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    a = ax.bar(x - w / 2, [mA[k] for k in keys], w, label=f"A (Champion) {C.MODELS['A']['short_name']}")
    b = ax.bar(x + w / 2, [mB[k] for k in keys], w, label=f"B (Challenger) {C.MODELS['B']['short_name']}")
    ax.bar_label(a, fmt="%.1f", fontsize=8); ax.bar_label(b, fmt="%.1f", fontsize=8)
    ax.set_xticks(x, keys); ax.set_ylim(0, 105); ax.set_ylabel("score (0-100, higher is better)")
    ax.set_title(f"Champion vs Challenger on {len_eval} SQuAD v2 questions"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_latency(mA, mB, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    names = [C.MODELS["A"]["short_name"], C.MODELS["B"]["short_name"]]
    bars = ax.bar(names, [mA["avg_latency_ms"], mB["avg_latency_ms"]], color=["#4C72B0", "#DD8452"])
    ax.bar_label(bars, fmt="%.1f ms")
    ax.set_ylabel("avg latency per question, ms (lower is better)"); ax.set_title("Inference latency on CPU")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_f1_hist(pA, pB, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 11)
    ax.hist([pA.f1, pB.f1], bins=bins, label=[C.MODELS["A"]["short_name"], C.MODELS["B"]["short_name"]])
    ax.set_xlabel("per-question F1"); ax.set_ylabel("number of questions")
    ax.set_title("Per-question F1 distribution"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_answer_matrix(p, name, path):
    """2x2 'confusion matrix' for answerable vs unanswerable: did the model answer or abstain?"""
    mat = np.array([[((p.is_impossible == 0) & (p.abstained == 0)).sum(), ((p.is_impossible == 0) & (p.abstained == 1)).sum()],
                    [((p.is_impossible == 1) & (p.abstained == 0)).sum(), ((p.is_impossible == 1) & (p.abstained == 1)).sum()]])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, mat[i, j], ha="center", va="center", fontsize=14,
                    color="white" if mat[i, j] > mat.max() / 2 else "black")
    ax.set_xticks([0, 1], ["answered", "abstained"]); ax.set_yticks([0, 1], ["has answer", "no answer"])
    ax.set_xlabel("model output"); ax.set_ylabel("reference"); ax.set_title(f"Answer / abstain matrix\n{name}")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- decision
def decide(mA, mB):
    delta = mB[C.PRIMARY_METRIC] - mA[C.PRIMARY_METRIC]
    ratio = mB["avg_latency_ms"] / mA["avg_latency_ms"]
    if ratio > C.MAX_LATENCY_RATIO:
        return "A", delta, ratio, f"Challenger rejected by guardrail: latency {ratio:.2f}x > {C.MAX_LATENCY_RATIO}x"
    if delta >= C.MARGIN:
        return "B", delta, ratio, f"Challenger beats Champion by {delta:.2f} F1 (>= margin {C.MARGIN}) within latency guardrail"
    if delta <= -C.MARGIN:
        return "A", delta, ratio, f"Champion better by {-delta:.2f} F1"
    winner = "A" if mA["avg_latency_ms"] <= mB["avg_latency_ms"] else "B"
    return winner, delta, ratio, f"Difference {delta:.2f} F1 within margin {C.MARGIN}; tie-breaker picks faster model {winner}"


def lib_tags():
    return {"transformers_version": transformers.__version__, "torch_version": torch.__version__,
            "mlflow_version": mlflow.__version__,
            "library_versions": f"transformers={transformers.__version__}, torch={torch.__version__}, mlflow={mlflow.__version__}"}


# --------------------------------------------------------------------------- main
def main():
    global len_eval
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=C.EVAL_SAMPLES)
    args = ap.parse_args()
    len_eval = args.samples
    os.makedirs(C.DATA_DIR, exist_ok=True); os.makedirs(C.OUTPUT_DIR, exist_ok=True)

    df = build_eval_set(args.samples)
    eval_path = os.path.join(C.DATA_DIR, "eval_inputs.csv"); df.to_csv(eval_path, index=False)
    print(f"Saved {len(df)} questions ({df.is_impossible.sum()} unanswerable) -> {eval_path}")

    mlflow.set_tracking_uri(C.TRACKING_URI)
    mlflow.set_experiment(C.EXPERIMENT_NAME)

    preds, metrics, run_ids = {}, {}, {}
    with mlflow.start_run(run_name=f"compare_{C.TASK}") as parent:
        for key in ("A", "B"):
            m_cfg = C.MODELS[key]
            run_name = f"{key}_{C.TASK}_{m_cfg['short_name']}"
            print(f"\n=== {run_name}: {m_cfg['hf_model_id']} ===")
            with mlflow.start_run(run_name=run_name, nested=True) as run:
                answers, scores, lat = run_model(m_cfg["hf_model_id"], df)
                p, m = score_predictions(df, answers, scores, lat)
                preds[key], metrics[key], run_ids[key] = p, m, run.info.run_id
                print(json.dumps(m, indent=2))

                mlflow.log_params({"hf_model_id": m_cfg["hf_model_id"], "task": C.TASK, "role": m_cfg["role"],
                                   "max_seq_len": C.MAX_SEQ_LEN, "doc_stride": C.DOC_STRIDE,
                                   "max_answer_len": C.MAX_ANSWER_LEN,
                                   "handle_impossible_answer": C.HANDLE_IMPOSSIBLE, "batch_size": 1,
                                   "eval_samples": len(df), "seed": C.SEED, "device": "cpu"})
                mlflow.log_metrics(m)
                mlflow.set_tags({"student_usn": C.STUDENT_USN, "student_name": C.STUDENT_NAME,
                                 "dataset_name": C.DATASET_NAME, "model_key": key, **lib_tags()})

                short = m_cfg["short_name"]
                pred_path = os.path.join(C.OUTPUT_DIR, f"predictions_{short}.csv"); p.to_csv(pred_path, index=False)
                met_path = os.path.join(C.OUTPUT_DIR, f"metrics_{short}.json")
                json.dump({"hf_model_id": m_cfg["hf_model_id"], **m}, open(met_path, "w"), indent=2)
                cm_path = os.path.join(C.OUTPUT_DIR, f"answer_matrix_{short}.png")
                plot_answer_matrix(p, short, cm_path)
                for f in (pred_path, met_path, cm_path, eval_path):
                    mlflow.log_artifact(f)
                mlflow.log_table(p, artifact_file=f"predictions_table_{short}.json")

                info = mlflow.pyfunc.log_model(
                    name="model", python_model=HFQuestionAnsweringModel(),
                    code_paths=["hf_qa_model.py"],
                    model_config={"hf_model_id": m_cfg["hf_model_id"], "max_seq_len": C.MAX_SEQ_LEN,
                                  "doc_stride": C.DOC_STRIDE, "max_answer_len": C.MAX_ANSWER_LEN,
                                  "handle_impossible_answer": C.HANDLE_IMPOSSIBLE},
                    input_example=df[["question", "context"]].head(2),
                    pip_requirements=["mlflow>=3.0", "transformers>=4.40,<5", "torch>=2.2", "pandas"],
                )
                mlflow.set_tag("logged_model_uri", info.model_uri)

        # ----------------------------------------------------------------- summary run
        mA, mB = metrics["A"], metrics["B"]
        winner, delta, ratio, reason = decide(mA, mB)
        print(f"\nDecision: winner={winner} ({C.MODELS[winner]['hf_model_id']}) - {reason}")

        both = preds["A"][["id", "question", "is_impossible", "reference_answers"]].copy()
        for k in ("A", "B"):
            both[f"pred_{k}"] = preds[k]["prediction"]; both[f"f1_{k}"] = preds[k]["f1"]; both[f"score_{k}"] = preds[k]["score"]
        dis = both[[normalize_answer(a) != normalize_answer(b) for a, b in zip(both.pred_A, both.pred_B)]].copy()
        dis["f1_gap_B_minus_A"] = (dis.f1_B - dis.f1_A).round(4)
        dis = dis.reindex(dis.f1_gap_B_minus_A.abs().sort_values(ascending=False).index)
        dis_path = os.path.join(C.OUTPUT_DIR, "disagreements_all.csv"); dis.to_csv(dis_path, index=False)
        b_better, a_better = int((dis.f1_gap_B_minus_A > 0).sum()), int((dis.f1_gap_B_minus_A < 0).sum())

        summary = {"student_usn": C.STUDENT_USN, "eval_samples": len(df),
                   "unanswerable_in_eval": int(df.is_impossible.sum()),
                   "rule": {"primary_metric": C.PRIMARY_METRIC, "margin_f1_points": C.MARGIN,
                            "max_latency_ratio": C.MAX_LATENCY_RATIO, "tie_breaker": C.TIE_BREAKER},
                   "A": {"hf_model_id": C.MODELS["A"]["hf_model_id"], **mA},
                   "B": {"hf_model_id": C.MODELS["B"]["hf_model_id"], **mB},
                   "delta_primary_metric": round(delta, 4), "latency_ratio_B_over_A": round(ratio, 3),
                   "disagreements": len(dis), "disagreements_B_better": b_better, "disagreements_A_better": a_better,
                   "winner": winner, "winner_hf_model_id": C.MODELS[winner]["hf_model_id"], "decision_reason": reason,
                   "run_ids": run_ids}
        sum_path = os.path.join(C.OUTPUT_DIR, "metrics_summary.json"); json.dump(summary, open(sum_path, "w"), indent=2)

        plot_metric_bars(mA, mB, os.path.join(C.OUTPUT_DIR, "comparison_metrics_bar.png"))
        plot_latency(mA, mB, os.path.join(C.OUTPUT_DIR, "comparison_latency_bar.png"))
        plot_f1_hist(preds["A"], preds["B"], os.path.join(C.OUTPUT_DIR, "f1_distribution.png"))

        mlflow.log_params({"primary_metric": C.PRIMARY_METRIC, "margin": C.MARGIN,
                           "max_latency_ratio": C.MAX_LATENCY_RATIO, "tie_breaker": C.TIE_BREAKER,
                           "eval_samples": len(df), "seed": C.SEED})
        mlflow.log_metrics({"delta_primary_metric": round(delta, 4), "latency_ratio_B_over_A": round(ratio, 3),
                            "A_f1": mA["f1"], "B_f1": mB["f1"], "A_exact_match": mA["exact_match"],
                            "B_exact_match": mB["exact_match"], "A_avg_latency_ms": mA["avg_latency_ms"],
                            "B_avg_latency_ms": mB["avg_latency_ms"], "n_disagreements": len(dis)})
        mlflow.set_tags({"winner": winner, "winner_hf_model_id": C.MODELS[winner]["hf_model_id"],
                         "decision_reason": reason, "student_usn": C.STUDENT_USN, "dataset_name": C.DATASET_NAME,
                         "A_run_id": run_ids["A"], "B_run_id": run_ids["B"], **lib_tags()})
        for f in ("comparison_metrics_bar.png", "comparison_latency_bar.png", "f1_distribution.png",
                  "metrics_summary.json", "disagreements_all.csv"):
            mlflow.log_artifact(os.path.join(C.OUTPUT_DIR, f))
        top5 = os.path.join(C.OUTPUT_DIR, "disagreements_top5.csv")
        if os.path.exists(top5):
            mlflow.log_artifact(top5)
        mlflow.log_artifact(eval_path)
        if os.path.exists("README.md"):
            mlflow.log_artifact("README.md")
        print(f"Summary run: {parent.info.run_id}")


if __name__ == "__main__":
    main()
