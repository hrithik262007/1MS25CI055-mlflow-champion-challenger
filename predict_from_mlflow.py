"""Load the Champion from the MLflow registry by alias and answer new questions.

No Hugging Face import here: everything goes through MLflow (the pyfunc wrapper calls HF inside).

    python predict_from_mlflow.py                    # Champion only
    python predict_from_mlflow.py --with-challenger  # Champion and Challenger side by side
"""
import argparse
import os

import mlflow
import pandas as pd
from mlflow import MlflowClient

import config as C
from squad_metrics import compute_f1, normalize_answer


def load_alias(client, alias):
    mv = client.get_model_version_by_alias(C.REGISTERED_MODEL_NAME, alias)
    model = mlflow.pyfunc.load_model(f"models:/{C.REGISTERED_MODEL_NAME}@{alias}")
    hf_id = model.model_config.get("hf_model_id")   # read from the model's own metadata
    print(f"Loaded models:/{C.REGISTERED_MODEL_NAME}@{alias} -> version {mv.version}, hf_model_id = {hf_id}")
    return model, mv.version, hf_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-challenger", action="store_true")
    ap.add_argument("--out", default=os.path.join(C.OUTPUT_DIR, "new_data_predictions.csv"))
    args = ap.parse_args()

    mlflow.set_tracking_uri(C.TRACKING_URI)
    client = MlflowClient()

    new = pd.read_csv(os.path.join(C.DATA_DIR, "new_data.csv"))
    ev = pd.read_csv(os.path.join(C.DATA_DIR, "eval_inputs.csv"))
    overlap = set(new.question.str.lower()) & set(ev.question.str.lower())
    assert not overlap, f"new_data overlaps the evaluation set: {overlap}"
    print(f"{len(new)} new questions, overlap with eval_inputs.csv: {len(overlap)}\n")

    out = new[["id", "question", "expected_answer"]].copy()
    aliases = ["champion"] + (["challenger"] if args.with_challenger else [])
    for alias in aliases:
        model, version, hf_id = load_alias(client, alias)
        pred = model.predict(new[["question", "context"]])
        out[f"{alias}_answer"] = pred["answer"].replace("", "(no answer)")
        out[f"{alias}_score"] = pred["score"]
        exp = new.expected_answer.fillna("")
        out[f"{alias}_f1"] = [round(compute_f1(e, a), 2) for e, a in zip(exp, pred["answer"])]
        out[f"{alias}_version"] = version
        out[f"{alias}_hf_model_id"] = hf_id
    out["expected_answer"] = out.expected_answer.fillna("(no answer)")

    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 46)
    cols = ["id", "question", "expected_answer"] + [c for a in aliases for c in (f"{a}_answer", f"{a}_score", f"{a}_f1")]
    print()
    print(out[cols].to_string(index=False))
    for a in aliases:
        print(f"\n{a}: mean F1 vs my expected answers = {out[f'{a}_f1'].mean():.3f} "
              f"({(out[f'{a}_f1'] == 1).sum()}/{len(out)} exact)")
    out.to_csv(args.out, index=False)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
