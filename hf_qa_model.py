"""Custom mlflow.pyfunc wrapper around a Hugging Face extractive QA model (Lab 9 pattern).

Input : DataFrame with columns `question` and `context`.
Output: DataFrame with `question`, `answer` ("" means the model thinks it is unanswerable)
        and `score` (the pipeline's confidence for that answer).
"""
import mlflow
import pandas as pd


class HFQuestionAnsweringModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from transformers import pipeline  # imported here so callers never touch HF directly

        cfg = context.model_config
        self.hf_model_id = cfg["hf_model_id"]
        self.max_seq_len = int(cfg.get("max_seq_len", 384))
        self.doc_stride = int(cfg.get("doc_stride", 128))
        self.max_answer_len = int(cfg.get("max_answer_len", 30))
        self.handle_impossible = bool(cfg.get("handle_impossible_answer", True))
        self.qa = pipeline("question-answering", model=self.hf_model_id,
                           tokenizer=self.hf_model_id, device=-1)

    def predict(self, context, model_input, params=None):
        df = pd.DataFrame(model_input)
        rows = []
        for q, c in zip(df["question"], df["context"]):
            out = self.qa(question=str(q), context=str(c),
                          max_seq_len=self.max_seq_len, doc_stride=self.doc_stride,
                          max_answer_len=self.max_answer_len,
                          handle_impossible_answer=self.handle_impossible)
            rows.append({"question": q, "answer": out["answer"].strip(),
                         "score": round(float(out["score"]), 4)})
        return pd.DataFrame(rows)
