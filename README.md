# 1MS25CI055 – MLflow Champion–Challenger (Hugging Face Question Answering)

**Student:** I.V. HRITHIK KUMAR · **USN:** 1MS25CI055
Skill Lab (MLOps with MLflow) – Mini Project, extension of Labs 4, 5, 8, 9, 10, 11 and 13.

## 1. Task and models

| | |
|---|---|
| Task | Extractive question answering (find the answer span in a paragraph, or say there is no answer) |
| Dataset | **SQuAD v2.0** (`rajpurkar/squad_v2`), validation split, held out from training |
| Subset | 200 questions, shuffled with seed 55, saved in `data/eval_inputs.csv` (102 answerable, 98 unanswerable) |
| Model A (Champion) | `deepset/minilm-uncased-squad2` (~33M parameters) |
| Model B (Challenger) | `deepset/roberta-base-squad2` (~125M parameters) |

This is the brief's pairing "7 vs 6". Both models were fine-tuned on SQuAD v2, so both can return an empty answer when a question cannot be answered from the paragraph (`handle_impossible_answer=True`).

## 2. How to run

```bash
# 0) clone and install (Python 3.10+; CPU is enough)
git clone https://github.com/hrithik262007/1MS25CI055-mlflow-champion-challenger.git
cd 1MS25CI055-mlflow-champion-challenger
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt                      # note: transformers < 5 (v5 removed the QA pipeline)

# 1) terminal 1 - start the MLflow tracking server (Lab 7)
python -m mlflow server --host 127.0.0.1 --port 5000

# 2) terminal 2 - run both models on the same 200 questions and log everything
python evaluate.py --samples 10    # quick debug run first
python evaluate.py                 # full run

# 3) register both models as "qa-model" and set @champion / @challenger from the rule's decision
python register_champion.py

# 4) load the Champion by alias and answer new questions
python predict_from_mlflow.py                    # Champion only
python predict_from_mlflow.py --with-challenger  # Champion and Challenger side by side

# 5) stretch goal - flip the aliases, rerun step 4 without editing it, then flip back
python register_champion.py --swap
python predict_from_mlflow.py --out outputs/new_data_predictions_after_alias_swap.csv
python register_champion.py
```

The tracking URI defaults to `http://127.0.0.1:5000` (`config.py`). To run without a server, set
`MLFLOW_TRACKING_URI=sqlite:///mlflow.db`. The first run downloads both checkpoints (~600 MB in total) into the Hugging Face cache.

### Repository layout

```
├── README.md
├── requirements.txt
├── config.py                 # all settings + the Champion rule
├── squad_metrics.py          # SQuAD v2 answer normalisation, exact match and token F1
├── hf_qa_model.py            # custom mlflow.pyfunc.PythonModel wrapper (Lab 9)
├── evaluate.py               # runs A and B on the same questions, logs to MLflow, applies the rule
├── register_champion.py      # registers both models, sets @champion / @challenger (--swap for the stretch goal)
├── predict_from_mlflow.py    # loads models:/qa-model@champion, answers new questions (no HF import)
├── data/
│   ├── eval_inputs.csv       # 200 shared evaluation questions + contexts + reference answers
│   └── new_data.csv          # 8 new questions written by me (MSRIT, Chandrayaan-3, Hampi, MLflow), 2 unanswerable
├── outputs/                  # predictions, metrics, plots, disagreements, new-data predictions
└── screenshots/              # MLflow compare view, registry aliases, predict_from_mlflow terminal output
```

### What gets logged in MLflow

Experiment **`hf-qa-champion-challenger`**, one summary run with two nested model runs:

| Run | Contents |
|---|---|
| `compare_qa` (summary) | params `primary_metric`, `margin`, `max_latency_ratio`, `tie_breaker`, `eval_samples`, `seed`; metrics `delta_primary_metric`, `latency_ratio_B_over_A`, both models' F1/EM/latency, `n_disagreements`; tags `winner`, `winner_hf_model_id`, `decision_reason`, `student_usn`, `A_run_id`, `B_run_id`; artifacts: metric bar chart, latency chart, F1 histogram, `metrics_summary.json`, `disagreements_all.csv`, `disagreements_top5.csv`, `eval_inputs.csv`, `README.md` |
| `A_qa_minilm-squad2` | params `hf_model_id`, `task`, `role=champion`, `max_seq_len`, `doc_stride`, `max_answer_len`, `handle_impossible_answer`, `batch_size`, `eval_samples`, `seed`, `device`; metrics `exact_match`, `f1`, `has_ans_exact`, `has_ans_f1`, `no_ans_accuracy`, `abstain_rate`, `n_zero_f1`, `avg_latency_ms`, `total_inference_s`; tags `student_usn`, `dataset_name`, `library_versions`; artifacts: predictions CSV + table, metrics JSON, answer/abstain matrix; the pyfunc **model** |
| `B_qa_roberta-base-squad2` | same as A, with `role=challenger` |

Both models are logged as a custom `mlflow.pyfunc.PythonModel` (`hf_qa_model.py`). The Hugging Face ID and inference settings are stored in the model's `model_config`, so the loaded model knows which checkpoint it is. Input is a DataFrame with `question` and `context`; output is a DataFrame with `question`, `answer` (empty = no answer) and `score`.

## 3. Champion rule (written before the final run)

- **Primary metric:** SQuAD v2 **F1** (token overlap between prediction and the best-matching reference, 0–100, averaged over all 200 questions; for an unanswerable question F1 is 100 only if the model returns no answer). **Higher is better.**
- **Minimum improvement margin:** the Challenger must beat the Champion by **at least 2.0 F1 points**, otherwise the Champion stays. With 200 questions one question is worth 0.5 points, and partial-credit F1 is noisier than accuracy, so a gap under 2 points (four questions) is treated as noise.
- **Tie-breaker:** if the difference is within the margin, prefer the **faster** model (lower `avg_latency_ms`).
- **Guardrail:** reject the Challenger outright if its average latency is **more than 5× the Champion's** on the same CPU.
- **Direction of every metric:** `exact_match`, `f1`, `has_ans_exact`, `has_ans_f1`, `no_ans_accuracy`: higher is better. `n_zero_f1`, `avg_latency_ms`, `total_inference_s`: lower is better. `abstain_rate` has no good direction on its own (the right value is the share of unanswerable questions, 49% here).

The same rule is in `config.py` and is logged in the summary run as `primary_metric`, `margin`, `max_latency_ratio` and `tie_breaker`.

## 4. Results table

200 questions from the SQuAD v2 validation split, seed 55, CPU (2 cores), one question at a time, max_seq_len 384, doc_stride 128.

| Metric (direction) | A: MiniLM (Champion) | B: RoBERTa-base (Challenger) | B − A |
|---|---|---|---|
| **F1** (↑, primary) | **80.38** | **78.85** | **−1.52** |
| Exact match (↑) | 78.00 | 76.00 | −2.00 |
| F1 on answerable questions (↑) | 88.00 | 84.03 | −3.97 |
| Exact match on answerable (↑) | 83.33 | 78.43 | −4.90 |
| Correct "no answer" on unanswerable (↑) | 72.45 | 73.47 | +1.02 |
| Abstain rate (target 49%) | 40.5% | 42.0% | +1.5 |
| Questions with F1 = 0 (↓) | 37 | 39 | +2 |
| Avg latency per question, ms (↓) | 91.9 | 299.2 | 3.26× slower |
| Total inference time, s (↓) | 18.4 | 59.9 | +41.5 s |
| Parameters | ~33M | ~125M | ~3.8× |

The two models give different answers on 33 of 200 questions: B scores higher on 10, A on 16, and 7 are partial-credit ties (`outputs/disagreements_all.csv`). Five of them, with my comments, are in `outputs/disagreements_top5.csv`.

Plots (in `outputs/` and logged to MLflow): `comparison_metrics_bar.png`, `comparison_latency_bar.png`, `f1_distribution.png`, `answer_matrix_minilm-squad2.png`, `answer_matrix_roberta-base-squad2.png`.

### Five qualitative disagreements

| id | Question (shortened) | Reference | A (MiniLM) | B (RoBERTa) | Comment |
|---|---|---|---|---|---|
| 170 | Company formed by the merger of Sky Television and BSB? | BSkyB | BSkyB | (no answer) | Unusual wording ("equal merger") made RoBERTa abstain; MiniLM found the span right after it. |
| 142 | Portion of Jacksonville households with only one person? | 29.7% | 29.7% | 7.9% | Two percentages side by side; RoBERTa took the "living alone, 65+" figure. |
| 110 | What are colonial powers blamed for? | ended the true Islamic system | (no answer) | ended the true Islamic system | Long, opinion-heavy paragraph: MiniLM gave up, RoBERTa found an accepted span. |
| 68 | Which port is the United States's third-largest seaport? | (no answer) | (no answer) | Port of Jacksonville | Trap question: the text says *Florida's* third-largest. RoBERTa fell for it, MiniLM abstained. |
| 71 | In what state is Yale located? | (no answer) | Connecticut | (no answer) | The paragraph mentions Connecticut only for the regatta; MiniLM answered from a nearby word. |

### Prediction on new data (Step 5.6)

`predict_from_mlflow.py` loads `models:/qa-model@champion` → **version 1, `deepset/minilm-uncased-squad2`** (version from the registry, model ID from the model's own `model_config`, never hard-coded). The 8 new questions in `data/new_data.csv` were written by me and are checked in code to not overlap the evaluation set. Against my expected answers the Champion scored mean F1 0.86 (6/8 exact): it answered "lunar south pole" instead of "near the lunar south pole" and abstained on the MLflow-alias question. With `--with-challenger`, RoBERTa scored 0.96 (7/8). Both correctly said "no answer" to the two unanswerable questions (mission cost, last Vijayanagara ruler). Output: `outputs/new_data_predictions.csv`.

**Stretch goal:** `python register_champion.py --swap` moved `@champion` to version 2; rerunning the *unchanged* `predict_from_mlflow.py` loaded version 2 / RoBERTa-base instead (`outputs/new_data_predictions_after_alias_swap.csv`). The aliases were then set back to the rule's decision.

### Screenshots

| File | Shows |
|---|---|
| `screenshots/01_mlflow_compare_view.png` | MLflow Compare view of `A_qa_minilm-squad2` vs `B_qa_roberta-base-squad2` |
| `screenshots/02_registry_aliases.png` | `qa-model` in the registry: v1 `@champion`, v2 `@challenger` |
| `screenshots/03_predict_from_mlflow_terminal.png` | Terminal output of `python predict_from_mlflow.py --with-challenger` |
| `screenshots/00_mlflow_experiment_runs.png` | The experiment's run list (summary run + two nested runs) |

## 5. Decision

**Champion: Model A, `deepset/minilm-uncased-squad2` (registry version 1, alias `@champion`). The Challenger does not replace it; RoBERTa-base keeps `@challenger`.**

Following the rule in section 3, the primary metric is SQuAD v2 F1. RoBERTa-base scored 78.85 against MiniLM's 80.38, a difference of −1.52 points. That is inside the ±2.0 margin, so neither model clearly wins on quality and the tie-breaker applies: MiniLM is faster (92 ms vs 299 ms per question), so it stays Champion. The latency guardrail was not triggered (3.26× is under the 5× limit).

Is the difference meaningful? No. With 200 questions one question is 0.5 F1 points, so the gap is about three questions. The models disagree on 33 questions, B wins 10 and A wins 16; a sign test on 10 vs 16 gives p ≈ 0.33, which is far from significant. The breakdown is mixed: MiniLM was better on answerable questions (88.0 vs 84.0 F1), while RoBERTa was slightly better at recognising unanswerable ones (73.5% vs 72.4%). Both models answer too often: they abstain on about 41–42% of questions when 49% are unanswerable.

The trade-off favours the Champion. RoBERTa-base is ~3.8× larger and ~3.3× slower on CPU and did not deliver better answers on this sample. I expected the bigger model to win, so this is an honest negative result. A larger evaluation set (1,000+ questions) would be needed before changing the decision, and RoBERTa's better results on my own new questions (7/8 vs 6/8) show the picture could change on other text.

## 6. Limitations and declaration

- **Sample size:** 200 questions from one dataset (Wikipedia paragraphs). A 1.5-point gap is about three questions and is not statistically significant (sign test p ≈ 0.33).
- **In-domain test:** both models were fine-tuned on SQuAD v2 train, so this is an in-domain test for both; results may not transfer to textbooks, news or Indian-English text.
- **Latency** was measured on a 2-core CPU, one question at a time, after one warm-up call; numbers will differ on other hardware and on GPU.
- **New data** is only 8 short questions written by me, so it shows the pipeline works rather than measuring accuracy.
- The logged pyfunc model stores the Hugging Face model ID and downloads the weights from the Hub in `load_context` (same approach as Lab 9), so the weights are not stored in MLflow and loading needs internet the first time.
- `transformers` is pinned below 5.0 because version 5 removed the `question-answering` pipeline.
- `mlruns/`, `mlartifacts/`, `mlflow.db` and model files are excluded by `.gitignore` as required.
- **AI assistance declaration:** I used an AI assistant (Hark) to help write and debug the code and to draft parts of this README. I have read and run every script, and I can explain each line and design choice in the viva.
