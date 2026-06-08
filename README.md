# Years-of-Experience (YoE) Classifier

A hybrid classifier that assigns each job posting one of four years-of-experience bands based on the job title and description:

| Label | Band              |
|-------|-------------------|
| 1     | 0–2 years (entry) |
| 2     | 3–5 years (early) |
| 3     | 6–15 years (mid)  |
| 4     | 15+ years (senior/executive) |

The system combines two layers:

1. **Deterministic regex rules** — high-precision pattern matching for explicit signals like "5+ years required", "Senior Director", "VP of Engineering", etc.
2. **Calibrated linear ML model** — TF-IDF over title and description (word + char n-grams) plus engineered numeric features, fed into a calibrated classifier (Logistic Regression or Linear SVC — chosen automatically during training).

At prediction time, rule hits with confidence ≥ 0.90 override the model. Lower-confidence rule hits are blended with the model. Anything below the chosen confidence threshold is flagged as `unsure` for human review. An executive-floor guardrail forces label 4 when the title carries an executive signal but the model produced label 1.

---

## Repository layout

```
end/
├── yoe_classifier/              # the package
│   ├── cli/
│   │   ├── train.py             # train the model
│   │   ├── evaluate.py          # cross-validation report
│   │   └── score_batch.py       # score a CSV of jobs
│   ├── data/                    # symlinks to gold / silver / jobs CSV
│   ├── artifacts/               # trained model + meta + eval report
│   ├── rules.py                 # regex rule engine + feature extractor
│   ├── features.py              # sklearn FeatureUnion (TF-IDF + numeric)
│   ├── model.py                 # candidate estimators + save/load
│   ├── predict.py               # hybrid combiner
│   ├── data_io.py               # CSV/JSON loaders
│   └── requirements.txt
├── Candogram-5000-random-jobs.csv
├── gold100.json                 # 100 hand-labeled jobs
├── sliver400.json               # 400 silver-labeled jobs
├── flagged.csv / flagged.json   # rows flagged as unsure
└── predictions.csv              # batch-scored output
```

---

## Installation

Requires Python 3.10+.

```bash
cd /Users/talha/end
python -m venv .venv
source .venv/bin/activate
pip install -r yoe_classifier/requirements.txt
```

Dependencies: `scikit-learn`, `pandas`, `numpy`, `scipy`, `joblib`, `regex`.

---

## Data layout

The training/eval CLIs expect three files inside `yoe_classifier/data/`:

| File             | What it is                                | Format |
|------------------|-------------------------------------------|--------|
| `jobs.csv`       | Source job postings (the universe)        | CSV with `UniqueJobAdID`, `OrigJobTitle`, `JobDescription` |
| `gold100.json`   | High-quality hand labels (100 rows)       | JSON list of `{job_id, label, …}` |
| `silver400.json` | Larger, lower-confidence labels (400 rows)| JSON list of `{job_id, label, unsure, …}` |

The labels join back to the job text by `job_id` → `UniqueJobAdID`.

In this repo the `data/` folder already contains symlinks pointing to the top-level files. If you bring your own data, either replace the symlinks or pass `--data-dir` to each CLI.

### Label JSON schema

Each entry in `gold100.json` / `silver400.json` looks like:

```json
{
  "job_id": "12345",
  "label": 3,
  "unsure": false,
  "decided_by": "explicit_yoe",
  "evidence_snippet": "minimum 7 years of experience",
  "evidence_source": "description"
}
```

Only `job_id` and `label` are required. Gold rows get a sample weight of 3.0, silver rows 1.0, and silver+unsure rows 0.3 — so adding silver data is cheap but not destructive.

---

## Training

```bash
cd /Users/talha/end
python -m yoe_classifier.cli.train
```

What happens, in order:

1. Loads `gold100.json` + `silver400.json` and joins to the job text from `jobs.csv`.
2. Runs **5-fold stratified cross-validation on gold** (silver is always in the train fold) for each candidate estimator in `model.CANDIDATES`:
   - `logreg_C0.5`, `logreg_C1.0`, `logreg_C2.0`
   - `linsvc_C0.5`, `linsvc_C1.0`
3. Picks the **winner** by hybrid macro-F1 (rules + model combined, not model alone).
4. Sweeps the `unsure` confidence threshold over `[0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]` and picks the smallest threshold whose kept-rows accuracy ≥ 95% and flagged share ≤ 30% (falls back to ≥ 90% / ≤ 40%, then to `kept_acc · √coverage` if nothing qualifies).
5. **Refits the winner on gold + silver** (all rows) with sample weights.
6. Writes artifacts to `yoe_classifier/artifacts/`:
   - `model.joblib` — the pickled sklearn `Pipeline`
   - `meta.json` — estimator name, threshold, classes, CV summary

### Training flags

```
--data-dir DIR    default: yoe_classifier/data
--out-dir DIR     default: yoe_classifier/artifacts
--gold FILE       default: gold100.json
--silver FILE     default: silver400.json
--jobs FILE       default: jobs.csv
```

Example with custom paths:

```bash
python -m yoe_classifier.cli.train \
    --data-dir /path/to/my/data \
    --out-dir /tmp/yoe_run_01 \
    --gold my_gold.json \
    --silver my_silver.json \
    --jobs my_jobs.csv
```

A typical training run takes a few minutes on a laptop. The CV grid is sequential, not parallelized.

---

## Evaluation

After training, get a full report on the picked estimator:

```bash
python -m yoe_classifier.cli.evaluate
```

This re-runs the same 5-fold CV with the winner from `meta.json` and prints:

- Hybrid accuracy, macro-F1, ordinal MAE, coverage, flagged-share, kept-only accuracy
- Per-class precision/recall/F1 (sklearn `classification_report`)
- Confusion matrix for all rows, and a second one for kept (non-flagged) rows
- Rules-only precision broken down by confidence bucket and by which rule fired (`decided_by`)
- A coverage-vs-accuracy sweep over the threshold grid

The full numerical report is written to `yoe_classifier/artifacts/evaluation.json`.

Same `--data-dir` / `--out-dir` / `--gold` / `--silver` / `--jobs` flags as `train`.

---

## Batch scoring

To label a whole CSV of jobs, point `--input` at your jobs CSV and `--output` at wherever you want the predictions written:

```bash
python -m yoe_classifier.cli.score_batch \
    --input  Candogram-5000-random-jobs.csv \
    --output predictions.csv
```

### Using your own input and output paths

Both `--input` and `--output` accept **any path** — relative to the current working directory, absolute, or pointing to a different drive. The tool does not care where the files live, as long as:

- the **input** path is an existing CSV the process can read, and
- the **output** path lives in a directory the process can write to (the directory must already exist — the script will not create parent folders for you).

A few concrete examples:

```bash
# 1) Absolute paths (recommended when scripting / running from cron):
python -m yoe_classifier.cli.score_batch \
    --input  /Users/talha/data/incoming/jobs_2026_06.csv \
    --output /Users/talha/data/outgoing/predictions_2026_06.csv

# 2) Only the rows the model is unsure about — handy for relabeling rounds:
python -m yoe_classifier.cli.score_batch \
    --input  /path/to/big_jobs.csv \
    --output /path/to/needs_review.csv \
    --only-flagged
```

If you keep your trained model somewhere other than the default
`yoe_classifier/artifacts/`, also pass `--artifacts-dir`:

```bash
python -m yoe_classifier.cli.score_batch \
    --input        /data/jobs.csv \
    --output       /data/predictions.csv \
    --artifacts-dir /models/yoe_run_07
```

#### Input CSV requirements

The input CSV must have columns `UniqueJobAdID`, `OrigJobTitle`, `JobDescription` (it also accepts `job_id`, `title`, `description` as fallbacks). Extra columns are ignored. Rows with missing title or description are still scored — they just get less signal.

#### Output CSV columns

The output file is overwritten on every run. Its columns are:

| Column            | Meaning |
|-------------------|---------|
| `job_id`          | Pass-through from input |
| `title`           | Pass-through from input |
| `label`           | Predicted band: 1–4 |
| `confidence`      | Float in [0, 1] — rule confidence if a high-conf rule fired, otherwise model `predict_proba` of the top class |
| `decided_by`      | `rule:<name>`, `rule:<name>+model`, `model`, or `…+executive_guardrail` |
| `evidence_snippet`| Short snippet of source text that triggered the rule (empty if pure model) |
| `evidence_source` | `title`, `description`, or `none` |
| `unsure`          | `true` if the row should be reviewed (low conf, ambiguous margin, or rule/model disagreement) |

### All batch scoring flags

```
--input PATH            required — path to the input jobs CSV
--output PATH           required — path where predictions CSV will be written
--artifacts-dir DIR     default: yoe_classifier/artifacts
--batch-size N          default: 2000  (rows per model.predict_proba call)
--only-flagged          write only rows where unsure=true (for relabeling)
```

`--only-flagged` is the typical workflow when you want to send rows back for human review and grow your gold set.

Throughput is around a few thousand rows/sec on a laptop; progress prints every batch.

---

## Programmatic use

For single-row or interactive scoring:

```python
from yoe_classifier.model import TrainedArtifacts
from yoe_classifier.predict import predict_one, predict_batch

artifacts = TrainedArtifacts.load("yoe_classifier/artifacts")

p = predict_one(
    title="Senior Staff Software Engineer",
    description="We are looking for an engineer with 10+ years of experience...",
    artifacts=artifacts,
)
print(p.label, p.confidence, p.decided_by, p.unsure)

# Batch form — much faster than calling predict_one in a loop
rows = [{"title": t, "description": d} for t, d in pairs]
preds = predict_batch(rows, artifacts)
```

`Prediction` fields match the batch-scoring CSV columns one-for-one.

---

## How the hybrid combiner decides

Given a single row, `predict_batch` does this:

1. Run all rules → get a `RuleHit` or `None`.
2. Run the model → get class probabilities, top class, top-1 confidence, top1–top2 margin.
3. Combine:
   - **Rule confidence ≥ 0.90** → use the rule's label. Flag `unsure` only if it disagrees with the model by ≥ 2 bands.
   - **Lower-confidence rule that agrees with the model** → use the model's label, bump confidence by +0.10, decided_by = `<rule>+model`.
   - **Lower-confidence rule that disagrees** → use the model's label but flag `unsure=true`.
   - **No rule** → use the model's label. Flag `unsure` if confidence < threshold OR top1–top2 margin < 0.12.
4. **Executive-floor guardrail**: if the final label is 1 but the title contains an executive signal (VP, Director, Head of, Chief, etc.) and no high-conf rule fired, force label to 4 and flag `unsure`.

The two key knobs:
- `RULE_HIGH_CONF = 0.90` and `MARGIN_UNSURE = 0.12` in `predict.py`.
- `threshold` is auto-tuned during training and stored in `meta.json`.

---

## Typical workflow

1. **Train**: `python -m yoe_classifier.cli.train`
2. **Inspect**: `python -m yoe_classifier.cli.evaluate` — check macro-F1, kept-only accuracy, per-class numbers.
3. **Score the universe**: `python -m yoe_classifier.cli.score_batch --input jobs.csv --output predictions.csv`
4. **Grow the gold set**: re-run with `--only-flagged` to dump just the uncertain rows, hand-label them, merge into `gold100.json`, and retrain.
5. Iterate.

---

## Troubleshooting

- **"no candidate succeeded" during train** — `requirements.txt` didn't install cleanly. Re-run `pip install -r yoe_classifier/requirements.txt` in your venv.
- **"N rows had no matching job in CSV"** — your label JSON references `job_id`s that aren't in `jobs.csv`. Either the CSV is wrong/stale or you're pointing at the wrong data dir.
- **`FileNotFoundError` when batch-scoring** — the directory you pointed `--output` at doesn't exist yet. Create it with `mkdir -p /path/to/dir` first.
- **Model file is huge / slow to load** — TF-IDF vocabularies dominate the artifact. Drop `max_features` in `features.build_pipeline` if you need to shrink it.
- **Predictions look too cautious (everything flagged)** — threshold may have been pushed high. Look at `meta.json → cv_summary → chosen_threshold` and the threshold sweep in `evaluation.json`.
