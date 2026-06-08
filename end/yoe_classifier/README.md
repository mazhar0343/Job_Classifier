# YoE Classifier

Hybrid years-of-experience classifier for job descriptions. Predicts a label
from `1` to `4`:

| Label | Range          | Bucket            |
|-------|----------------|-------------------|
| 1     | 0-2 years      | entry / no exp    |
| 2     | 3-5 years      | early career      |
| 3     | 6-15 years     | mid career        |
| 4     | 15+ years      | senior            |

It also returns a calibrated `confidence` and an `unsure` flag so low-confidence
predictions can be routed to a human.

## How it works

1. **Rules engine** (`rules.py`) extracts explicit signals from the JD with
   regex: `"5+ years"`, `"3 to 5 years"`, `"no experience required"`,
   `MS + 2 yrs OR 5 yrs`, plus title seniority lexicon. Most labeled examples
   were decided this way (~88%), so rules cover the majority at very high
   precision and provide an evidence snippet for free.
2. **ML fallback** (`features.py` + `model.py`) is a calibrated multinomial
   logistic regression on TF-IDF (title word, JD word, title char n-grams)
   plus a handful of engineered numeric features. Used when rules don't fire
   or when we want a second opinion.
3. **Hybrid combiner** (`predict.py`) returns the rule's answer when it is
   strong, otherwise the model's argmax. The `unsure` flag is set when the
   model's top probability is below a tuned threshold or the top-2 margin is
   small, or when rules and model disagree by 2 or more buckets.

## Layout

```
yoe_classifier/
  rules.py              # regex YoE / negation / education / title seniority
  features.py           # TF-IDF + engineered numerics
  model.py              # calibrated LogReg (+ optional LinearSVC / LightGBM)
  predict.py            # hybrid combiner
  cli/
    train.py            # train, persist artifacts, pick best estimator by CV
    evaluate.py         # 5-fold CV, confusion matrix, coverage-vs-accuracy
    score_batch.py      # batch score a CSV of jobs
  data/                 # symlinks to inputs (jobs.csv, gold100.json, silver400.json)
  artifacts/            # model.joblib, vectorizer.joblib, threshold.json
```

## Quick start

```bash
pip install -r requirements.txt

# 1. Train (fits on gold + silver, runs CV, prints picked model)
python -m yoe_classifier.cli.train

# 2. Evaluate (5-fold CV report on gold + coverage-vs-accuracy curve)
python -m yoe_classifier.cli.evaluate

# 3. Batch score the 5000-row CSV
python -m yoe_classifier.cli.score_batch \
    --input data/jobs.csv \
    --output predictions.csv
```

Output schema mirrors the gold/silver labeling format:

```
job_id, title, label, confidence, decided_by, evidence_snippet, evidence_source, unsure
```
