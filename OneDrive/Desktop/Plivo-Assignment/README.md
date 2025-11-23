# PII NER Assignment Skeleton

This repo is a skeleton for a token-level NER model that tags PII in STT-style transcripts.

## Submission

The submission folder contains the following files
- `**dev.jsonl**`: This file contains 200 dev datapoints with labels
- `**dev_pred.json**`: This file contains the model NER Predictions for all the above 200 datapoints in the same order
- `**metrics.txt**`: This file contains the prediction metrics (Precision,F1,etc:) for all the label classes
- `**latency.txt**`: This file contains the p50 and p95 latency values for 50 runs

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py \
  --model_name distilbert-base-uncased \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out
```

## Predict

```bash
python src/predict.py \
  --model_dir out \
  --input data/dev.jsonl \
  --output out/dev_pred.json
```

## Evaluate

```bash
python src/eval_span_f1.py \
  --gold data/dev.jsonl \
  --pred out/dev_pred.json
```

## Measure latency

```bash
python src/measure_latency.py \
  --model_dir out \
  --input data/dev.jsonl \
  --runs 50
```

Your task in the assignment is to modify the model and training code to improve entity and PII detection quality while keeping **p95 latency below ~20 ms** per utterance (batch size 1, on a reasonably modern CPU).

