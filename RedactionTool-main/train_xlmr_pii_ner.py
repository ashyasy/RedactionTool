from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any

import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

# =========================
# Config
# =========================
MODEL_NAME = "xlm-roberta-base"
DATASET_NAME = "DataikuNLP/kiji-pii-training-data"
OUTPUT_DIR = "./xlmr_pii_ner"
MAX_LENGTH = 256
SEED = 42

# Start with English + Spanish since you want multilingual support.
# Change to {"English"} if you want English-only first.
ALLOWED_LANGUAGES = {"English", "Spanish"}

# Map dataset labels to your project labels
LABEL_MAP = {
    "FIRSTNAME": "PERSON",
    "MIDDLENAME": "PERSON",
    "SURNAME": "PERSON",
    "NAME": "PERSON",

    "COMPANYNAME": "ORGANIZATION",

    "CITY": "LOCATION",
    "STATE": "LOCATION",
    "COUNTRY": "LOCATION",

    "STREET": "ADDRESS",
    "BUILDINGNUM": "ADDRESS",
    "ZIP": "ZIP_CODE",

    "CREDITCARDNUMBER": "CREDIT_CARD",

    "EMAIL": "EMAIL_ADDRESS",
    "EMAIL_ADDRESS": "EMAIL_ADDRESS",
    "PHONENUMBER": "PHONE_NUMBER",
    "PHONE_NUMBER": "PHONE_NUMBER",

    "DATE": "DATE_TIME",
    "BIRTHDATE": "DATE_TIME",
}

TARGET_ENTITY_TYPES = {
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "ADDRESS",
    "ZIP_CODE",
    "CREDIT_CARD",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "DATE_TIME",
}


# =========================
# Utilities
# =========================
def build_bio_labels(entity_types: list[str]) -> tuple[list[str], dict[str, int], dict[int, str]]:
    labels = ["O"]
    for ent in sorted(entity_types):
        labels.append(f"B-{ent}")
        labels.append(f"I-{ent}")
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    return labels, label2id, id2label


def choose_label(raw_label: str) -> str | None:
    mapped = LABEL_MAP.get(raw_label)
    if mapped in TARGET_ENTITY_TYPES:
        return mapped
    return None


def is_boundary_char(ch: str) -> bool:
    return not ch.isalnum() and ch != "_"


def find_all_non_overlapping_spans(text: str, value: str) -> list[tuple[int, int]]:
    """
    Find all matches of a value in text with boundary checks.
    Falls back to plain exact match if boundary-based matching fails.
    """
    value = value.strip()
    if not value:
        return []

    spans: list[tuple[int, int]] = []
    pattern = re.escape(value)

    for m in re.finditer(pattern, text):
        start, end = m.start(), m.end()

        left_ok = start == 0 or is_boundary_char(text[start - 1])
        right_ok = end == len(text) or is_boundary_char(text[end])

        if left_ok and right_ok:
            spans.append((start, end))

    if spans:
        return spans

    return [(m.start(), m.end()) for m in re.finditer(pattern, text)]


def extract_entity_spans(example: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Convert privacy_mask values into character spans in raw text.
    Longer values are processed first to reduce overlap issues.
    """
    text = example["text"]
    privacy_mask = example["privacy_mask"]

    candidates: list[dict[str, Any]] = []
    for item in privacy_mask:
        raw_label = item.get("label")
        value = item.get("value", "")
        mapped = choose_label(raw_label)

        if not mapped or not value or not value.strip():
            continue

        candidates.append(
            {
                "raw_label": raw_label,
                "entity": mapped,
                "value": value,
                "value_len": len(value),
            }
        )

    candidates.sort(key=lambda x: x["value_len"], reverse=True)

    occupied = np.zeros(len(text), dtype=np.int8)
    spans: list[dict[str, Any]] = []

    for cand in candidates:
        matches = find_all_non_overlapping_spans(text, cand["value"])
        for start, end in matches:
            if occupied[start:end].any():
                continue

            occupied[start:end] = 1
            spans.append(
                {
                    "start": start,
                    "end": end,
                    "label": cand["entity"],
                    "text": text[start:end],
                    "raw_label": cand["raw_label"],
                }
            )

    spans.sort(key=lambda x: (x["start"], x["end"]))
    return spans


def tokenize_and_align_labels_factory(tokenizer, label2id: dict[str, int]):
    def tokenize_and_align_labels(example: dict[str, Any]) -> dict[str, Any]:
        text = example["text"]
        spans = extract_entity_spans(example)

        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            return_offsets_mapping=True,
        )

        offsets = tokenized["offset_mapping"]
        labels = []
        span_idx = 0

        for start, end in offsets:
            # special tokens
            if start == end:
                labels.append(-100)
                continue

            while span_idx < len(spans) and spans[span_idx]["end"] <= start:
                span_idx += 1

            if span_idx >= len(spans):
                labels.append(label2id["O"])
                continue

            current = spans[span_idx]

            if not (end <= current["start"] or start >= current["end"]):
                prefix = "B" if start == current["start"] else "I"
                labels.append(label2id[f"{prefix}-{current['label']}"])
            else:
                labels.append(label2id["O"])

        tokenized["labels"] = labels
        tokenized.pop("offset_mapping")
        return tokenized

    return tokenize_and_align_labels


def compute_metrics_factory(id2label: dict[int, str]):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        true_predictions = []
        true_labels = []

        for pred_row, label_row in zip(predictions, labels):
            pred_labels = []
            gold_labels = []

            for pred_id, label_id in zip(pred_row, label_row):
                if label_id == -100:
                    continue
                pred_labels.append(id2label[int(pred_id)])
                gold_labels.append(id2label[int(label_id)])

            true_predictions.append(pred_labels)
            true_labels.append(gold_labels)

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }

    return compute_metrics


def print_dataset_summary(ds: DatasetDict) -> None:
    print("\nDataset summary")
    print(ds)

    lang_counts = Counter(ds["train"]["language"])
    print("\nTop train languages:")
    for lang, count in lang_counts.most_common(10):
        print(f"  {lang}: {count}")


def filter_languages(ds: DatasetDict) -> DatasetDict:
    return DatasetDict({
        split: dataset.filter(lambda x: x.get("language") in ALLOWED_LANGUAGES)
        for split, dataset in ds.items()
    })


def preview_alignment(ds, n: int = 3) -> None:
    print("\nPreview of extracted spans")
    for i in range(min(n, len(ds["train"]))):
        ex = ds["train"][i]
        spans = extract_entity_spans(ex)
        print(f"\nExample {i}")
        print("Text:", ex["text"][:400].replace("\n", " "))
        print("Spans:", spans[:10])


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Loading dataset...")
    ds = load_dataset(DATASET_NAME)

    print_dataset_summary(ds)
    ds = filter_languages(ds)

    if len(ds["train"]) == 0 or len(ds["test"]) == 0:
        raise ValueError("Dataset became empty after language filtering.")

    print("\nAfter language filtering:")
    print(ds)

    preview_alignment(ds, n=2)

    entity_types = sorted(TARGET_ENTITY_TYPES)
    all_labels, label2id, id2label = build_bio_labels(entity_types)

    with open(os.path.join(OUTPUT_DIR, "label_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": MODEL_NAME,
                "allowed_languages": sorted(ALLOWED_LANGUAGES),
                "target_entity_types": entity_types,
                "labels": all_labels,
                "label2id": label2id,
                "id2label": {str(k): v for k, v in id2label.items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    print("Tokenizing and aligning labels...")
    tokenize_fn = tokenize_and_align_labels_factory(tokenizer, label2id)

    tokenized_ds = ds.map(
        tokenize_fn,
        remove_columns=ds["train"].column_names,
        desc="Tokenizing dataset",
    )

    print("\nTokenized dataset:")
    print(tokenized_ds)

    print("\nLoading model...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    compute_metrics = compute_metrics_factory(id2label)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")
    trainer.train()

    print("\nEvaluating best model...")
    metrics = trainer.evaluate()
    print(metrics)

    print("\nSaving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(os.path.join(OUTPUT_DIR, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDone. Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()