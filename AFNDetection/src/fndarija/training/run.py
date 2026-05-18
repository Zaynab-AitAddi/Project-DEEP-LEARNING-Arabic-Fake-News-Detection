from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import inspect
import numpy as np
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

import evaluate

from fndarija.models.classifier import load_tokenizer_and_model


def build_metrics():
    acc = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1.compute(predictions=preds, references=labels, average="binary")["f1"],
        }

    return compute_metrics


def tokenize_splits(
    datasets: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> DatasetDict:
    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    cols = datasets["train"].column_names
    remove_cols = [c for c in cols if c not in {"label"}]
    tokenized = datasets.map(tok, batched=True, remove_columns=remove_cols)
    tokenized = tokenized.rename_column("label", "labels")
    fmt_cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in tokenized["train"].column_names:
        fmt_cols.insert(2, "token_type_ids")
    tokenized.set_format(type="torch", columns=fmt_cols)
    return tokenized


def train_from_config(cfg: dict[str, Any], project_dir: Path) -> None:
    from fndarija.data.loaders import (
        dataframes_to_hf,
        load_arabic_hub,
        load_arabic_local,
        stratified_split_df,
    )

    language = cfg.get("language", "arabic")
    if language != "arabic":
        raise ValueError(f"Only 'arabic' is supported (got {language!r}).")
    seed = int(cfg.get("seed", 42))

    id2label = {0: "fake", 1: "real"}
    label2id = {"fake": 0, "real": 1}

    ds_cfg = cfg["dataset"]
    local_path = ds_cfg.get("local_path")
    lp = Path(local_path).expanduser() if local_path else None
    if lp is not None and not lp.is_absolute():
        lp = project_dir / lp
    if lp is not None and lp.exists():
        df = load_arabic_local(
            lp,
            text_mode=ds_cfg.get("text_mode", "title_plus_text"),
            max_samples=cfg.get("max_samples"),
            seed=seed,
        )
        csv_note = str(lp)
    else:
        df = load_arabic_hub(
            hub_id=ds_cfg["hub_id"],
            split=ds_cfg.get("split", "train"),
            text_mode=ds_cfg.get("text_mode", "title_plus_text"),
            max_samples=cfg.get("max_samples"),
            seed=seed,
        )
        csv_note = ds_cfg["hub_id"]

    if len(df) < 20:
        raise ValueError(f"Trop peu d'exemples après nettoyage: {len(df)} (fichier/source: {csv_note})")

    val_ratio = float(cfg.get("val_ratio", 0.1))
    test_ratio = float(cfg.get("test_ratio", 0.1))
    train_df, val_df, test_df = stratified_split_df(df, "label", val_ratio, test_ratio, seed)
    datasets = dataframes_to_hf(train_df, val_df, test_df)

    model_cfg = cfg["model"]
    tok_model, model = load_tokenizer_and_model(
        model_cfg["pretrained"],
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    max_length = int(model_cfg.get("max_length", 256))
    tokenized = tokenize_splits(datasets, tok_model, max_length=max_length)

    tr_cfg = cfg["training"]
    out_dir = Path(tr_cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = project_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "label_schema.json").open("w", encoding="utf-8") as f:
        json.dump({"id2label": id2label, "label2id": label2id, "source": csv_note}, f, ensure_ascii=False, indent=2)

    eval_strategy = tr_cfg.get("eval_strategy", "steps")
    save_strategy = tr_cfg.get("save_strategy", "steps")

    base_kw = dict(
        output_dir=str(out_dir),
        num_train_epochs=float(tr_cfg.get("num_train_epochs", 3)),
        per_device_train_batch_size=int(tr_cfg.get("per_device_train_batch_size", 8)),
        per_device_eval_batch_size=int(tr_cfg.get("per_device_eval_batch_size", 16)),
        learning_rate=float(tr_cfg.get("learning_rate", 2e-5)),
        weight_decay=float(tr_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(tr_cfg.get("warmup_ratio", 0.06)),
        logging_steps=int(tr_cfg.get("logging_steps", 50)),
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=int(tr_cfg.get("save_total_limit", 2)),
        load_best_model_at_end=bool(tr_cfg.get("load_best_model_at_end", True)),
        metric_for_best_model=tr_cfg.get("metric_for_best_model", "f1"),
        greater_is_better=bool(tr_cfg.get("greater_is_better", True)),
        fp16=bool(tr_cfg.get("fp16", False)),
        seed=seed,
        report_to=[],
    )
    if eval_strategy != "epoch":
        base_kw["eval_steps"] = int(tr_cfg.get("eval_steps", 500))
    if save_strategy != "epoch":
        base_kw["save_steps"] = int(tr_cfg.get("save_steps", 500))

    args = TrainingArguments(**base_kw)

    data_collator = DataCollatorWithPadding(tokenizer=tok_model)

    trainer_kw: dict[str, Any] = dict(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=build_metrics(),
    )
    sig = inspect.signature(Trainer.__init__)
    if "processing_class" in sig.parameters:
        trainer_kw["processing_class"] = tok_model
    elif "tokenizer" in sig.parameters:
        trainer_kw["tokenizer"] = tok_model

    trainer = Trainer(**trainer_kw)

    trainer.train()
    trainer.save_model(str(out_dir))
    tok_model.save_pretrained(str(out_dir))

    metrics = trainer.evaluate(tokenized["test"])
    with (out_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items() if isinstance(v, (float, int))}, f, indent=2)
