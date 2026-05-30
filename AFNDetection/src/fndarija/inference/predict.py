from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_classifier(checkpoint_dir: str | Path, device: str | None = None):
    checkpoint_dir = Path(checkpoint_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir))
    model.to(device)
    model.eval()
    schema_path = checkpoint_dir / "label_schema.json"
    schema = {}
    if schema_path.exists():
        with schema_path.open(encoding="utf-8") as f:
            schema = json.load(f)
    return tok, model, device, schema


@torch.inference_mode()
def predict_text(text: str, tokenizer, model, device: str, max_length: int = 256):
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(**enc).logits[0]
    probs = torch.softmax(logits, dim=-1).tolist()
    pred_id = int(torch.argmax(logits))
    raw_map = getattr(model.config, "id2label", None) or {}
    # Hugging Face peut stocker les clés en str
    label = raw_map.get(pred_id, raw_map.get(str(pred_id), str(pred_id)))
    return label, probs, pred_id
