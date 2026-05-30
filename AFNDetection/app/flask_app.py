from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fndarija.data.preprocess import normalize_arabic_text  # noqa: E402
from fndarija.inference.predict import load_classifier, predict_text  # noqa: E402

AR_UI = {
    "app_title": "AFNDetection - كشف الأخبار المزيفة",
    "subtitle": "فحص الأخبار بالعربية الفصحى",
    "caption": (
        "تصنيف ثنائي: مزيف مقابل حقيقي لنصوص إخبارية بالعربية الفصحى "
        "(نموذج AraBERT مدرب على مجموعة عامة من Hugging Face)."
    ),
    "input_label": "الصق النص العربي هنا (عنوان، مقال، منشور...)",
    "input_placeholder": "مثال: ذكرت بعض المصادر أن الحكومة أعلنت عن...",
    "analyze": "تحليل النص",
    "result_title": "نتيجة التحليل",
    "confidence": "احتمالات الفئات",
    "disclaimer": (
        "«مزيف» يعني أن النموذج يرجح، ضمن تدريبه، أن النص مضلل أو غير موثوق؛ "
        "«حقيقي» يرجح الصدق. راجع المصادر الأصلية والسياق دائما."
    ),
    "limits_md": (
        "قيود مهمة: النموذج مدرب على نطاق محدد بالعربية الفصحى؛ قد تختلف الدقة "
        "في لهجات أخرى أو مواضيع جديدة. هذه الأداة تعليمية وليست بديلا عن التحقق من الحقائق."
    ),
    "empty_warn": "يرجى إدخال نص قبل الضغط على «تحليل النص».",
    "non_arabic_warn": (
        "لا يبدو أن النص عربيا بشكل واضح (قليل من الحروف العربية أو لا يوجد). "
        "النموذج مدرب على العربية الفصحى؛ النتيجة قد تكون غير موثوقة إذا كان الإدخال بلغة أخرى."
    ),
    "load_error": "تعذر تحميل النموذج. تأكد من إعداد المسار في configs/app.yaml.",
    "model_error": "المسار لا يحتوي على نموذج صالح (مطلوب config.json وملفات الأوزان).",
}

LABEL_AR = {
    "fake": "مزيف",
    "real": "حقيقي",
}

_AR_LETTERS = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\ufb50-\ufdff\ufe70-\ufeff]"
)
_LATIN_LETTERS = re.compile(r"[A-Za-z\u00C0-\u024F]")


def _is_primarily_arabic_script(text: str) -> bool:
    if not text.strip():
        return True
    arabic = len(_AR_LETTERS.findall(text))
    latin = len(_LATIN_LETTERS.findall(text))
    total = arabic + latin
    if total < 4:
        return True
    if arabic == 0:
        return False
    return (arabic / total) >= 0.30


def _label_ar(en: str) -> str:
    return LABEL_AR.get(str(en).strip().lower(), en)


def _load_app_config() -> dict:
    cfg_path = _ROOT / "configs" / "app.yaml"
    if not cfg_path.exists():
        return {}
    with cfg_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return raw if isinstance(raw, dict) else {}


def _resolve_checkpoint() -> Path:
    app_cfg = _load_app_config()
    rel = app_cfg.get("arabic_checkpoint") or "checkpoints/arabic"
    ckpt = Path(rel)
    if not ckpt.is_absolute():
        ckpt = _ROOT / ckpt
    return ckpt


def _id2label_map(model, schema: dict) -> dict[int, str]:
    raw = {}
    if isinstance(schema.get("id2label"), dict):
        raw = schema["id2label"]
    if not raw:
        cfg = getattr(model.config, "id2label", None)
        raw = dict(cfg) if cfg else {}
    out: dict[int, str] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = str(v)
        except (TypeError, ValueError):
            continue
    return out


@lru_cache(maxsize=1)
def _load_runtime():
    ckpt = _resolve_checkpoint()
    if not (ckpt / "config.json").exists():
        raise FileNotFoundError(f"{AR_UI['model_error']} ({ckpt})")
    tokenizer, model, device, schema = load_classifier(str(ckpt))
    id2label = _id2label_map(model, schema)
    max_len = getattr(model.config, "max_position_embeddings", 256) or 256
    max_len = min(int(max_len), 256)
    return {
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "id2label": id2label,
        "max_len": max_len,
    }


app = Flask(
    __name__,
    template_folder=str(_ROOT / "app" / "web" / "templates"),
    static_folder=str(_ROOT / "app" / "web" / "static"),
)


@app.get("/")
def home():
    return render_template("index.html", ui=AR_UI)


@app.get("/logo")
def logo():
    logo_path = _ROOT / "docs" / "images" / "LOGO.png"
    if not logo_path.exists():
        return jsonify({"error": "Logo not found"}), 404
    return send_file(logo_path, mimetype="image/png")


@app.get("/api/health")
def health():
    try:
        _load_runtime()
        return jsonify({"status": "ok"})
    except Exception as exc:  # pragma: no cover
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    raw_text = str(payload.get("text", ""))
    normalized = normalize_arabic_text(raw_text)
    if not normalized.strip():
        return jsonify({"error": AR_UI["empty_warn"]}), 400

    warning = None
    if not _is_primarily_arabic_script(normalized):
        warning = AR_UI["non_arabic_warn"]

    try:
        runtime = _load_runtime()
        label, probs, _ = predict_text(
            normalized,
            runtime["tokenizer"],
            runtime["model"],
            runtime["device"],
            max_length=runtime["max_len"],
        )
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"{AR_UI['load_error']} ({exc})"}), 500

    id2label = runtime["id2label"]
    order = sorted(id2label.keys()) if id2label else list(range(len(probs)))
    probabilities: list[dict] = []
    for i in order:
        if i >= len(probs):
            continue
        label_en = id2label.get(i, f"class_{i}")
        p = float(probs[i])
        probabilities.append(
            {
                "id": i,
                "label_en": label_en,
                "label_ar": _label_ar(label_en),
                "probability": p,
                "percentage": round(max(0.0, min(100.0, p * 100)), 1),
            }
        )

    return jsonify(
        {
            "prediction_en": label,
            "prediction_ar": _label_ar(label),
            "probabilities": probabilities,
            "warning": warning,
            "disclaimer": AR_UI["disclaimer"],
            "limits": AR_UI["limits_md"],
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=True)
