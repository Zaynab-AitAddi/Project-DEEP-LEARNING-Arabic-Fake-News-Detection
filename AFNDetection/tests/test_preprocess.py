import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from fndarija.data.preprocess import compose_document, normalize_arabic_text


def test_normalize_strips_diacritics():
    raw = "\u0645\u064e\u0631\u064f\u062d\u064e\u0628\u064b\u0627"
    out = normalize_arabic_text(raw)
    assert "\u064e" not in out


def test_compose_title_body():
    t = compose_document("عنوان", "نص الخبر", "title_plus_text")
    assert "عنوان" in t and "نص" in t
