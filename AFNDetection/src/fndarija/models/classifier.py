from __future__ import annotations

from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def load_tokenizer_and_model(
    pretrained: str,
    num_labels: int,
    id2label: dict[int, str],
    label2id: dict[str, int],
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return tokenizer, model
