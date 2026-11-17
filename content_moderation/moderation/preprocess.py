import re
import unicodedata
from dataclasses import dataclass, field
from typing import List

MAX_LEN = 2000

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)
_REPEAT_RE = re.compile(r"(.)\1{3,}")

PII_PATTERNS = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("URL", re.compile(r"(?:https?://|www\.)\S+", re.I)),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")),
    ("ADDRESS", re.compile(
        r"\b\d{1,5}\s+(?:[A-Za-z]+\s+){1,3}"
        r"(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|Boulevard|Blvd|Way)\b\.?"
        r"(?:\s+(?:Apt|Unit|Suite|#)\s*\w+)?", re.I)),
]


@dataclass
class PreprocessResult:
    text: str
    pii_types: List[str] = field(default_factory=list)
    valid: bool = True
    reason: str = ""


def preprocess(raw) -> PreprocessResult:
    if raw is None or not str(raw).strip():
        return PreprocessResult("", valid=False, reason="empty message")

    t = unicodedata.normalize("NFKC", str(raw)).translate(_ZERO_WIDTH)
    t = "".join(ch for ch in t if ch.isprintable() or ch in "\n\t")

    pii = []
    for name, rx in PII_PATTERNS:
        t, n = rx.subn(f"[REDACTED_{name}]", t)
        if n:
            pii.append(name.lower())

    t = _REPEAT_RE.sub(r"\1\1\1", t)
    t = re.sub(r"\s+", " ", t).strip()[:MAX_LEN]
    return PreprocessResult(t, sorted(set(pii)))
