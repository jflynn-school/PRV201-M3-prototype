"""Stage 3 - Analysis layer: violation classifier.

A scikit-learn TF-IDF (word + character n-gram) logistic-regression model that
outputs category, confidence, severity and a short explanation built from the
terms that pushed the model toward its answer.

An optional pretrained toxicity model (Hugging Face) can be plugged in as a
second opinion; it is never required for the prototype to run.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

CATEGORIES = ["none", "harassment", "hate_speech", "threat", "doxxing", "spam"]

_WEAPONS = ("knife", "gun", "weapon", "bomb", "shoot", "stab")
_VIOLENCE = ("kill", "hurt", "beat", "dead", "destroy", "end you", "make you pay", "bleed")
_INSULTS = ("loser", "idiot", "moron", "freak", "pathetic", "worthless", "clown",
            "creep", "fraud", "joke", "waste of space", "embarrassment")


class ClassifierError(Exception):
    """Raised when the model cannot produce a result (treated as a system failure)."""


@dataclass
class ClassificationResult:
    category: str
    confidence: float
    probabilities: Dict[str, float]
    severity: str
    explanation: str
    top_terms: List[str] = field(default_factory=list)
    toxicity: Optional[float] = None


def estimate_severity(category: str, text: str) -> str:
    """Rule-based severity tiers (none/low/medium/high/critical)."""
    t = text.lower()
    if category == "none":
        return "none"
    if category == "threat":
        if any(w in t for w in _WEAPONS) or "[redacted_address]" in t:
            return "critical"
        return "high"
    if category == "doxxing":
        return "high"
    if category == "hate_speech":
        return "high" if any(w in t for w in _VIOLENCE) else "medium"
    if category == "harassment":
        return "medium" if sum(w in t for w in _INSULTS) >= 2 else "low"
    return "low"  # spam


class PretrainedToxicity:
    """Optional wrapper around a Hugging Face toxicity model (needs `transformers`)."""

    def __init__(self, model: str = "unitary/toxic-bert"):
        from transformers import pipeline  # imported lazily; optional dependency
        self._pipe = pipeline("text-classification", model=model, top_k=None)

    def score(self, text: str) -> float:
        out = self._pipe(text[:512])[0]
        return max((d["score"] for d in out if d["label"].lower() == "toxic"), default=0.0)


class ViolationClassifier:
    model_version = "tfidf-logreg-v1"

    def __init__(self, toxicity_model=None):
        self.toxicity_model = toxicity_model
        self.pipe: Optional[Pipeline] = None
        self._names = None

    def fit(self, texts, labels):
        self.pipe = Pipeline([
            ("features", FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
            ])),
            ("clf", LogisticRegression(C=20, max_iter=3000)),
        ])
        self.pipe.fit(texts, labels)
        self._names = self.pipe.named_steps["features"].get_feature_names_out()
        return self

    def _top_terms(self, text: str, category: str, k: int = 3) -> List[str]:
        vec = self.pipe.named_steps["features"]
        clf = self.pipe.named_steps["clf"]
        ci = list(clf.classes_).index(category)
        row = vec.transform([text]).tocoo()
        scored = [(self._names[j][len("word__"):], v * clf.coef_[ci][j])
                  for j, v in zip(row.col, row.data) if self._names[j].startswith("word__")]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda s: -s[1])
        return [name for name, _ in scored[:k]]

    def predict(self, text: str) -> ClassificationResult:
        if self.pipe is None:
            raise ClassifierError("model has not been trained")
        try:
            proba = self.pipe.predict_proba([text])[0]
            classes = list(self.pipe.classes_)
            best = int(proba.argmax())
            category, conf = classes[best], float(proba[best])
            terms = self._top_terms(text, category)
            tox = self.toxicity_model.score(text) if self.toxicity_model else None
        except Exception as exc:  # any model failure must fail safe
            raise ClassifierError(str(exc)) from exc

        severity = estimate_severity(category, text)
        if category == "none":
            expl = f"No policy violation detected ({conf:.0%} confidence)."
        else:
            signals = ", ".join(f"'{t}'" for t in terms) or "overall phrasing"
            expl = (f"Likely {category.replace('_', ' ')} ({conf:.0%} confidence, "
                    f"{severity} severity). Key signals: {signals}.")
        return ClassificationResult(category, conf, dict(zip(classes, map(float, proba))),
                                    severity, expl, terms, tox)
