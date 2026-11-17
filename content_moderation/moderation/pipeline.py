import time
from dataclasses import dataclass, field
from typing import List, Optional

from .classifier import ClassificationResult, ClassifierError, ViolationClassifier
from .context import ContextResult, ContextUnavailable, HeuristicContext, OllamaContext, OpenAIContext
from .database import ModerationDB
from .dataset import build_training_set
from .decision import Decision, DecisionEngine, Policy
from .preprocess import preprocess


@dataclass
class Message:
    text: str
    user_id: str
    target_id: Optional[str] = None
    thread_id: Optional[str] = None
    platform: str = "demo"
    timestamp: Optional[str] = None


@dataclass
class ModerationResult:
    message_id: Optional[int]
    clean_text: str
    decision: Decision
    classification: Optional[ClassificationResult]
    context: Optional[ContextResult]
    latency_ms: float
    pii_types: List[str] = field(default_factory=list)


class ModerationPipeline:
    def __init__(self, classifier, db: ModerationDB, context=None, policy: Optional[Policy] = None):
        self.classifier, self.db, self.context = classifier, db, context
        self.engine = DecisionEngine(policy)

    def _needs_context(self, cls, history) -> bool:
        uncertain = cls.confidence < self.engine.policy.auto_remove_conf
        flagged_history = any(h.get("category") not in (None, "none") for h in history)
        return uncertain or flagged_history

    def moderate(self, msg: Message) -> ModerationResult:
        t0 = time.perf_counter()
        pre = preprocess(msg.text)
        if not pre.valid:
            return ModerationResult(None, "", Decision("auto_allow", f"Ignored: {pre.reason}", ["invalid_input"]),
                                    None, None, 0.0)

        user_h, target_h = self.db.hash_id(msg.user_id), self.db.hash_id(msg.target_id)
        dup = self.db.is_duplicate(user_h, pre.text)
        mid = self.db.add_message(msg.platform, user_h, target_h, msg.thread_id, pre.text,
                                  pre.pii_types, dup, msg.timestamp)

        cls = ctx = None
        error = None
        try:
            cls = self.classifier.predict(pre.text)
        except ClassifierError as exc:
            error = f"classifier_failure: {exc}"

        history = self.db.recent_thread(msg.thread_id, mid)
        if cls is not None and self.context is not None and self._needs_context(cls, history):
            try:
                ctx = self.context.analyze(pre.text, history, user_h)
            except ContextUnavailable as exc:
                error = f"context_unavailable: {exc}"

        prior = self.db.prior_flag_count(user_h, mid)
        decision = self.engine.decide(cls, prior_flags=prior, context=ctx, system_error=error)
        latency = (time.perf_counter() - t0) * 1000

        if cls is not None:
            self.db.add_classification(mid, cls, ctx.summary if ctx else "",
                                       getattr(self.classifier, "model_version", ""))
        self.db.add_decision(mid, decision)
        if dup:
            self.db.audit("duplicate_message", mid, "identical text previously posted by this user")
        return ModerationResult(mid, pre.text, decision, cls, ctx, latency, pre.pii_types)

    def resolve(self, message_id: int, action: str, reviewer: str, note: str = ""):
        self.db.resolve_review(message_id, action, reviewer, note)


def build_default_pipeline(db_path: str = ":memory:", context: str = "ollama",
                           policy: Optional[Policy] = None) -> ModerationPipeline:
    texts, labels = build_training_set()
    clf = ViolationClassifier().fit(texts, labels)
    ctx = {"heuristic": HeuristicContext, "ollama": OllamaContext, "openai": OpenAIContext,
           "none": lambda: None}[context]()
    return ModerationPipeline(clf, ModerationDB(db_path), ctx, policy)
