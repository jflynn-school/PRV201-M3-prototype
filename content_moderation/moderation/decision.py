from dataclasses import dataclass, field
from typing import List, Optional

SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


@dataclass
class Policy:
    auto_allow_conf: float = 0.80
    auto_remove_conf: float = 0.90
    max_auto_severity: str = "medium"
    always_review_categories: tuple = ("threat", "doxxing")
    auto_remove_categories: tuple = ("spam", "harassment", "hate_speech")
    repeat_offender_limit: int = 2
    toxicity_disagreement: float = 0.90


@dataclass
class Decision:
    action: str
    reason: str
    rules: List[str] = field(default_factory=list)


class DecisionEngine:
    def __init__(self, policy: Optional[Policy] = None):
        self.policy = policy or Policy()

    def decide(self, result, *, prior_flags: int = 0, context=None, system_error: Optional[str] = None) -> Decision:
        p = self.policy
        review = lambda reason, rule: Decision("human_review", reason, [rule])

        if system_error:
            return review(f"System failure, failing safe to a human: {system_error}", "system_failure")
        if result is None:
            return review("No classification available.", "no_result")

        cat, conf = result.category, result.confidence

        if cat in p.always_review_categories:
            return review(f"{cat} is always reviewed by a human.", "always_review_category")
        if SEVERITY_RANK[result.severity] > SEVERITY_RANK[p.max_auto_severity]:
            return review(f"Severity '{result.severity}' exceeds the automatic limit.", "severity_limit")
        if context is not None and context.escalate:
            return review(f"Context escalation: {context.summary}", "context_escalation")
        if cat != "none" and prior_flags >= p.repeat_offender_limit:
            return review(f"User already has {prior_flags} flagged messages (repeat pattern).", "repeat_offender")
        if cat == "none" and result.toxicity is not None and result.toxicity >= p.toxicity_disagreement:
            return review("Classifier says clean but toxicity model disagrees.", "model_disagreement")

        if cat == "none":
            if conf >= p.auto_allow_conf:
                return Decision("auto_allow", f"No violation, {conf:.0%} confidence.", ["confident_clean"])
            return review(f"Uncertain ({conf:.0%} confidence).", "low_confidence")

        if cat in p.auto_remove_categories:
            if conf >= p.auto_remove_conf:
                return Decision("auto_remove", f"Clear {cat.replace('_', ' ')} violation, {conf:.0%} confidence.",
                                ["confident_violation"])
            return review(f"Possible {cat.replace('_', ' ')} but only {conf:.0%} confident.", "low_confidence")

        return review("No automatic rule applies.", "default_to_human")
