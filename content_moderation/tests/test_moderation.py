import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moderation import Message, build_default_pipeline
from moderation.classifier import ClassificationResult, ClassifierError, estimate_severity
from moderation.context import ContextResult, SimulatedOutageContext
from moderation.database import ModerationDB
from moderation.decision import DecisionEngine
from moderation.preprocess import preprocess


def fake(category="none", conf=0.95, severity=None, tox=None):
    return ClassificationResult(category, conf, {category: conf}, severity or estimate_severity(category, ""),
                                "", [], tox)


class PreprocessTests(unittest.TestCase):
    def test_redacts_pii(self):
        r = preprocess("call 973-555-0142 or mail a.b@school.edu at 42 Maple Street, see http://x.example/y")
        for token in ("[REDACTED_PHONE]", "[REDACTED_EMAIL]", "[REDACTED_ADDRESS]", "[REDACTED_URL]"):
            self.assertIn(token, r.text)
        self.assertNotIn("555", r.text)
        self.assertEqual(r.pii_types, ["address", "email", "phone", "url"])

    def test_normalizes_and_rejects_empty(self):
        self.assertEqual(preprocess("soooooo   bad!!!!!!").text, "sooo bad!!!")
        self.assertFalse(preprocess("   ").valid)
        self.assertFalse(preprocess(None).valid)


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.e = DecisionEngine()

    def test_confident_spam_auto_removed(self):
        self.assertEqual(self.e.decide(fake("spam", 0.97)).action, "auto_remove")

    def test_confident_clean_auto_allowed(self):
        self.assertEqual(self.e.decide(fake("none", 0.95)).action, "auto_allow")

    def test_threat_and_doxxing_always_human(self):
        for cat in ("threat", "doxxing"):
            self.assertEqual(self.e.decide(fake(cat, 0.999)).action, "human_review")

    def test_low_confidence_human(self):
        self.assertEqual(self.e.decide(fake("harassment", 0.7)).action, "human_review")
        self.assertEqual(self.e.decide(fake("none", 0.6)).action, "human_review")

    def test_high_severity_human(self):
        self.assertEqual(self.e.decide(fake("hate_speech", 0.99, "high")).action, "human_review")

    def test_repeat_offender_human(self):
        self.assertEqual(self.e.decide(fake("spam", 0.99), prior_flags=2).action, "human_review")

    def test_context_escalation_human(self):
        ctx = ContextResult("pattern", escalate=True)
        self.assertEqual(self.e.decide(fake("none", 0.99), context=ctx).action, "human_review")

    def test_system_error_fails_safe(self):
        d = self.e.decide(fake("spam", 0.99), system_error="boom")
        self.assertEqual((d.action, d.rules), ("human_review", ["system_failure"]))

    def test_toxicity_disagreement_human(self):
        self.assertEqual(self.e.decide(fake("none", 0.99, tox=0.97)).action, "human_review")


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipe = build_default_pipeline()

    def test_end_to_end_storage_and_review(self):
        r = self.pipe.moderate(Message("I know where you live and I'm coming for you.", "u1", "t1", "th"))
        self.assertEqual(r.decision.action, "human_review")
        pending = [p for p in self.pipe.db.pending_reviews() if p["id"] == r.message_id]
        self.assertEqual(len(pending), 1)
        self.pipe.resolve(r.message_id, "remove", "tester")
        self.assertFalse([p for p in self.pipe.db.pending_reviews() if p["id"] == r.message_id])

    def test_stores_redacted_text_and_hashed_ids(self):
        r = self.pipe.moderate(Message("call me at 973-555-0142", "secret_user_name"))
        df = self.pipe.db.export_dataframe()
        row = df[df.id == r.message_id].iloc[0]
        self.assertNotIn("973", row.text)
        self.assertNotIn("secret_user_name", row.user_hash)

    def test_api_outage_routes_to_human(self):
        p = build_default_pipeline()
        p.moderate(Message("Honestly, you are such a loser, nobody likes you.", "a", "b", "t"))
        p.context = SimulatedOutageContext()
        r = p.moderate(Message("ok whatever", "c", "b", "t"))
        self.assertEqual(r.decision.action, "human_review")
        self.assertIn("context_unavailable", r.decision.reason)

    def test_classifier_failure_routes_to_human(self):
        p = build_default_pipeline()

        class Broken:
            def predict(self, text):
                raise ClassifierError("model offline")

        p.classifier = Broken()
        r = p.moderate(Message("anything", "x"))
        self.assertEqual(r.decision.action, "human_review")
        self.assertIsNone(r.classification)

    def test_repeat_offender_escalates(self):
        p = build_default_pipeline()
        actions = [p.moderate(Message(t, "bully", "victim", "th")).decision.action for t in (
            "Honestly, you are such a loser, nobody likes you.",
            "Seriously, stop posting, you worthless clown.",
            "Everyone is laughing at you, you pathetic freak.")]
        self.assertEqual(actions[-1], "human_review")

    def test_empty_message_ignored(self):
        r = self.pipe.moderate(Message("   ", "u"))
        self.assertIsNone(r.message_id)


class DatabaseTests(unittest.TestCase):
    def test_retention_purge(self):
        db = ModerationDB()
        u = db.hash_id("u")
        db.add_message("t", u, None, None, "old", [], False, "2020-01-01T00:00:00+00:00")
        db.add_message("t", u, None, None, "new", [], False)
        self.assertEqual(db.purge_older_than(30), 1)
        self.assertEqual(len(db.user_timeline(u)), 1)


if __name__ == "__main__":
    unittest.main()
