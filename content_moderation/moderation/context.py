import json
import os
import urllib.request
from dataclasses import dataclass
from typing import List, Optional


class ContextUnavailable(Exception):
    """Context backend is down, slow, or returned something unusable."""


@dataclass
class ContextResult:
    summary: str
    escalate: bool = False
    banter_likely: bool = False
    backend: str = "heuristic"


_BANTER = ("lol", "haha", "lmao", "jk", "just kidding", "😂", "🤣")


class HeuristicContext:
    name = "heuristic"

    def analyze(self, text: str, history: List[dict], sender_hash: Optional[str] = None) -> ContextResult:
        flagged = [h for h in history if h.get("category") not in (None, "none")]
        flagged_by_sender = [h for h in flagged if h["user_hash"] == sender_hash]
        banter = sum(any(b in h["text"].lower() for b in _BANTER) for h in history)

        if len(flagged_by_sender) >= 2:
            return ContextResult(
                f"Sender has {len(flagged_by_sender)} earlier flagged messages in this thread - "
                "possible sustained pattern.", escalate=True, backend=self.name)
        if banter >= 2 and not flagged:
            return ContextResult("Thread reads as friendly banter (jokes/laughter markers); "
                                 "message may be non-hostile.", banter_likely=True, backend=self.name)
        if flagged:
            return ContextResult(f"{len(flagged)} earlier message(s) in this thread were flagged.",
                                 backend=self.name)
        if history:
            return ContextResult(f"{len(history)} prior message(s) reviewed; nothing notable.", backend=self.name)
        return ContextResult("No prior thread context available.", backend=self.name)


class SimulatedOutageContext:
    name = "simulated-outage"

    def analyze(self, text, history, sender_hash=None):
        raise ContextUnavailable("simulated API outage")


def _prompt(text: str, history: List[dict]) -> str:
    lines = "\n".join(f"{h['user_hash']}: {h['text']}" for h in history[-8:]) or "(no earlier messages)"
    return ("You assist a content moderator. Given the earlier thread and the new message, reply ONLY "
            'with JSON: {"banter": bool, "escalate": bool, "summary": "one sentence"}.\n'
            f"Thread:\n{lines}\n\nNew message: {text}")


def _parse(raw: str, backend: str) -> ContextResult:
    try:
        d = json.loads(raw)
        return ContextResult(str(d.get("summary", ""))[:300], bool(d.get("escalate", False)),
                             bool(d.get("banter", False)), backend)
    except (ValueError, AttributeError, TypeError) as exc:
        raise ContextUnavailable(f"unparseable model output: {exc}") from exc


def _post_json(url, payload, timeout, headers=None):
    req = urllib.request.Request(url, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as exc:  # network, timeout, HTTP, JSON - all treated as "unavailable"
        raise ContextUnavailable(str(exc)) from exc


class OllamaContext:
    name = "ollama"

    def __init__(self, model="llama3.2", url="http://localhost:11434", timeout=10):
        self.model, self.url, self.timeout = model, url, timeout

    def analyze(self, text, history, sender_hash=None):
        data = _post_json(f"{self.url}/api/generate",
                          {"model": self.model, "prompt": _prompt(text, history),
                           "stream": False, "format": "json"}, self.timeout)
        return _parse(data.get("response", ""), self.name)


class OpenAIContext:
    name = "openai"

    def __init__(self, model="gpt-4o-mini", timeout=10):
        self.model, self.timeout = model, timeout

    def analyze(self, text, history, sender_hash=None):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ContextUnavailable("OPENAI_API_KEY not set")
        data = _post_json("https://api.openai.com/v1/chat/completions",
                          {"model": self.model,
                           "messages": [{"role": "user", "content": _prompt(text, history)}],
                           "response_format": {"type": "json_object"}},
                          self.timeout, {"Authorization": f"Bearer {key}"})
        try:
            return _parse(data["choices"][0]["message"]["content"], self.name)
        except (KeyError, IndexError) as exc:
            raise ContextUnavailable(f"unexpected response shape: {exc}") from exc
