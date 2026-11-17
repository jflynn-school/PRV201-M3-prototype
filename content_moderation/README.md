# AI-Based Automated Content Moderation & Triage - Python Prototype

Hybrid moderation: an AI classifier handles clear-cut cases, and a human moderator
gets everything uncertain, serious, or anything that fails. Built to match the M2 design
and diagram (stages 1-7).

## Quick start
```bash
pip install -r requirements.txt
python run_demo.py                  # scripted walkthrough with fictional messages
python evaluate.py                  # manual vs. automated vs. hybrid comparison (writes results/)
python review_cli.py                # act as the moderator on demo_moderation.db
python -m unittest discover -s tests
```

## How the code maps to the M2 diagram
| Diagram stage | File | What it does |
|---|---|---|
| 1. Content ingestion | `pipeline.py` (`Message`) | Platform-agnostic message object: text, user, target, thread, timestamp |
| 2. Preprocessing & privacy | `preprocess.py` | Unicode/whitespace cleanup, redacts emails, phones, URLs, addresses |
| 3. Analysis layer | `classifier.py`, `context.py` | TF-IDF + logistic regression -> category, severity, confidence, explanation; optional Ollama / OpenAI / Hugging Face toxicity hooks |
| 4. Context & case database | `database.py` | SQLite: messages, classifications, decisions; hashed user ids; retention purge; pandas export |
| 5. Decision engine | `decision.py` | Auditable routing rules (see below) |
| 6. Human review | `review_cli.py` | Severity-ordered queue with AI explanation, context, and case history |
| 7. Audit & forensics | `database.py` (`audit_log`), `forensics.py` | Every action logged; per-user case folders (timeline, targets, linked accounts) |

## Decision rules (`Policy` in `decision.py`, all configurable)
A message goes to a **human** if any of these hold (checked in this order):
1. A system failure occurred (classifier or LLM/API unavailable)
2. Category is `threat` or `doxxing`
3. Severity is above `medium`
4. Context analysis flags a sustained pattern
5. The user already has 2+ flagged messages
6. The optional toxicity model disagrees with a "clean" verdict
7. Confidence is below 80% (clean) / 90% (violation)

Otherwise: confident clean -> `auto_allow`; confident spam/harassment/hate speech -> `auto_remove`.
Context analysis can only enrich or escalate a case; it never auto-approves one.

## Using a real LLM for context (optional)
```python
build_default_pipeline(context="ollama")   # needs `ollama serve` and a pulled model (default llama3.1)
build_default_pipeline(context="openai")   # needs OPENAI_API_KEY
```
Only redacted text is sent. Any timeout/error/bad output routes the message to a human.

Optional pretrained toxicity second opinion (needs `pip install transformers torch`):
```python
from moderation.classifier import ViolationClassifier, PretrainedToxicity
clf = ViolationClassifier(toxicity_model=PretrainedToxicity())
```

## Evaluation (`evaluate.py`)
Compares **manual only**, **fully automated**, and **hybrid** on the same labeled set, against the
M2.6 goals (accuracy >= 85%, false-positive rate < 5%, latency < 5 s, less manual review than manual-only).

Important caveats - say these in your write-up:
- Data is **fictional**: training messages are template-generated; the 58-message test set is hand-written
  and separate (14 benign, 44 violations, 17 "hard" cases with sarcasm/idioms/context). That is too small
  for strong statistical claims.
- Human moderators are **simulated** (95% accurate, 30 s per message by default; change with
  `--human-acc` / `--review-seconds`). Manual-only results depend entirely on these assumptions.
- Latency is local-model latency only; LLM backends will be slower (still expected under the 5 s target for the
  uncertain minority).
- For a stronger M3 result, replace the dataset with a public labeled corpus (e.g. Jigsaw toxic comments, HateXplain)
  and re-run.

## Privacy choices
Salted-hash user ids (`MOD_HASH_SALT` env var), redacted text only stored, `purge_older_than(days)` retention,
no real user data used anywhere in the repo.

## Known limitations / next steps
- Small training set; swap in a pretrained toxicity model + public datasets.
- Doxxing without a phone/email/address pattern ("found his dorm and parents' info") is easy to miss.
- Review UI is a terminal tool; a web UI (Flask/Streamlit) is the natural next step.
- Relationship analysis between users (M2.7 stretch goal) is limited to shared threads and targets.
