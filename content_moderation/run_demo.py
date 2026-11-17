import os

from moderation import Message, build_default_pipeline
from moderation.context import SimulatedOutageContext
from moderation.forensics import build_case, case_to_markdown

DB = "demo_moderation.db"
if os.path.exists(DB):
    os.remove(DB)
pipe = build_default_pipeline(DB)
real_context = pipe.context

SCRIPT = [
    ("t1", "alice", "bob", "Anyone want to study for the calculus exam tonight? lol"),
    ("t1", "bob", "alice", "haha yes, library at 7?"),
    ("t2", "carlos", "dana", "Honestly, you are such a loser, nobody likes you."),
    ("t2", "carlos", "dana", "Seriously, stop posting, you worthless clown."),
    ("t2", "carlos", "dana", "Everyone is laughing at you, you pathetic freak."),
    ("t2", "carlos", "dana", "I know where you live and I'm coming for you."),
    ("t3", "eve", None, "WIN a FREE iPhone!!! Click http://prizes.example/win now!!!"),
    ("t4", "gina", "hank", "Her address is 42 Maple Street Apt 3 and her cell is 973-555-0142, go pay her a visit."),
    ("t5", "ivan", None, "I'm going to kill this presentation tomorrow, wish me luck!"),
    ("t6", "jo", None, "Immigrants are ruining this school and should be kicked out."),
    ("t6", "kim", None, "   "),
]

print(f"{'#':>3} {'sender':7} {'action':13} {'category':12} {'sev':8} {'conf':>5}  message")
for thread, sender, target, text in SCRIPT:
    r = pipe.moderate(Message(text, sender, target, thread))
    c = r.classification
    print(f"{str(r.message_id or '-'):>3} {sender:7} {r.decision.action:13} "
          f"{(c.category if c else '-'):12} {(c.severity if c else '-'):8} {(f'{c.confidence:.0%}' if c else '-'):>5}  "
          f"{r.clean_text[:60]}")
    print(f"      why: {r.decision.reason}")
    if r.context:
        print(f"      context[{r.context.backend}]: {r.context.summary}")

print("\n--- Simulated API outage: a message in a thread that already has flagged content ---")
pipe.context = SimulatedOutageContext()
r = pipe.moderate(Message("ok whatever", "frank", "dana", "t2"))
print(f"{r.decision.action}: {r.decision.reason}")
pipe.context = real_context

print("\n--- Moderator queue (most severe first) ---")
queue = pipe.db.pending_reviews()
for q in queue:
    print(f"#{q['id']} [{q['severity']}] {q['category']}: {q['text'][:70]}\n     AI: {q['explanation']}\n     Routed because: {q['reason']}")

if queue:
    top = queue[0]
    pipe.resolve(top["id"], "remove", "demo_moderator", "confirmed threat")
    print(f"\nModerator removed #{top['id']}.")

print("\n--- Forensics case folder for 'carlos' ---")
print(case_to_markdown(build_case(pipe.db, pipe.db.hash_id("carlos"))))
print(f"\nDatabase saved to {DB}. Run: python review_cli.py --db {DB}")
