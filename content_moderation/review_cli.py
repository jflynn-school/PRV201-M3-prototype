import argparse

from moderation.database import ModerationDB
from moderation.forensics import build_case, case_to_markdown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="demo_moderation.db")
    ap.add_argument("--reviewer", default="moderator")
    args = ap.parse_args()
    db = ModerationDB(args.db)

    queue = db.pending_reviews()
    print(f"{len(queue)} case(s) awaiting review (most severe first).")
    for item in queue:
        print("\n" + "=" * 70)
        print(f"Message #{item['id']}  thread={item['thread_id']}  user={item['user_hash']}  at {item['created_at']}")
        print(f"Text:     {item['text']}")
        print(f"AI view:  {item['explanation'] or 'n/a (classifier unavailable)'}")
        print(f"Routed:   {item['reason']}")
        if item["context_note"]:
            print(f"Context:  {item['context_note']}")
        while True:
            key = input("[r]emove / [a]llow / [c]ase history / [s]kip / [q]uit > ").strip().lower()
            if key == "c":
                print(case_to_markdown(build_case(db, item["user_hash"])))
            elif key in ("r", "a"):
                note = input("note (optional): ")
                db.resolve_review(item["id"], "remove" if key == "r" else "allow", args.reviewer, note)
                print("Saved.")
                break
            elif key == "s":
                break
            elif key == "q":
                return
    print("\nQueue finished.")


if __name__ == "__main__":
    main()
