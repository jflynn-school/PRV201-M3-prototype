from collections import Counter

from .database import ModerationDB

_SEV = ["none", "low", "medium", "high", "critical"]


def build_case(db: ModerationDB, user_hash: str) -> dict:
    timeline = db.user_timeline(user_hash)
    flagged = [t for t in timeline if t["category"] not in (None, "none")]
    sev = max((t["severity"] for t in flagged), key=lambda s: _SEV.index(s), default="none")
    return {
        "subject": user_hash,
        "total_messages": len(timeline),
        "flagged_messages": len(flagged),
        "categories": dict(Counter(t["category"] for t in flagged)),
        "max_severity": sev,
        "targets": db.targets_of(user_hash),
        "linked_users": db.linked_users(user_hash),
        "timeline": timeline,
    }


def case_to_markdown(case: dict) -> str:
    targets = ", ".join(t["target_hash"] + " (x" + str(t["n"]) + ")" for t in case["targets"]) or "none"
    lines = [f"# Case folder - user {case['subject']}",
             f"- Messages on record: {case['total_messages']} ({case['flagged_messages']} flagged)",
             f"- Categories: {case['categories'] or 'none'}",
             f"- Highest severity: {case['max_severity']}",
             f"- Targeted accounts: {targets}",
             f"- Linked accounts (shared threads): {', '.join(case['linked_users']) or 'none'}",
             "", "## Timeline"]
    for t in case["timeline"]:
        outcome = t["final_action"] or ("pending review" if t["action"] == "human_review" else t["action"])
        lines.append(f"- #{t['id']} {t['created_at']} [{t['category']}/{t['severity']}] -> {outcome}: {t['text'][:90]}")
    return "\n".join(lines)
