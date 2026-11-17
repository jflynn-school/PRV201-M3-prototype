import hashlib
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages(
  id INTEGER PRIMARY KEY AUTOINCREMENT, platform TEXT, user_hash TEXT, target_hash TEXT,
  thread_id TEXT, text TEXT, pii_types TEXT, is_duplicate INTEGER DEFAULT 0, created_at TEXT);
CREATE TABLE IF NOT EXISTS classifications(
  message_id INTEGER PRIMARY KEY, category TEXT, severity TEXT, confidence REAL, probabilities TEXT,
  explanation TEXT, toxicity REAL, context_note TEXT, model_version TEXT);
CREATE TABLE IF NOT EXISTS decisions(
  message_id INTEGER PRIMARY KEY, action TEXT, reason TEXT, rules TEXT, status TEXT, final_action TEXT,
  reviewer TEXT, review_note TEXT, decided_at TEXT, reviewed_at TEXT);
CREATE TABLE IF NOT EXISTS audit_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, event TEXT, message_id INTEGER, detail TEXT);
"""
_SEV = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4, None: 0}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ModerationDB:
    def __init__(self, path: str = ":memory:", salt: Optional[str] = None):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.salt = salt or os.environ.get("MOD_HASH_SALT", "dev-only-salt")

    def hash_id(self, raw) -> Optional[str]:
        if raw is None:
            return None
        return hashlib.sha256(f"{self.salt}:{raw}".encode()).hexdigest()[:12]

    def is_duplicate(self, user_hash, text) -> bool:
        r = self.conn.execute("SELECT 1 FROM messages WHERE user_hash=? AND text=? LIMIT 1", (user_hash, text))
        return r.fetchone() is not None

    def add_message(self, platform, user_hash, target_hash, thread_id, text, pii_types, is_dup, created_at=None) -> int:
        cur = self.conn.execute(
            "INSERT INTO messages(platform,user_hash,target_hash,thread_id,text,pii_types,is_duplicate,created_at)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (platform, user_hash, target_hash, thread_id, text, ",".join(pii_types), int(is_dup), created_at or _now()))
        self.conn.commit()
        return cur.lastrowid

    def add_classification(self, mid, res, context_note="", model_version=""):
        self.conn.execute(
            "INSERT OR REPLACE INTO classifications VALUES(?,?,?,?,?,?,?,?,?)",
            (mid, res.category, res.severity, res.confidence, json.dumps(res.probabilities),
             res.explanation, res.toxicity, context_note, model_version))
        self.conn.commit()

    def add_decision(self, mid, decision):
        final = {"auto_remove": "remove", "auto_allow": "allow"}.get(decision.action)
        status = "pending" if decision.action == "human_review" else "auto"
        self.conn.execute(
            "INSERT OR REPLACE INTO decisions(message_id,action,reason,rules,status,final_action,decided_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (mid, decision.action, decision.reason, ",".join(decision.rules), status, final, _now()))
        self.conn.commit()
        self.audit(decision.action, mid, decision.reason)

    def resolve_review(self, mid, final_action: str, reviewer: str, note: str = ""):
        assert final_action in ("allow", "remove")
        self.conn.execute(
            "UPDATE decisions SET status='resolved', final_action=?, reviewer=?, review_note=?, reviewed_at=?"
            " WHERE message_id=?", (final_action, reviewer, note, _now(), mid))
        self.conn.commit()
        self.audit("human_review_resolved", mid, f"{final_action} by {reviewer}")

    def audit(self, event, message_id=None, detail=""):
        self.conn.execute("INSERT INTO audit_log(ts,event,message_id,detail) VALUES(?,?,?,?)",
                          (_now(), event, message_id, detail))
        self.conn.commit()

    def recent_thread(self, thread_id, before_id, limit=10) -> List[dict]:
        if not thread_id:
            return []
        rows = self.conn.execute(
            "SELECT m.id,m.user_hash,m.target_hash,m.text,m.created_at,c.category FROM messages m "
            "LEFT JOIN classifications c ON c.message_id=m.id WHERE m.thread_id=? AND m.id<? "
            "ORDER BY m.id DESC LIMIT ?", (thread_id, before_id, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def prior_flag_count(self, user_hash, exclude_id=-1) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM messages m JOIN classifications c ON c.message_id=m.id "
            "LEFT JOIN decisions d ON d.message_id=m.id WHERE m.user_hash=? AND m.id<>? AND c.category<>'none' "
            "AND (d.final_action IS NULL OR d.final_action='remove')", (user_hash, exclude_id)).fetchone()[0]

    def pending_reviews(self) -> List[dict]:
        rows = self.conn.execute(
            "SELECT m.id,m.user_hash,m.target_hash,m.thread_id,m.text,m.created_at,c.category,c.severity,"
            "c.confidence,c.explanation,c.context_note,d.reason FROM decisions d "
            "JOIN messages m ON m.id=d.message_id LEFT JOIN classifications c ON c.message_id=m.id "
            "WHERE d.status='pending'").fetchall()
        out = [dict(r) for r in rows]
        out.sort(key=lambda r: (-_SEV.get(r["severity"], 0), r["id"]))   # most severe first (triage)
        return out

    def user_of(self, message_id) -> Optional[str]:
        r = self.conn.execute("SELECT user_hash FROM messages WHERE id=?", (message_id,)).fetchone()
        return r["user_hash"] if r else None

    def user_timeline(self, user_hash) -> List[dict]:
        rows = self.conn.execute(
            "SELECT m.id,m.created_at,m.thread_id,m.target_hash,m.text,c.category,c.severity,c.confidence,"
            "d.action,d.final_action FROM messages m LEFT JOIN classifications c ON c.message_id=m.id "
            "LEFT JOIN decisions d ON d.message_id=m.id WHERE m.user_hash=? ORDER BY m.id", (user_hash,)).fetchall()
        return [dict(r) for r in rows]

    def targets_of(self, user_hash) -> List[dict]:
        rows = self.conn.execute(
            "SELECT target_hash, COUNT(*) AS n FROM messages WHERE user_hash=? AND target_hash IS NOT NULL "
            "GROUP BY target_hash ORDER BY n DESC", (user_hash,)).fetchall()
        return [dict(r) for r in rows]

    def linked_users(self, user_hash) -> List[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT m2.user_hash FROM messages m1 JOIN messages m2 ON m1.thread_id=m2.thread_id "
            "WHERE m1.user_hash=? AND m2.user_hash<>m1.user_hash AND m1.thread_id IS NOT NULL", (user_hash,)).fetchall()
        return [r[0] for r in rows]

    def purge_older_than(self, days: int) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
        ids = [r[0] for r in self.conn.execute("SELECT id FROM messages WHERE created_at<?", (cutoff,))]
        for table, col in (("classifications", "message_id"), ("decisions", "message_id"), ("messages", "id")):
            self.conn.executemany(f"DELETE FROM {table} WHERE {col}=?", [(i,) for i in ids])
        self.conn.commit()
        self.audit("retention_purge", None, f"deleted {len(ids)} messages older than {days} days")
        return len(ids)

    def export_dataframe(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT m.id,m.created_at,m.user_hash,m.target_hash,m.thread_id,m.text,c.category,c.severity,"
            "c.confidence,c.explanation,d.action,d.final_action,d.reason,d.reviewer FROM messages m "
            "LEFT JOIN classifications c ON c.message_id=m.id LEFT JOIN decisions d ON d.message_id=m.id "
            "ORDER BY m.id", self.conn)
