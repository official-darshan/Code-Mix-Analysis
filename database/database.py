from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "database" / "support_analytics.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_message TEXT NOT NULL,
                agent_response TEXT,
                language TEXT,
                intent TEXT,
                sentiment TEXT,
                resolution TEXT,
                escalation TEXT,
                escalation_probability REAL,
                priority TEXT,
                confidence REAL,
                status TEXT,
                intent_source TEXT,
                sentiment_source TEXT,
                resolution_source TEXT,
                escalation_source TEXT,
                created_at TEXT
            )
            """
        )

        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(conversations)").fetchall()
        }

        additions = {
            "agent_response": "TEXT",
            "confidence": "REAL",
            "status": "TEXT",
            "intent_source": "TEXT",
            "sentiment_source": "TEXT",
            "resolution_source": "TEXT",
            "escalation_source": "TEXT",
        }

        for column, dtype in additions.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE conversations ADD COLUMN {column} {dtype}")

        conn.commit()


def save_analysis(message: str, result: dict, agent_response: str = ""):
    init_db()

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO conversations (
                customer_message,
                agent_response,
                language,
                intent,
                sentiment,
                resolution,
                escalation,
                escalation_probability,
                priority,
                confidence,
                status,
                intent_source,
                sentiment_source,
                resolution_source,
                escalation_source,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message,
                agent_response,
                result.get("language"),
                result.get("intent"),
                result.get("sentiment"),
                result.get("resolution"),
                result.get("escalation"),
                result.get("escalation_probability"),
                result.get("priority"),
                result.get("confidence"),
                result.get("status"),
                result.get("intent_source"),
                result.get("sentiment_source"),
                result.get("resolution_source"),
                result.get("escalation_source"),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()


def fetch_all(limit: int = 500):
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                id,
                customer_message,
                agent_response,
                language,
                intent,
                sentiment,
                resolution,
                escalation,
                escalation_probability,
                priority,
                confidence,
                status,
                intent_source,
                sentiment_source,
                resolution_source,
                escalation_source,
                created_at
            FROM conversations
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]


def clear_records():
    init_db()
    with get_connection() as conn:
        conn.execute("DELETE FROM conversations")
        conn.commit()
