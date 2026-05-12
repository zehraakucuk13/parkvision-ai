from pathlib import Path
import sqlite3


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "parkvision.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS parking_slots (
                spot_id TEXT PRIMARY KEY,
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS parking_sessions (
                session_id TEXT PRIMARY KEY,
                vehicle_id TEXT NOT NULL,
                spot_id TEXT NOT NULL,
                customer_name TEXT DEFAULT '',
                source TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                duration_min REAL DEFAULT 0,
                fee REAL DEFAULT 0,
                payment_status TEXT DEFAULT 'active',
                paid_at TEXT
            )
            """
        )
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(parking_sessions)").fetchall()
        }
        if "payment_method_label" not in columns:
            conn.execute("ALTER TABLE parking_sessions ADD COLUMN payment_method_label TEXT DEFAULT ''")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS customer_payment_methods (
                customer_name TEXT PRIMARY KEY,
                cardholder_name TEXT NOT NULL,
                card_brand TEXT NOT NULL,
                card_last4 TEXT NOT NULL,
                expiry_month INTEGER NOT NULL,
                expiry_year INTEGER NOT NULL,
                display_name TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def upsert_slots(spot_boxes: dict[str, list[int]]) -> None:
    with get_connection() as conn:
        for spot_id, box in spot_boxes.items():
            x, y, w, h = box
            conn.execute(
                """
                INSERT INTO parking_slots (spot_id, x, y, width, height)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(spot_id) DO UPDATE SET
                    x = excluded.x,
                    y = excluded.y,
                    width = excluded.width,
                    height = excluded.height
                """,
                (spot_id, x, y, w, h),
            )
        conn.commit()
