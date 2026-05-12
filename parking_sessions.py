from dataclasses import dataclass
from datetime import datetime, timedelta
import random

from database import get_connection, init_db


@dataclass
class ParkingSession:
    session_id: str
    vehicle_id: str
    spot_id: str
    started_at: datetime
    source: str
    customer_name: str = ""
    paid: bool = False
    ended_at: datetime | None = None
    fee: float = 0.0
    payment_method_label: str = ""


class ParkingSessionManager:
    def __init__(
        self,
        hourly_rate: float = 60.0,
        free_minutes: int = 10,
        demo_minutes_per_second: float = 1.0,
    ):
        init_db()
        self.hourly_rate = hourly_rate
        self.free_minutes = free_minutes
        self.demo_minutes_per_second = demo_minutes_per_second
        self.demo_vehicle_id = "V-0077"
        self.demo_customer_name = ""
        self.demo_started_at: datetime | None = None

    def set_demo_user(self, vehicle_id: str, customer_name: str, started_at: datetime) -> None:
        self.demo_vehicle_id = vehicle_id
        self.demo_customer_name = customer_name.strip()
        self.demo_started_at = started_at

        with get_connection() as conn:
            conn.execute(
                """
                UPDATE parking_sessions
                SET customer_name = ?, started_at = ?, source = 'Giriş yaptı'
                WHERE vehicle_id = ? AND ended_at IS NULL
                """,
                (self.demo_customer_name, started_at.isoformat(), vehicle_id),
            )
            conn.commit()
        self.ensure_demo_session()

    def next_vehicle_id(self) -> str:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT vehicle_id
                FROM parking_sessions
                WHERE vehicle_id LIKE 'V-%'
                """
            ).fetchall()

        highest_id = 76
        for row in rows:
            vehicle_id = row["vehicle_id"]
            suffix = vehicle_id.removeprefix("V-")
            if suffix.isdigit():
                highest_id = max(highest_id, int(suffix))

        return f"V-{highest_id + 1:04d}"

    def start_demo_session(self, vehicle_id: str, customer_name: str, started_at: datetime) -> ParkingSession:
        self.demo_vehicle_id = vehicle_id
        self.demo_customer_name = customer_name.strip()
        self.demo_started_at = started_at

        session = self._new_session(
            vehicle_id=vehicle_id,
            spot_id="ENTRY",
            source="Giriş yaptı",
            started_at=started_at,
        )
        self._insert_session(session)
        return session

    def ensure_demo_session(self) -> ParkingSession:
        session = self._active_session_by_vehicle(self.demo_vehicle_id)
        if session is not None:
            return session

        session = self._new_session(
            vehicle_id=self.demo_vehicle_id,
            spot_id="ENTRY",
            source="Giriş yaptı",
            started_at=self.demo_started_at or datetime.now(),
        )
        self._insert_session(session)
        return session

    def close_demo_session(self) -> ParkingSession:
        session = self.ensure_demo_session()
        now = datetime.now()
        session.ended_at = now
        fee = self.fee_for(session)
        duration = self.duration_minutes_for(session)

        with get_connection() as conn:
            conn.execute(
                """
                UPDATE parking_sessions
                SET ended_at = ?,
                    duration_min = ?,
                    fee = ?,
                    payment_status = 'payment_pending'
                WHERE session_id = ?
                """,
                (now.isoformat(), duration, fee, session.session_id),
            )
            conn.commit()

        return self.get_by_session_id(session.session_id)

    def close_session(self, session_id: str) -> ParkingSession | None:
        session = self.get_by_session_id(session_id)
        if session is None:
            return None

        now = datetime.now()
        session.ended_at = now
        fee = self.fee_for(session)
        duration = self.duration_minutes_for(session)

        with get_connection() as conn:
            conn.execute(
                """
                UPDATE parking_sessions
                SET ended_at = ?,
                    duration_min = ?,
                    fee = ?,
                    payment_status = 'payment_pending'
                WHERE session_id = ?
                """,
                (now.isoformat(), duration, fee, session_id),
            )
            conn.commit()

        return self.get_by_session_id(session_id)

    def sync(
        self,
        parked_vehicle_ids: dict[str, str],
        entry_vehicle_ids: set[str],
        completed_vehicle_ids: list[str],
    ) -> None:
        now = datetime.now()

        for spot_id, vehicle_id in parked_vehicle_ids.items():
            active_session = self._active_session_by_vehicle(vehicle_id)
            if active_session is None:
                session = self._new_session(
                    vehicle_id=vehicle_id,
                    spot_id=spot_id,
                    source=self._source_for(vehicle_id, entry_vehicle_ids),
                    started_at=self._started_at_for(vehicle_id, now),
                )
                self._insert_session(session)
            elif active_session.spot_id == "ENTRY":
                with get_connection() as conn:
                    conn.execute(
                        "UPDATE parking_sessions SET spot_id = ? WHERE session_id = ?",
                        (spot_id, active_session.session_id),
                    )
                    conn.commit()

        for vehicle_id in completed_vehicle_ids:
            session = self._active_session_by_vehicle(vehicle_id)
            if session is not None:
                session.ended_at = now
                fee = self.fee_for(session)
                duration = self.duration_minutes_for(session)
                with get_connection() as conn:
                    conn.execute(
                        """
                        UPDATE parking_sessions
                        SET ended_at = ?,
                            duration_min = ?,
                            fee = ?,
                            payment_status = 'payment_pending'
                        WHERE session_id = ?
                        """,
                        (now.isoformat(), duration, fee, session.session_id),
                    )
                    conn.commit()

    def claim(self, session_id: str, customer_name: str) -> None:
        with get_connection() as conn:
            conn.execute(
                "UPDATE parking_sessions SET customer_name = ? WHERE session_id = ?",
                (customer_name.strip(), session_id),
            )
            conn.commit()

    def mark_paid(self, session_id: str) -> None:
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE parking_sessions
                SET payment_status = 'paid', paid_at = ?
                WHERE session_id = ?
                """,
                (datetime.now().isoformat(), session_id),
            )
            conn.commit()

    def mark_paid_with_method(self, session_id: str, payment_method_label: str = "") -> None:
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE parking_sessions
                SET payment_status = 'paid',
                    paid_at = ?,
                    payment_method_label = ?
                WHERE session_id = ?
                """,
                (datetime.now().isoformat(), payment_method_label, session_id),
            )
            conn.commit()

    def save_payment_method(
        self,
        customer_name: str,
        cardholder_name: str,
        card_brand: str,
        card_last4: str,
        expiry_month: int,
        expiry_year: int,
    ) -> str:
        display_name = f"{card_brand} •••• {card_last4} - {expiry_month:02d}/{expiry_year}"
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO customer_payment_methods (
                    customer_name,
                    cardholder_name,
                    card_brand,
                    card_last4,
                    expiry_month,
                    expiry_year,
                    display_name,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(customer_name) DO UPDATE SET
                    cardholder_name = excluded.cardholder_name,
                    card_brand = excluded.card_brand,
                    card_last4 = excluded.card_last4,
                    expiry_month = excluded.expiry_month,
                    expiry_year = excluded.expiry_year,
                    display_name = excluded.display_name,
                    updated_at = excluded.updated_at
                """,
                (
                    customer_name.strip(),
                    cardholder_name.strip(),
                    card_brand,
                    card_last4,
                    expiry_month,
                    expiry_year,
                    display_name,
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
        return display_name

    def payment_method_for_customer(self, customer_name: str) -> dict | None:
        clean_name = customer_name.strip()
        if not clean_name:
            return None

        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM customer_payment_methods
                WHERE lower(customer_name) = lower(?)
                """,
                (clean_name,),
            ).fetchone()
        return dict(row) if row else None

    def payment_method_rows(self) -> list[dict]:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM customer_payment_methods
                ORDER BY updated_at DESC
                """
            ).fetchall()

        return [
            {
                "Müşteri": row["customer_name"],
                "Kart Sahibi": row["cardholder_name"],
                "Kart": row["display_name"],
                "Son 4 Hane": row["card_last4"],
                "Son Kullanma": f"{row['expiry_month']:02d}/{row['expiry_year']}",
                "Güncellendi": datetime.fromisoformat(row["updated_at"]).strftime("%d.%m.%Y %H:%M"),
            }
            for row in rows
        ]

    def get_by_session_id(self, session_id: str) -> ParkingSession | None:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM parking_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return self._session_from_row(row) if row else None

    def rows(self) -> list[dict]:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM parking_sessions ORDER BY started_at DESC"
            ).fetchall()
        return [self._row(self._session_from_row(row)) for row in rows]

    def demo_session(self) -> ParkingSession | None:
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM parking_sessions
                WHERE vehicle_id = ?
                ORDER BY
                    CASE WHEN ended_at IS NULL THEN 0 ELSE 1 END,
                    started_at DESC
                LIMIT 1
                """,
                (self.demo_vehicle_id,),
            ).fetchone()
        return self._session_from_row(row) if row else None

    def active_rows(self) -> list[dict]:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM parking_sessions
                WHERE ended_at IS NULL
                ORDER BY started_at DESC
                """
            ).fetchall()
        return [self._row(self._session_from_row(row)) for row in rows]

    def payable_rows(self) -> list[dict]:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM parking_sessions
                WHERE ended_at IS NOT NULL
                  AND payment_status != 'paid'
                ORDER BY ended_at DESC
                """
            ).fetchall()
        return [self._row(self._session_from_row(row)) for row in rows]

    def total_revenue(self) -> float:
        with get_connection() as conn:
            value = conn.execute(
                """
                SELECT COALESCE(SUM(fee), 0)
                FROM parking_sessions
                WHERE payment_status = 'paid'
                """
            ).fetchone()[0]
        return float(value)

    def report_rows(self) -> list[dict]:
        with get_connection() as conn:
            total_sessions = conn.execute("SELECT COUNT(*) FROM parking_sessions").fetchone()[0]
            active_sessions = conn.execute(
                "SELECT COUNT(*) FROM parking_sessions WHERE ended_at IS NULL"
            ).fetchone()[0]
            pending_payments = conn.execute(
                """
                SELECT COUNT(*)
                FROM parking_sessions
                WHERE payment_status = 'payment_pending'
                """
            ).fetchone()[0]
            revenue = conn.execute(
                """
                SELECT COALESCE(SUM(fee), 0)
                FROM parking_sessions
                WHERE payment_status = 'paid'
                """
            ).fetchone()[0]

        return [
            {"Metrik": "Toplam oturum", "Değer": total_sessions},
            {"Metrik": "Aktif oturum", "Değer": active_sessions},
            {"Metrik": "Ödeme bekleyen", "Değer": pending_payments},
            {"Metrik": "Toplam hasılat", "Değer": f"{float(revenue):.0f} TL"},
        ]

    def fee_for(self, session: ParkingSession) -> float:
        parked_minutes = self.duration_minutes_for(session)
        if parked_minutes <= self.free_minutes:
            return 0.0
        if parked_minutes <= 60:
            return 50.0
        if parked_minutes <= 120:
            return 80.0
        if parked_minutes <= 240:
            return 120.0
        return 200.0

    def tariff_rows(self) -> list[dict]:
        return [
            {"Süre": f"İlk {self.free_minutes} dakika", "Ücret": "Ücretsiz"},
            {"Süre": "10-60 dakika", "Ücret": "50 TL"},
            {"Süre": "1-2 saat", "Ücret": "80 TL"},
            {"Süre": "2-4 saat", "Ücret": "120 TL"},
            {"Süre": "4 saat üzeri", "Ücret": "200 TL"},
        ]

    def duration_minutes_for(self, session: ParkingSession) -> float:
        end_time = session.ended_at or datetime.now()
        real_seconds = max(0.0, (end_time - session.started_at).total_seconds())
        return (real_seconds / 60) * self.demo_minutes_per_second

    def _new_session(self, vehicle_id: str, spot_id: str, source: str, started_at: datetime) -> ParkingSession:
        return ParkingSession(
            session_id=self._next_session_id(),
            vehicle_id=vehicle_id,
            spot_id=spot_id,
            source=source,
            started_at=started_at,
            customer_name=self.demo_customer_name if vehicle_id == self.demo_vehicle_id else "",
        )

    def _source_for(self, vehicle_id: str, entry_vehicle_ids: set[str]) -> str:
        if vehicle_id == self.demo_vehicle_id:
            return "Giriş yaptı"
        return "Giriş yaptı" if vehicle_id in entry_vehicle_ids else "Park halinde"

    def _started_at_for(self, vehicle_id: str, now: datetime) -> datetime:
        if vehicle_id == self.demo_vehicle_id and self.demo_started_at is not None:
            return self.demo_started_at

        seed = sum(ord(char) for char in vehicle_id)
        parked_minutes = random.Random(seed).randint(25, 180)
        real_seconds = parked_minutes * 60 / self.demo_minutes_per_second
        return now - timedelta(seconds=real_seconds)

    def _insert_session(self, session: ParkingSession) -> None:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO parking_sessions (
                    session_id,
                    vehicle_id,
                    spot_id,
                    customer_name,
                    source,
                    started_at,
                    ended_at,
                    duration_min,
                    fee,
                    payment_status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    session.vehicle_id,
                    session.spot_id,
                    session.customer_name,
                    session.source,
                    session.started_at.isoformat(),
                    session.ended_at.isoformat() if session.ended_at else None,
                    self.duration_minutes_for(session),
                    self.fee_for(session),
                    "active",
                ),
            )
            conn.commit()

    def _active_session_by_vehicle(self, vehicle_id: str) -> ParkingSession | None:
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM parking_sessions
                WHERE vehicle_id = ? AND ended_at IS NULL
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (vehicle_id,),
            ).fetchone()
        return self._session_from_row(row) if row else None

    def _next_session_id(self) -> str:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT session_id
                FROM parking_sessions
                WHERE session_id LIKE 'P-%'
                """
            ).fetchall()

        highest_id = 0
        for row in rows:
            suffix = row["session_id"].removeprefix("P-")
            if suffix.isdigit():
                highest_id = max(highest_id, int(suffix))

        return f"P-{highest_id + 1:05d}"

    @staticmethod
    def _session_from_row(row) -> ParkingSession:
        return ParkingSession(
            session_id=row["session_id"],
            vehicle_id=row["vehicle_id"],
            spot_id=row["spot_id"],
            started_at=datetime.fromisoformat(row["started_at"]),
            source=row["source"],
            customer_name=row["customer_name"] or "",
            paid=row["payment_status"] == "paid",
            ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
            fee=float(row["fee"] or 0),
            payment_method_label=row["payment_method_label"] if "payment_method_label" in row.keys() else "",
        )

    def _row(self, session: ParkingSession) -> dict:
        if session.paid:
            payment_status = "Ödendi"
        elif session.ended_at is not None:
            payment_status = "Ödeme bekliyor"
        else:
            payment_status = "Aktif"

        source = {
            "Checked in": "Giriş yaptı",
            "Parked": "Park halinde",
        }.get(session.source, session.source)
        spot = "Giriş" if session.spot_id == "ENTRY" else session.spot_id

        return {
            "Oturum ID": session.session_id,
            "Araç ID": session.vehicle_id,
            "Park Yeri": spot,
            "Durum": source if session.ended_at is None else "Çıkış yaptı",
            "Müşteri": session.customer_name or "-",
            "Giriş Saati": session.started_at.strftime("%H:%M"),
            "Çıkış Saati": session.ended_at.strftime("%H:%M") if session.ended_at else "-",
            "Süre (dk)": round(self.duration_minutes_for(session), 1),
            "Ücret (TL)": self.fee_for(session),
            "Ödeme": payment_status,
            "Ödeme Yöntemi": session.payment_method_label or "-",
        }
