from dataclasses import dataclass
from datetime import datetime, timedelta
import random


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


class ParkingSessionManager:
    def __init__(
        self,
        hourly_rate: float = 60.0,
        free_minutes: int = 10,
        demo_minutes_per_second: float = 1.0,
    ):
        self.hourly_rate = hourly_rate
        self.free_minutes = free_minutes
        self.demo_minutes_per_second = demo_minutes_per_second
        self._session_counter = 0
        self.sessions_by_vehicle: dict[str, ParkingSession] = {}
        self.demo_vehicle_id = "V-0077"
        self.demo_customer_name = ""
        self.demo_started_at: datetime | None = None

    def set_demo_user(self, vehicle_id: str, customer_name: str, started_at: datetime) -> None:
        self.demo_vehicle_id = vehicle_id
        self.demo_customer_name = customer_name.strip()
        self.demo_started_at = started_at
        session = self.sessions_by_vehicle.get(vehicle_id)
        if session is not None:
            session.customer_name = self.demo_customer_name
            session.started_at = started_at
            session.source = "Checked in"

    def sync(
        self,
        parked_vehicle_ids: dict[str, str],
        entry_vehicle_ids: set[str],
        completed_vehicle_ids: list[str],
    ) -> None:
        now = datetime.now()

        for spot_id, vehicle_id in parked_vehicle_ids.items():
            if vehicle_id not in self.sessions_by_vehicle:
                self.sessions_by_vehicle[vehicle_id] = self._new_session(
                    vehicle_id=vehicle_id,
                    spot_id=spot_id,
                    source=self._source_for(vehicle_id, entry_vehicle_ids),
                    started_at=self._started_at_for(vehicle_id, now),
                )

        for vehicle_id in completed_vehicle_ids:
            session = self.sessions_by_vehicle.get(vehicle_id)
            if session is not None and session.ended_at is None:
                session.ended_at = now

    def claim(self, session_id: str, customer_name: str) -> None:
        session = self.get_by_session_id(session_id)
        if session is not None:
            session.customer_name = customer_name.strip()

    def mark_paid(self, session_id: str) -> None:
        session = self.get_by_session_id(session_id)
        if session is not None:
            session.paid = True

    def get_by_session_id(self, session_id: str) -> ParkingSession | None:
        for session in self.sessions_by_vehicle.values():
            if session.session_id == session_id:
                return session
        return None

    def rows(self) -> list[dict]:
        return [self._row(session) for session in self.sessions_by_vehicle.values()]

    def demo_session(self) -> ParkingSession | None:
        return self.sessions_by_vehicle.get(self.demo_vehicle_id)

    def active_rows(self) -> list[dict]:
        return [
            self._row(session)
            for session in self.sessions_by_vehicle.values()
            if session.ended_at is None
        ]

    def payable_rows(self) -> list[dict]:
        return [
            self._row(session)
            for session in self.sessions_by_vehicle.values()
            if session.ended_at is not None and not session.paid
        ]

    def total_revenue(self) -> float:
        return sum(self.fee_for(session) for session in self.sessions_by_vehicle.values() if session.paid)

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
            {"Duration": f"First {self.free_minutes} minutes", "Fee": "Free"},
            {"Duration": "10-60 minutes", "Fee": "50 TL"},
            {"Duration": "1-2 hours", "Fee": "80 TL"},
            {"Duration": "2-4 hours", "Fee": "120 TL"},
            {"Duration": "Over 4 hours", "Fee": "200 TL"},
        ]

    def duration_minutes_for(self, session: ParkingSession) -> float:
        end_time = session.ended_at or datetime.now()
        real_seconds = max(0.0, (end_time - session.started_at).total_seconds())
        return (real_seconds / 60) * self.demo_minutes_per_second

    def _new_session(self, vehicle_id: str, spot_id: str, source: str, started_at: datetime) -> ParkingSession:
        self._session_counter += 1
        return ParkingSession(
            session_id=f"P-{self._session_counter:05d}",
            vehicle_id=vehicle_id,
            spot_id=spot_id,
            source=source,
            started_at=started_at,
            customer_name=self.demo_customer_name if vehicle_id == self.demo_vehicle_id else "",
        )

    def _source_for(self, vehicle_id: str, entry_vehicle_ids: set[str]) -> str:
        if vehicle_id == self.demo_vehicle_id:
            return "Checked in"
        return "Checked in" if vehicle_id in entry_vehicle_ids else "Parked"

    def _started_at_for(self, vehicle_id: str, now: datetime) -> datetime:
        if vehicle_id == self.demo_vehicle_id and self.demo_started_at is not None:
            return self.demo_started_at

        seed = sum(ord(char) for char in vehicle_id)
        parked_minutes = random.Random(seed).randint(25, 180)
        real_seconds = parked_minutes * 60 / self.demo_minutes_per_second
        return now - timedelta(seconds=real_seconds)

    def _row(self, session: ParkingSession) -> dict:
        if session.paid:
            payment_status = "Paid"
        elif session.ended_at is not None:
            payment_status = "Payment pending"
        else:
            payment_status = "Active"

        return {
            "Session ID": session.session_id,
            "Vehicle ID": session.vehicle_id,
            "Spot": session.spot_id,
            "Status": session.source if session.ended_at is None else "Exited",
            "Customer": session.customer_name or "-",
            "Entry Time": session.started_at.strftime("%H:%M"),
            "Exit Time": session.ended_at.strftime("%H:%M") if session.ended_at else "-",
            "Duration (min)": round(self.duration_minutes_for(session), 1),
            "Fee (TL)": self.fee_for(session),
            "Payment": payment_status,
        }
