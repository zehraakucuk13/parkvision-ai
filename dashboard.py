from pathlib import Path
from datetime import datetime, timedelta
import random
import time

import cv2
import pandas as pd
import streamlit as st

from detector import ParkingDetector
from parking_sessions import ParkingSessionManager
from vehicle_tracker import ParkingVehicleTracker


BASE_DIR = Path(__file__).resolve().parent
DISPLAY_WIDTH = 1280
PANEL_UPDATE_EVERY_FRAMES = 10
ROI = (260, 419, 1200, 1040)
END_HOLD_SECONDS = 4
DEFAULT_VIDEO_PATH = BASE_DIR / "samples" / "parking_1920_1080.mp4"
DEFAULT_MASK_PATH = BASE_DIR / "mask_1920_1080.png"
DEFAULT_SPEED = 8


def get_session_manager() -> ParkingSessionManager:
    if "parking_session_manager" not in st.session_state:
        st.session_state.parking_session_manager = ParkingSessionManager()
    return st.session_state.parking_session_manager


def main():
    st.set_page_config(page_title="ParkVision AI", layout="wide")
    inject_theme()
    st.title("ParkVision AI")
    st.session_state.setdefault("demo_finished", False)
    st.session_state.setdefault("demo_user_ready", False)
    st.session_state.setdefault("screen", "entry")
    if "default_entry_time" not in st.session_state:
        st.session_state.default_entry_time = datetime.now() - timedelta(minutes=random.randint(35, 125))
    session_manager = get_session_manager()

    if st.session_state.screen == "payment":
        render_payment_screen(session_manager)
        return

    if not st.session_state.demo_user_ready:
        render_entry_gate(session_manager)
        return

    session_manager.set_demo_user(
        st.session_state.demo_vehicle_id,
        st.session_state.demo_customer_name,
        st.session_state.demo_started_at,
    )
    video_path = str(DEFAULT_VIDEO_PATH)
    mask_path = str(DEFAULT_MASK_PATH)
    speed = DEFAULT_SPEED

    with st.sidebar:
        run = st.toggle("Start live demo", value=False)

    video_col, info_col = st.columns([2.2, 1])
    frame_slot = video_col.empty()
    metric_slot = info_col.empty()
    vehicle_slot = st.empty()

    if not run:
        st.session_state.demo_finished = False
        st.info("Start the demo to assign temporary IDs to parked vehicles and track exiting vehicles with the same ID.")
        render_parking_product(session_manager)
        return

    if st.session_state.demo_finished:
        st.info("Video ended.")
        render_parking_product(session_manager)
        return

    detector = ParkingDetector(mask_path, step=1, roi=ROI)
    vehicle_tracker = ParkingVehicleTracker()
    spot_boxes = dict(zip(detector.spot_ids, detector.spots))
    frame_index = 0
    last_annotated = None
    last_statuses = detector.status_dict

    def render_side_panel(statuses):
        total = len(statuses)
        empty = sum(statuses.values())
        occupied = total - empty
        entries = len(vehicle_tracker.entry_vehicle_ids)
        exits = len(vehicle_tracker.exiting_tracks) + len(vehicle_tracker.completed_vehicle_ids)
        session_manager.sync(
            vehicle_tracker.parked_vehicle_ids,
            vehicle_tracker.entry_vehicle_ids,
            vehicle_tracker.completed_vehicle_ids,
        )

        with metric_slot.container():
            cols = st.columns(2)
            cols[0].metric("Available", empty)
            cols[1].metric("Occupied", occupied)
            cols[0].metric("Entries", entries)
            cols[1].metric("Exits", exits)

        vehicle_slot.dataframe(pd.DataFrame(vehicle_tracker.rows()), use_container_width=True, hide_index=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Video could not be opened: {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = 1 / (fps * speed)

    while run:
        ok, frame = cap.read()
        if not ok:
            vehicle_tracker.complete_exiting_tracks()
            st.session_state.demo_finished = True
            if last_annotated is not None:
                frame_slot.image(last_annotated, channels="RGB", use_column_width=True)
            render_side_panel(last_statuses)
            st.info("Video ended.")
            time.sleep(END_HOLD_SECONDS)
            break

        annotated, statuses = detector.process(frame)
        last_statuses = statuses
        vehicle_tracker.update(statuses, spot_boxes, annotated, detector.diff_by_spot_id)
        annotated = detector.draw_spots(annotated)
        annotated = vehicle_tracker.draw(annotated, spot_boxes)
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        if annotated.shape[1] > DISPLAY_WIDTH:
            display_height = int(annotated.shape[0] * DISPLAY_WIDTH / annotated.shape[1])
            annotated = cv2.resize(annotated, (DISPLAY_WIDTH, display_height), interpolation=cv2.INTER_AREA)
        last_annotated = annotated
        frame_slot.image(annotated, channels="RGB", use_column_width=True)

        if frame_index % PANEL_UPDATE_EVERY_FRAMES == 0:
            render_side_panel(statuses)

        for _ in range(speed - 1):
            ok, _ = cap.read()
            if not ok:
                vehicle_tracker.complete_exiting_tracks()
                st.session_state.demo_finished = True
                if last_annotated is not None:
                    frame_slot.image(last_annotated, channels="RGB", use_column_width=True)
                render_side_panel(last_statuses)
                st.info("Video ended.")
                time.sleep(END_HOLD_SECONDS)
                run = False
                break

        frame_index += 1
        time.sleep(frame_delay)

    cap.release()
    render_parking_product(session_manager)


def render_entry_gate(session_manager: ParkingSessionManager):
    st.subheader("User Check-In")
    st.caption("Get your vehicle ID from the QR/ticket screen, confirm your entry time, and continue to the system.")

    default_started_at = st.session_state.default_entry_time
    vehicle_id = st.selectbox("Vehicle ID", ["V-0077"], index=0)
    customer_name = st.text_input("Name / phone", value="Demo User")
    entry_time = st.time_input("Parking entry time", value=default_started_at.time().replace(second=0, microsecond=0))
    started_at = datetime.combine(datetime.now().date(), entry_time)
    if started_at > datetime.now():
        started_at -= timedelta(days=1)

    qr_col, info_col = st.columns(2)
    with qr_col:
        st.markdown("**QR / Entry Ticket**")
        st.code(f"PARKVISION://check-in?vehicle={vehicle_id}", language="text")
    with info_col:
        st.metric("Assigned vehicle ID", vehicle_id)
        st.write("This ID will be matched with your session on the payment screen.")

    if st.button("Enter system", type="primary"):
        st.session_state.demo_vehicle_id = vehicle_id
        st.session_state.demo_customer_name = customer_name
        st.session_state.demo_started_at = started_at
        st.session_state.demo_user_ready = True
        st.session_state.screen = "main"
        session_manager.set_demo_user(vehicle_id, customer_name, started_at)
        st.rerun()


def render_parking_product(session_manager: ParkingSessionManager):
    st.divider()
    st.subheader("Session and Payment System")

    demo_session = session_manager.demo_session()
    if demo_session is not None and demo_session.ended_at is not None and not demo_session.paid:
        st.warning("You have exited. Continue to the payment page.")
        if st.button("Continue to payment", type="primary"):
            st.session_state.screen = "payment"
            st.rerun()

    st.markdown("**Parking Sessions**")
    rows = session_manager.rows()
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No sessions yet.")


def inject_theme():
    st.markdown(
        """
        <style>
        :root {
            --pv-black: #0b0b0d;
            --pv-white: #ffffff;
            --pv-soft: #f4f4f5;
            --pv-line: #d7d7db;
            --pv-muted: #62626a;
            --pv-blue: #2563ff;
        }
        .stApp {
            background: #050506;
            color: var(--pv-white);
        }
        [data-testid="stSidebar"] {
            background: #0b0b0d;
            border-right: 1px solid #26262a;
        }
        h1, h2, h3 {
            color: var(--pv-white);
            letter-spacing: 0;
        }
        h1 {
            font-weight: 800;
        }
        [data-testid="stMetric"] {
            background: #101013;
            border: 1px solid #2f2f35;
            border-radius: 8px;
            padding: 16px 18px;
            box-shadow: 0 1px 0 rgba(255, 255, 255, 0.04);
        }
        [data-testid="stMetricLabel"] {
            color: #b9b9c0;
        }
        [data-testid="stMetricValue"] {
            color: var(--pv-white);
            font-weight: 700;
        }
        a, a:visited {
            color: var(--pv-blue);
        }
        .stButton > button {
            background: var(--pv-blue);
            color: var(--pv-white);
            border: 1px solid var(--pv-blue);
            border-radius: 8px;
            padding: 0.55rem 1rem;
        }
        .stButton > button:hover {
            background: #174cff;
            color: var(--pv-white);
            border-color: #174cff;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid #2f2f35;
            border-radius: 8px;
            overflow: hidden;
        }
        .stAlert {
            border-radius: 8px;
            border: 1px solid #2f2f35;
        }
        [data-testid="stAlert"] {
            background: #101013;
            color: var(--pv-white);
        }
        [data-testid="stAlert"] * {
            color: var(--pv-white);
        }
        [data-testid="stAlert"] svg {
            color: var(--pv-blue);
            fill: var(--pv-blue);
        }
        [data-testid="stToggle"] [role="switch"][aria-checked="true"] {
            background: var(--pv-blue);
        }
        label, p, span, div {
            color: inherit;
        }
        input, textarea, select, [data-baseweb="select"] > div {
            background: #101013 !important;
            color: var(--pv-white) !important;
            border-color: #2f2f35 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_payment_screen(session_manager: ParkingSessionManager):
    st.subheader("Payment Page")
    demo_session = session_manager.demo_session()

    if demo_session is None:
        st.info("No session is linked to you yet.")
        if st.button("Back to check-in"):
            st.session_state.screen = "entry"
            st.session_state.demo_user_ready = False
            st.rerun()
        return

    tariff_col, payment_col = st.columns([1, 1])
    with tariff_col:
        st.markdown("**Pricing Tariff**")
        st.dataframe(pd.DataFrame(session_manager.tariff_rows()), use_container_width=True, hide_index=True)

    with payment_col:
        st.markdown("**Session Details**")
        st.write(
            {
                "Session ID": demo_session.session_id,
                "Vehicle ID": demo_session.vehicle_id,
                "Entry Time": demo_session.started_at.strftime("%H:%M"),
                "Exit Time": demo_session.ended_at.strftime("%H:%M") if demo_session.ended_at else "-",
                "Duration (min)": round(session_manager.duration_minutes_for(demo_session), 1),
            }
        )
        st.metric("Amount due", f"{session_manager.fee_for(demo_session):.0f} TL")

        if demo_session.paid:
            st.success("Payment completed. Have a safe trip.")
        elif demo_session.ended_at is None:
            st.info("The vehicle has not exited yet. Payment is collected after exit.")
        elif st.button("Complete payment", type="primary"):
            session_manager.mark_paid(demo_session.session_id)
            st.success("Payment completed. Have a safe trip.")

    st.markdown("**Session Record**")
    st.dataframe(pd.DataFrame([session_manager._row(demo_session)]), use_container_width=True, hide_index=True)

    if st.button("Back to overview"):
        st.session_state.screen = "main"
        st.rerun()


if __name__ == "__main__":
    main()
