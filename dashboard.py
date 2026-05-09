from pathlib import Path
from datetime import datetime, timedelta
import random
import time

import cv2
import pandas as pd
import streamlit as st

from database import upsert_slots
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

    with st.sidebar:
        run = st.toggle("Canlı demoyu başlat", value=False)

    video_col, info_col = st.columns([2.2, 1])
    frame_slot = video_col.empty()
    metric_slot = info_col.empty()
    vehicle_slot = st.empty()

    if not run:
        st.session_state.demo_finished = False
        st.info("Park doluluğunu algılamak, oturumları kaydetmek ve ödeme kayıtlarını güncellemek için demoyu başlatın.")
        render_parking_product(session_manager)
        return

    if st.session_state.demo_finished:
        st.info("Video bitti.")
        render_parking_product(session_manager)
        return

    detector = ParkingDetector(str(DEFAULT_MASK_PATH), step=1, roi=ROI)
    vehicle_tracker = ParkingVehicleTracker()
    spot_boxes = dict(zip(detector.spot_ids, detector.spots))
    upsert_slots(spot_boxes)

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
            cols[0].metric("Boş", empty)
            cols[1].metric("Dolu", occupied)
            cols[0].metric("Giriş", entries)
            cols[1].metric("Çıkış", exits)

        vehicle_rows = vehicle_tracker.rows()
        if vehicle_rows:
            vehicle_slot.dataframe(pd.DataFrame(vehicle_rows), use_container_width=True, hide_index=True)
        else:
            vehicle_slot.info("Henüz araç hareketi yok.")

    cap = cv2.VideoCapture(str(DEFAULT_VIDEO_PATH))
    if not cap.isOpened():
        st.error(f"Video açılamadı: {DEFAULT_VIDEO_PATH}")
        render_parking_product(session_manager)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = 1 / (fps * DEFAULT_SPEED)

    while run:
        ok, frame = cap.read()
        if not ok:
            vehicle_tracker.complete_exiting_tracks()
            st.session_state.demo_finished = True
            if last_annotated is not None:
                frame_slot.image(last_annotated, channels="RGB", width="stretch")
            render_side_panel(last_statuses)
            st.info("Video bitti.")
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
        frame_slot.image(annotated, channels="RGB", width="stretch")

        if frame_index % PANEL_UPDATE_EVERY_FRAMES == 0:
            render_side_panel(statuses)

        for _ in range(DEFAULT_SPEED - 1):
            ok, _ = cap.read()
            if not ok:
                vehicle_tracker.complete_exiting_tracks()
                st.session_state.demo_finished = True
                if last_annotated is not None:
                    frame_slot.image(last_annotated, channels="RGB", width="stretch")
                render_side_panel(last_statuses)
                st.info("Video bitti.")
                time.sleep(END_HOLD_SECONDS)
                run = False
                break

        frame_index += 1
        time.sleep(frame_delay)

    cap.release()
    render_parking_product(session_manager)


def render_entry_gate(session_manager: ParkingSessionManager):
    st.subheader("Kullanıcı Girişi")
    st.caption("Araç ID'nizi giriş fişi/QR ekranından alın, giriş saatinizi onaylayın ve sisteme devam edin.")

    default_started_at = st.session_state.default_entry_time
    vehicle_id = st.selectbox("Araç ID", ["V-0077"], index=0)
    customer_name = st.text_input("Ad / telefon", value="Demo Kullanıcı")
    entry_time = st.time_input("Park giriş saati", value=default_started_at.time().replace(second=0, microsecond=0))
    started_at = datetime.combine(datetime.now().date(), entry_time)
    if started_at > datetime.now():
        started_at -= timedelta(days=1)

    qr_col, info_col = st.columns(2)
    with qr_col:
        st.markdown("**QR / Giriş Fişi**")
        st.code(f"PARKVISION://giris?arac={vehicle_id}", language="text")
    with info_col:
        st.metric("Atanan araç ID", vehicle_id)
        st.write("Bu ID ödeme ekranında oturumunuzla eşleştirilecek.")

    if st.button("Sisteme gir", type="primary"):
        st.session_state.demo_vehicle_id = vehicle_id
        st.session_state.demo_customer_name = customer_name
        st.session_state.demo_started_at = started_at
        st.session_state.demo_user_ready = True
        st.session_state.screen = "main"
        session_manager.set_demo_user(vehicle_id, customer_name, started_at)
        st.rerun()


def render_parking_product(session_manager: ParkingSessionManager):
    st.divider()
    st.subheader("Oturum ve Ödeme Sistemi")

    demo_session = session_manager.demo_session()
    st.markdown("**Ödeme Paneli**")
    if demo_session is None:
        st.info("Ödeme oluşturmak için önce kullanıcı girişi yapılmalıdır.")
    else:
        duration = round(session_manager.duration_minutes_for(demo_session), 1)
        amount = session_manager.fee_for(demo_session)
        exit_time = demo_session.ended_at.strftime("%H:%M") if demo_session.ended_at else "-"
        status_text = "Ödendi" if demo_session.paid else "Ödeme bekliyor" if demo_session.ended_at else "Park devam ediyor"

        st.markdown(
            f"""
            <div class="payment-panel">
                <div>
                    <div class="payment-eyebrow">Güncel oturum</div>
                    <div class="payment-title">{demo_session.vehicle_id} için ödeme durumu</div>
                    <div class="payment-subtitle">Oturum: {demo_session.session_id} · Park yeri: {demo_session.spot_id}</div>
                </div>
                <div class="payment-amount">{amount:.0f} TL</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cols = st.columns(4)
        cols[0].metric("Durum", status_text)
        cols[1].metric("Giriş", demo_session.started_at.strftime("%H:%M"))
        cols[2].metric("Çıkış", exit_time)
        cols[3].metric("Süre", f"{duration} dk")

        action_col, note_col = st.columns([1, 1.4])
        with action_col:
            if demo_session.ended_at is None:
                if st.button("Araç çıkışını simüle et ve ücreti görüntüle", type="primary", use_container_width=True):
                    session_manager.close_demo_session()
                    st.session_state.screen = "payment"
                    st.rerun()
            elif not demo_session.paid:
                if st.button("Ödeme ekranına git", type="primary", use_container_width=True):
                    st.session_state.screen = "payment"
                    st.rerun()
            else:
                st.success("Bu oturumun ödemesi tamamlandı.")
        with note_col:
            if demo_session.ended_at is None:
                st.info("Araç çıkış yaptığında süre ve ücret kesinleşir.")
            elif not demo_session.paid:
                st.warning("Çıkış tamamlandı. Ödeme alınması bekleniyor.")
            else:
                st.info("Ödeme kaydı veritabanında tamamlandı olarak tutuluyor.")

    st.markdown("**Park Oturumları**")
    rows = session_manager.rows()
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Henüz oturum yok.")

    report_rows = session_manager.report_rows()
    st.markdown("**Raporlar**")
    st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)


def render_payment_screen(session_manager: ParkingSessionManager):
    st.subheader("Ödeme Sayfası")
    demo_session = session_manager.demo_session()

    if demo_session is None:
        st.info("Size bağlı bir oturum bulunamadı.")
        if st.button("Giriş ekranına dön"):
            st.session_state.screen = "entry"
            st.session_state.demo_user_ready = False
            st.rerun()
        return

    duration = round(session_manager.duration_minutes_for(demo_session), 1)
    amount = session_manager.fee_for(demo_session)
    exit_time = demo_session.ended_at.strftime("%H:%M") if demo_session.ended_at else "-"
    status_text = "Ödendi" if demo_session.paid else "Ödeme bekliyor" if demo_session.ended_at else "Park devam ediyor"

    st.markdown(
        f"""
        <div class="payment-page-hero">
            <div>
                <div class="payment-eyebrow">Ödenecek tutar</div>
                <div class="payment-page-amount">{amount:.0f} TL</div>
                <div class="payment-subtitle">{demo_session.vehicle_id} · {demo_session.session_id} · {status_text}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    detail_col, tariff_col = st.columns([1.2, 0.8])
    with detail_col:
        st.markdown("**Oturum Bilgileri**")
        detail_rows = [
            {"Alan": "Oturum ID", "Değer": demo_session.session_id},
            {"Alan": "Araç ID", "Değer": demo_session.vehicle_id},
            {"Alan": "Giriş Saati", "Değer": demo_session.started_at.strftime("%H:%M")},
            {"Alan": "Çıkış Saati", "Değer": exit_time},
            {"Alan": "Süre", "Değer": f"{duration} dk"},
            {"Alan": "Ödeme Durumu", "Değer": status_text},
        ]
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

        if demo_session.paid:
            st.success("Ödeme tamamlandı. İyi yolculuklar.")
        elif demo_session.ended_at is None:
            st.info("Araç henüz çıkış yapmadı. Ödeme çıkıştan sonra alınır.")
        elif st.button("Ödemeyi tamamla", type="primary", use_container_width=True):
            session_manager.mark_paid(demo_session.session_id)
            st.success("Ödeme tamamlandı. İyi yolculuklar.")
            st.rerun()

    with tariff_col:
        st.markdown("**Ücret Tarifesi**")
        st.dataframe(pd.DataFrame(session_manager.tariff_rows()), use_container_width=True, hide_index=True)

    st.markdown("**Oturum Kaydı**")
    st.dataframe(pd.DataFrame([session_manager._row(demo_session)]), use_container_width=True, hide_index=True)

    if st.button("Genel bakışa dön"):
        st.session_state.screen = "main"
        st.rerun()


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
        .payment-panel,
        .payment-page-hero {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 24px;
            background: #101013;
            border: 1px solid #2f2f35;
            border-left: 5px solid var(--pv-blue);
            border-radius: 8px;
            padding: 20px 22px;
            margin: 8px 0 16px 0;
        }
        .payment-page-hero {
            padding: 26px 28px;
            margin-bottom: 22px;
        }
        .payment-eyebrow {
            color: #b9b9c0;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 4px;
        }
        .payment-title {
            color: var(--pv-white);
            font-size: 1.35rem;
            font-weight: 750;
        }
        .payment-subtitle {
            color: #b9b9c0;
            font-size: 0.95rem;
            margin-top: 4px;
        }
        .payment-amount,
        .payment-page-amount {
            color: var(--pv-white);
            font-size: 2rem;
            font-weight: 800;
            white-space: nowrap;
        }
        .payment-page-amount {
            font-size: 3rem;
        }
        @media (max-width: 760px) {
            .payment-panel,
            .payment-page-hero {
                align-items: flex-start;
                flex-direction: column;
            }
            .payment-page-amount {
                font-size: 2.2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
