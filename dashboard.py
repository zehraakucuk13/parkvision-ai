from pathlib import Path
from datetime import datetime, timedelta
import random
import re
import time

import cv2
import pandas as pd
import streamlit as st

from database import reset_demo_data, upsert_slots
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


def normalize_card_number(card_number: str) -> str:
    return re.sub(r"\D", "", card_number or "")


def is_valid_card_number(card_number: str) -> bool:
    digits = normalize_card_number(card_number)
    if not 13 <= len(digits) <= 19:
        return False

    checksum = 0
    reversed_digits = digits[::-1]
    for index, digit in enumerate(reversed_digits):
        value = int(digit)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        checksum += value

    return checksum % 10 == 0


def is_valid_expiry(expiry: str) -> bool:
    parsed = parse_expiry(expiry)
    if parsed is None:
        return False

    month, year = parsed
    now = datetime.now()
    return (year, month) >= (now.year, now.month)


def parse_expiry(expiry: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"(0[1-9]|1[0-2])\s*/\s*(\d{2}|\d{4})", (expiry or "").strip())
    if not match:
        return None

    month = int(match.group(1))
    year_text = match.group(2)
    year = int(year_text) if len(year_text) == 4 else 2000 + int(year_text)
    return month, year


def is_valid_cvv(cvv: str) -> bool:
    return bool(re.fullmatch(r"\d{3,4}", (cvv or "").strip()))


def card_brand_for(card_number: str) -> str:
    digits = normalize_card_number(card_number)
    if digits.startswith("4"):
        return "Visa"
    if digits[:2] in {"51", "52", "53", "54", "55"} or 2221 <= int(digits[:4] or 0) <= 2720:
        return "Mastercard"
    if digits.startswith(("34", "37")):
        return "American Express"
    return "Kart"


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
    st.session_state.setdefault("role", None)
    st.session_state.setdefault("screen", "role_login")
    if "demo_db_reset_done" not in st.session_state:
        reset_demo_data()
        st.session_state.demo_db_reset_done = True
    if "default_entry_time" not in st.session_state:
        st.session_state.default_entry_time = datetime.now() - timedelta(minutes=random.randint(35, 125))

    session_manager = get_session_manager()

    if st.session_state.role is None:
        render_role_login()
        return

    with st.sidebar:
        role_label = "Yönetici" if st.session_state.role == "admin" else "Sürücü"
        st.caption(f"Giriş: {role_label}")
        if st.button("Çıkış yap", use_container_width=True):
            st.session_state.role = None
            st.session_state.screen = "role_login"
            st.session_state.demo_user_ready = False
            st.session_state.pop("demo_session_id", None)
            st.rerun()

    if st.session_state.role == "driver":
        if "demo_vehicle_id" in st.session_state:
            session_manager.demo_vehicle_id = st.session_state.demo_vehicle_id
            session_manager.demo_customer_name = st.session_state.get("demo_customer_name", "")
            session_manager.demo_started_at = st.session_state.get("demo_started_at")

        if not st.session_state.demo_user_ready:
            render_entry_gate(session_manager)
            return

        render_driver_session(session_manager)
        return

    if st.session_state.role == "payment":
        if st.session_state.screen == "payment":
            render_payment_screen(session_manager)
            return

        render_payment_lookup(session_manager)
        return

    render_parking_product(session_manager)


def render_tracking_video(session_manager: ParkingSessionManager):
    run = st.toggle("Giriş videosunu çalıştır", value=False)

    video_col, info_col = st.columns([2.2, 1])
    frame_slot = video_col.empty()
    metric_slot = info_col.empty()
    vehicle_slot = st.empty()

    if not run:
        st.info("Video çalıştırıldığında araç giriş-çıkışları algılanır ve kayıtlar güncellenir.")
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
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = 1 / (fps * DEFAULT_SPEED)

    while run:
        ok, frame = cap.read()
        if not ok:
            vehicle_tracker.complete_exiting_tracks()
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


def render_role_login():
    st.subheader("Giriş")
    driver_col, payment_col, admin_col = st.columns(3)

    with driver_col:
        st.markdown("**Sürücü Girişi**")
        if st.button("Sürücü olarak giriş yap", type="primary", use_container_width=True):
            st.session_state.role = "driver"
            st.session_state.screen = "entry"
            st.session_state.demo_user_ready = False
            st.session_state.pop("demo_session_id", None)
            st.rerun()

    with payment_col:
        st.markdown("**Ödeme Yapma**")
        if st.button("Ödeme ekranına geç", type="primary", use_container_width=True):
            st.session_state.role = "payment"
            st.session_state.screen = "payment_lookup"
            st.session_state.demo_user_ready = False
            st.session_state.pop("demo_session_id", None)
            st.rerun()

    with admin_col:
        st.markdown("**Yönetici Girişi**")
        with st.form("admin_login_form"):
            password = st.text_input("Yönetici parolası", type="password")
            submitted = st.form_submit_button("Yönetici olarak giriş yap", use_container_width=True)

        if submitted:
            if password == "123":
                st.session_state.role = "admin"
                st.session_state.screen = "admin"
                st.session_state.demo_user_ready = False
                st.rerun()
            else:
                st.error("Yönetici parolası hatalı.")


def render_driver_session(session_manager: ParkingSessionManager):
    st.subheader("Sürücü Oturumu")
    demo_session = session_manager.get_by_session_id(st.session_state.get("demo_session_id", ""))

    if demo_session is None:
        st.info("Aktif oturum bulunamadı.")
        if st.button("Giriş ekranına dön"):
            st.session_state.demo_user_ready = False
            st.session_state.screen = "entry"
            st.rerun()
        return

    duration = round(session_manager.duration_minutes_for(demo_session), 1)
    amount = session_manager.fee_for(demo_session)
    exit_time = demo_session.ended_at.strftime("%H:%M") if demo_session.ended_at else "-"
    status_text = "Ödendi" if demo_session.paid else "Ödeme bekliyor" if demo_session.ended_at else "Park devam ediyor"

    cols = st.columns(4)
    cols[0].metric("Araç ID", demo_session.vehicle_id)
    cols[1].metric("Durum", status_text)
    cols[2].metric("Süre", f"{duration} dk")
    cols[3].metric("Ücret", f"{amount:.0f} TL")

    detail_rows = [
        {"Alan": "Oturum ID", "Değer": demo_session.session_id},
        {"Alan": "Ad / telefon", "Değer": demo_session.customer_name or "-"},
        {"Alan": "Giriş Saati", "Değer": demo_session.started_at.strftime("%H:%M")},
        {"Alan": "Çıkış Saati", "Değer": exit_time},
    ]
    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

    if demo_session.ended_at is None:
        st.info("Araç çıkışı görüntü işleme tarafından algılandığında ödeme kaydı oluşur.")
    elif not demo_session.paid:
        st.warning("Bu araç için ödeme bekleyen kayıt var. Ana ekrandaki ödeme yapma kısmından araç ID ile ödeme alınır.")
    else:
        st.success("Bu oturumun ödemesi tamamlandı.")

    if st.button("Yeni kullanıcı girişi", use_container_width=True):
        st.session_state.demo_user_ready = False
        st.session_state.screen = "entry"
        st.session_state.pop("demo_session_id", None)
        st.rerun()


def render_payment_lookup(session_manager: ParkingSessionManager):
    st.subheader("Ödeme Yapma")
    vehicle_id = st.text_input("Ödenecek araç ID", value=st.session_state.get("payment_vehicle_id", ""))

    if st.button("Ödeme kaydını bul", type="primary", use_container_width=True):
        session = session_manager.payable_session_by_vehicle(vehicle_id)
        if session is None:
            st.error("Bu araç ID için ödeme bekleyen çıkış kaydı bulunamadı.")
        else:
            st.session_state.payment_vehicle_id = vehicle_id.strip()
            st.session_state.demo_vehicle_id = session.vehicle_id
            st.session_state.demo_customer_name = session.customer_name
            st.session_state.demo_started_at = session.started_at
            st.session_state.demo_session_id = session.session_id
            st.session_state.screen = "payment"
            st.rerun()

    payable_rows = session_manager.payable_rows()
    if payable_rows:
        st.markdown("**Ödeme Bekleyen Araçlar**")
        st.dataframe(pd.DataFrame(payable_rows), use_container_width=True, hide_index=True)


def render_entry_gate(session_manager: ParkingSessionManager):
    st.subheader("Kullanıcı Girişi")
    st.caption("Araç ID'nizi giriş fişi/QR ekranından alın, giriş saatinizi onaylayın ve sisteme devam edin.")
    render_tracking_video(session_manager)

    default_started_at = st.session_state.default_entry_time
    vehicle_id = st.text_input("Araç ID", value=st.session_state.get("demo_vehicle_id", "V-0077"))
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
        demo_session = session_manager.start_demo_session(vehicle_id, customer_name, started_at)
        st.session_state.demo_session_id = demo_session.session_id
        st.rerun()


def render_parking_product(session_manager: ParkingSessionManager):
    st.divider()
    is_admin = st.session_state.get("role") == "admin"
    st.subheader("Yönetim Paneli" if is_admin else "Oturum ve Ödeme Sistemi")

    demo_session = session_manager.demo_session()
    if is_admin:
        st.markdown("**Otopark ve Ödeme Kayıtları**")
    else:
        st.markdown("**Ödeme Paneli**")

    if demo_session is None and not is_admin:
        st.info("Ödeme oluşturmak için önce kullanıcı girişi yapılmalıdır.")
    elif demo_session is not None and not is_admin:
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
                if st.button("Yeni kullanıcı girişi", use_container_width=True):
                    st.session_state.demo_user_ready = False
                    st.session_state.screen = "entry"
                    st.session_state.pop("demo_session_id", None)
                    st.rerun()
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

    st.markdown("**Kullanıcı Ödeme Bilgileri**")
    payment_method_rows = session_manager.payment_method_rows()
    if payment_method_rows:
        st.dataframe(pd.DataFrame(payment_method_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Henüz kayıtlı ödeme yöntemi yok.")


def render_payment_screen(session_manager: ParkingSessionManager):
    st.subheader("Ödeme Sayfası")
    demo_session = session_manager.get_by_session_id(st.session_state.get("demo_session_id", ""))

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
            if st.button("Yeni ödeme ara", use_container_width=True):
                st.session_state.demo_user_ready = False
                st.session_state.screen = "payment_lookup"
                st.session_state.pop("demo_session_id", None)
                st.rerun()
        elif demo_session.ended_at is None:
            st.info("Araç henüz çıkış yapmadı. Ödeme çıkıştan sonra alınır.")
        else:
            saved_method = session_manager.payment_method_for_customer(demo_session.customer_name)
            if saved_method:
                st.info(f"Kayıtlı ödeme yöntemi bulundu: {saved_method['display_name']}")
                with st.form("saved_payment_form"):
                    saved_cvv = st.text_input("Kayıtlı kart CVV", type="password", placeholder="•••", max_chars=4)
                    saved_submitted = st.form_submit_button(
                        "Kayıtlı ödeme yöntemiyle öde",
                        type="primary",
                        use_container_width=True,
                    )

                if saved_submitted:
                    if not is_valid_cvv(saved_cvv):
                        st.error("Geçerli bir kredi kartı giriniz.")
                    else:
                        session_manager.mark_paid_with_method(demo_session.session_id, saved_method["display_name"])
                        st.success("Ödeme kayıtlı yöntemle tamamlandı. İyi yolculuklar.")
                        st.rerun()

            with st.form("payment_card_form"):
                st.markdown("**Kart Bilgileri**")
                cardholder_name = st.text_input("Kart üzerindeki ad soyad", value=demo_session.customer_name)
                card_number = st.text_input("Kart numarası", placeholder="1234 5678 9012 3456")
                expiry_col, cvv_col = st.columns(2)
                expiry = expiry_col.text_input("Son kullanma tarihi", placeholder="AA/YY veya AA/YYYY")
                cvv = cvv_col.text_input("CVV", type="password", placeholder="123", max_chars=4)
                save_payment_method = st.checkbox("Ödeme yolunu kaydet")
                submitted = st.form_submit_button("Ödemeyi tamamla", type="primary", use_container_width=True)

            if submitted:
                parsed_expiry = parse_expiry(expiry)
                if (
                    not cardholder_name.strip()
                    or not is_valid_card_number(card_number)
                    or parsed_expiry is None
                    or not is_valid_expiry(expiry)
                    or not is_valid_cvv(cvv)
                ):
                    st.error("Geçerli bir kredi kartı giriniz.")
                else:
                    payment_method_label = ""
                    if save_payment_method:
                        expiry_month, expiry_year = parsed_expiry
                        digits = normalize_card_number(card_number)
                        payment_method_label = session_manager.save_payment_method(
                            demo_session.customer_name or cardholder_name,
                            cardholder_name,
                            card_brand_for(card_number),
                            digits[-4:],
                            expiry_month,
                            expiry_year,
                        )
                    session_manager.mark_paid_with_method(demo_session.session_id, payment_method_label)
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
