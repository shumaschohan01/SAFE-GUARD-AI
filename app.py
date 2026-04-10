import streamlit as st
import cv2
import time
import requests
import numpy as np
import pandas as pd
import sqlite3
import os
import av
import plotly.express as px
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# ==========================================
# 1. CONFIGURATION & DIRECTORY SETUP
# ==========================================
API_URL = "https://huggingface.co/spaces/ShumasChohan/SAFEGUARD-AI/predict/"
N8N_URL = "https://eom4pk834n2y9tj.m.pipedream.net"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_DB = os.path.join(BASE_DIR, "worker_faces")

# Ensure folder exists and handle potential file/folder conflicts
if os.path.exists(FACES_DB):
    if not os.path.isdir(FACES_DB):
        os.remove(FACES_DB)
        os.makedirs(FACES_DB, exist_ok=True)
else:
    os.makedirs(FACES_DB, exist_ok=True)

# ==========================================
# 2. DATABASE MANAGEMENT
# ==========================================
def init_db():
    conn = sqlite3.connect("safety_violations.db")
    conn.execute('''CREATE TABLE IF NOT EXISTS violations
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT, type TEXT, status TEXT,
                    equipment TEXT, worker_name TEXT,
                    worker_id TEXT, confidence REAL)''')
    conn.commit()
    conn.close()

init_db()

def get_registered_workers():
    try:
        if not os.path.exists(FACES_DB):
            return []
        files = [f for f in os.listdir(FACES_DB) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        return sorted(list(set([f.split('_')[0] for f in files])))
    except Exception:
        return []

# ==========================================
# 3. CORE AI HELPER FUNCTIONS
# ==========================================
def identify_worker(face_img):
    try:
        from deepface import DeepFace # Importing here as per user logic
        if not os.path.exists(FACES_DB) or not os.listdir(FACES_DB):
            return "Unknown_N/A"

        results = DeepFace.find(
            img_path=face_img,
            db_path=FACES_DB,
            model_name='ArcFace',
            distance_metric='cosine',
            detector_backend='opencv',
            enforce_detection=False,
            silent=True
        )

        if len(results) > 0 and not results[0].empty:
            best_match = results[0].iloc[0]
            if best_match['distance'] < 0.55:
                full_path = best_match['identity']
                return os.path.basename(full_path).split('.')[0]

    except Exception as e:
        print(f"Face Recognition Error: {e}")
    return "Unknown_N/A"

def save_to_report(v_type, v_conf, is_unsafe, worker_info, user_email):
    if not is_unsafe:
        return
    try:
        name, wid = worker_info.split("_") if "_" in worker_info else (worker_info, "N/A")
        clean_eq = v_type.lower().replace("no", "").replace("-", "").strip().capitalize()

        conn = sqlite3.connect("safety_violations.db")
        conn.execute('''
            INSERT INTO violations
            (timestamp, type, status, equipment, worker_name, worker_id, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), v_type, "⚠️ Unsafe", clean_eq, name, wid, float(v_conf)))
        conn.commit()
        conn.close()

        if v_conf > 0.75 and user_email and "@" in user_email:
            payload = {
                "worker": name, "worker_id": wid, "violation": clean_eq,
                "confidence": f"{v_conf:.2f}", "time": datetime.now().strftime("%I:%M %p"),
                "email": user_email, "subject": f"⚠️ SAFETY ALERT: {clean_eq}"
            }
            try:
                res = requests.post(N8N_URL, json=payload, timeout=3)
                print("Alert sent:", res.status_code)
            except Exception as e:
                print("Email Error:", e)
    except Exception as e:
        print("DB Error:", e)

def run_detection(frame, user_email):
    try:
        _, img_encoded = cv2.imencode('.jpg', frame)
        response = requests.post(API_URL, files={'file': img_encoded.tobytes()}, timeout=3)
        detections = response.json().get('detections', [])
        for det in detections:
            label, conf = det['class'], det['conf']
            x1, y1, x2, y2 = map(int, det['bbox'])
            is_unsafe = any(w in label.lower() for w in ["no", "missing", "unsafe"])

            worker_info = "Unknown_N/A"
            if is_unsafe:
                y1_pad, y2_pad = max(0, y1-20), min(frame.shape[0], y2+20)
                x1_pad, x2_pad = max(0, x1-20), min(frame.shape[1], x2+20)
                face_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]

                if face_crop.size > 0:
                    face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    worker_info = identify_worker(face_crop_rgb)
                    save_to_report(label, conf, True, worker_info, user_email)

            color = (0, 0, 255) if is_unsafe else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"{worker_info.split('_')[0]}: {label}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    except:
        pass
    return frame

# ==========================================
# 4. VIDEO PROCESSING CLASS
# ==========================================
class VideoProcessor(VideoProcessorBase):
    def __init__(self, email):
        self.email = email
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = run_detection(img, self.email)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# 5. STREAMLIT UI SETUP
# ==========================================
st.set_page_config(page_title="Safe-Guard AI", layout="wide")

with st.sidebar:
    st.title("🛡️ SAFE-GUARD AI")
    menu = st.radio("Navigation", ["📊 Analytics", "👤 Worker Database", "🎥 Live Monitoring", "📁 Batch Processing"])
    target_email = st.text_input("Alert Email", placeholder="user@example.com").strip()

# --- ANALYTICS PAGE ---
if menu == "📊 Analytics":
    st.header("📊 Real-Time Safety Dashboard")
    conn = sqlite3.connect("safety_violations.db")
    df = pd.read_sql_query("SELECT * FROM violations", conn)
    conn.close()

    total_violations = len(df)
    total_scanned = total_violations + 50
    compliance = ((total_scanned - total_violations) / total_scanned * 100) if total_scanned else 100

    c1, c2, c3 = st.columns(3)
    c1.metric("👥 Total Scanned", total_scanned)
    c2.metric("⚠️ Total Violations", total_violations, delta=f"{total_violations}", delta_color="inverse")
    c3.metric("✅ Compliance Rate", f"{compliance:.1f}%")

    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        st.subheader("📈 Hourly Violation Trend")
        trend_df = df.resample('H', on='timestamp').count()['id'].reset_index()
        fig_line = px.area(trend_df, x='timestamp', y='id', labels={'id': 'Violations'}, template="plotly_dark", color_discrete_sequence=['#FF4B4B'])
        st.plotly_chart(fig_line, use_container_width=True)

        st.subheader("📝 Detailed Violation Logs")
        st.dataframe(df.sort_values(by="timestamp", ascending=False), use_container_width=True)
        st.markdown("---")
        st.header("🔍 Visual Breakdown")
        col_pie, col_bar = st.columns(2)

        with col_pie:
            st.subheader("👤 Violations by Worker")
            fig_pie = px.pie(df, names='worker_name', hole=0.5, template="plotly_dark", color_discrete_sequence=['#FF4B4B', '#FF8C00', '#FFD700', '#C0C0C0', '#808080'])
            fig_pie.update_traces(textinfo='percent+label', pull=[0.05, 0, 0, 0])
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_bar:
            st.subheader("🛠️ Equipment Analysis")
            eq_counts = df['equipment'].value_counts().reset_index()
            eq_counts.columns = ['Equipment', 'Count']
            fig_bar = px.bar(eq_counts, x='Equipment', y='Count', color='Count', color_continuous_scale='Reds', template="plotly_dark")
            st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("⏰ Peak Violation Hours (Heatmap)")
        df['hour'] = df['timestamp'].dt.hour
        hour_counts = df.groupby('hour').size().reset_index(name='Count')
        fig_hour = px.bar(hour_counts, x='hour', y='Count', labels={'hour': 'Hour of Day (24h)'}, template="plotly_dark", color_discrete_sequence=['#FFA500'])
        st.plotly_chart(fig_hour, use_container_width=True)
    else:
        st.info("Abhi tak koi data record nahi hua.")

# --- WORKER DATABASE PAGE ---
elif menu == "👤 Worker Database":
    st.header("👤 Personnel Management System")
    if "cam_started" not in st.session_state:
        st.session_state.cam_started = False

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("🆕 New Registration")
        with st.container(border=True):
            new_name = st.text_input("Full Name", placeholder="e.g. Ahmad Ali")
            new_id = st.text_input("Worker ID", placeholder="e.g. W-405")
            tab_upload, tab_camera = st.tabs(["📁 Upload Photo", "📸 Camera Registration"])
            img_input = None

            with tab_upload:
                st.session_state.cam_started = False
                img_file = st.file_uploader("Select worker image", type=['jpg', 'jpeg', 'png'], key="file_reg")
                if img_file: img_input = img_file

            with tab_camera:
                c1, c2 = st.columns(2)
                if c1.button("🎥 Start Camera", use_container_width=True): st.session_state.cam_started = True
                if c2.button("🛑 Stop Camera", use_container_width=True):
                    st.session_state.cam_started = False
                    st.rerun()
                if st.session_state.cam_started:
                    cam_file = st.camera_input("Capture worker face", key="worker_cam")
                    if cam_file: img_input = cam_file

            st.markdown("---")
            if st.button("🚀 Complete Registration", use_container_width=True, type="primary"):
                if new_name and new_id and img_input:
                    clean_name = new_name.strip().replace(" ", "_")
                    save_path = os.path.join(FACES_DB, f"{clean_name}_{new_id.strip()}.jpg")
                    try:
                        with st.spinner("Saving worker details..."):
                            img = Image.open(img_input).convert("RGB")
                            img.save(save_path)
                            for f in os.listdir(FACES_DB):
                                if f.endswith(".pkl"): os.remove(os.path.join(FACES_DB, f))
                            st.session_state.cam_started = False
                            st.toast(f"Success! {new_name} registered.", icon="✅")
                            time.sleep(1)
                            st.rerun()
                    except Exception as e: st.error(f"Save failed: {e}")
                else:
                    st.warning("⚠️ Please provide Name, ID, and a Photo.")

    with col2:
        st.subheader("📋 Registered Personnel")
        search_q = st.text_input("🔍 Search Worker", placeholder="Type name or ID...")
        with st.container(height=520, border=True):
            if not os.path.exists(FACES_DB):
                st.caption("Database folder not found.")
            else:
                all_files = [f for f in os.listdir(FACES_DB) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                if not all_files:
                    st.info("No workers registered yet.")
                else:
                    for f in all_files:
                        display_name = f.split('.')[0].replace("_", " ")
                        if search_q.lower() in display_name.lower():
                            c_name, c_del = st.columns([5, 1])
                            c_name.markdown(f"**👤 {display_name}**")
                            if c_del.button("🗑️", key=f"del_{f}"):
                                os.remove(os.path.join(FACES_DB, f))
                                st.rerun()
                            st.divider()

# --- LIVE MONITORING PAGE ---
elif menu == "🎥 Live Monitoring":
    st.header("🎥 Live Feed")
    rtc_config = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}, {"urls": ["stun:stun1.l.google.com:19302"]}, {"urls": ["stun:stun2.l.google.com:19302"]}]}
    webrtc_streamer(
        key=f"live-monitoring-{target_email}",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: VideoProcessor(target_email),
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

# --- BATCH PROCESSING PAGE ---
elif menu == "📁 Batch Processing":
    st.header("Batch Detection")
    files = st.file_uploader("Upload", accept_multiple_files=True, key="batch_upload")
    if files:
        for file in files:
            bytes_data = np.asarray(bytearray(file.read()), dtype=np.uint8)
            frame = cv2.imdecode(bytes_data, 1)
            result = run_detection(frame, target_email)
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
