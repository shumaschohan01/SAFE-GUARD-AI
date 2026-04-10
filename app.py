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
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# ==========================================
# 1. CONFIGURATION & DIRECTORIES
# ==========================================
API_URL = "https://huggingface.co/spaces/ShumasChohan/SAFEGUARD-AI/predict/"
N8N_URL = "https://eom4pk834n2y9tj.m.pipedream.net"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_DB = os.path.join(BASE_DIR, "worker_faces")

# Ensure worker database directory exists
if os.path.exists(FACES_DB):
    if not os.path.isdir(FACES_DB):
        os.remove(FACES_DB)
        os.makedirs(FACES_DB, exist_ok=True)
else:
    os.makedirs(FACES_DB, exist_ok=True)

# ==========================================
# 2. DATABASE LOGIC
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
# 3. AI CORE FUNCTIONS (Detection & Recognition)
# ==========================================
def identify_worker(face_img):
    try:
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
        ''', (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            v_type,
            "⚠️ Unsafe",
            clean_eq,
            name,
            wid,
            float(v_conf)
        ))
        conn.commit()
        conn.close()

        # Webhook Alert
        if v_conf > 0.75 and user_email and "@" in user_email:
            payload = {
                "worker": name,
                "worker_id": wid,
                "violation": clean_eq,
                "confidence": f"{v_conf:.2f}",
                "time": datetime.now().strftime("%I:%M %p"),
                "email": user_email,
                "subject": f"⚠️ SAFETY ALERT: {clean_eq}"
            }
            try:
                requests.post(N8N_URL, json=payload, timeout=3)
            except Exception: pass

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
                y1_p, y2_p = max(0, y1-20), min(frame.shape[0], y2+20)
                x1_p, x2_p = max(0, x1-20), min(frame.shape[1], x2+20)
                face_crop = frame[y1_p:y2_p, x1_p:x2_p]

                if face_crop.size > 0:
                    face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    worker_info = identify_worker(face_crop_rgb)
                
                save_to_report(label, conf, True, worker_info, user_email)
            
            color = (0, 0, 255) if is_unsafe else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"{worker_info.split('_')[0]}: {label}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    except: pass
    return frame

# ==========================================
# 4. VIDEO STREAMING CLASS
# ==========================================
class VideoProcessor(VideoProcessorBase):
    def __init__(self, email):
        self.email = email
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = run_detection(img, self.email)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# 5. USER INTERFACE (Main App)
# ==========================================
st.set_page_config(page_title="Safe-Guard AI", layout="wide", page_icon="🛡️")

with st.sidebar:
    st.title("🛡️ SAFE-GUARD AI")
    menu = st.radio("Navigation", ["📊 Analytics", "👤 Worker Database", "🎥 Live Monitoring", "📁 Batch Processing"])
    target_email = st.text_input("Alert Email", placeholder="user@example.com").strip()

# --- TAB: ANALYTICS ---
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
        fig_line = px.area(trend_df, x='timestamp', y='id', template="plotly_dark", color_discrete_sequence=['#FF4B4B'])
        st.plotly_chart(fig_line, use_container_width=True)

        st.subheader("📝 Detailed Violation Logs")
        st.dataframe(df.sort_values(by="timestamp", ascending=False), use_container_width=True)

        st.header("🔍 Visual Breakdown")
        col_pie, col_bar = st.columns(2)
        with col_pie:
            fig_pie = px.pie(df, names='worker_name', hole=0.5, template="plotly_dark")
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_bar:
            eq_counts = df['equipment'].value_counts().reset_index()
            eq_counts.columns = ['Equipment', 'Count']
            fig_bar = px.bar(eq_counts, x='Equipment', y='Count', color='Count', color_continuous_scale='Reds', template="plotly_dark")
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No data recorded yet.")

# --- TAB: WORKER DATABASE ---
elif menu == "👤 Worker Database":
    st.header("👤 Personnel Management System")
    if "cam_started" not in st.session_state:
        st.session_state.cam_started = False

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("🆕 New Registration")
        with st.container(border=True):
            new_name = st.text_input("Full Name")
            new_id = st.text_input("Worker ID")
            tab_upload, tab_camera = st.tabs(["📁 Upload Photo", "📸 Camera Registration"])
            img_input = None

            with tab_upload:
                img_file = st.file_uploader("Select image", type=['jpg', 'png'], key="file_reg")
                if img_file: img_input = img_file
            with tab_camera:
                if st.button("🎥 Start Camera"): st.session_state.cam_started = True
                if st.button("🛑 Stop Camera"): 
                    st.session_state.cam_started = False
                    st.rerun()
                if st.session_state.cam_started:
                    img_input = st.camera_input("Capture face")

            if st.button("🚀 Complete Registration", type="primary"):
                if new_name and new_id and img_input:
                    filename = f"{new_name.strip().replace(' ', '_')}_{new_id.strip()}.jpg"
                    save_path = os.path.join(FACES_DB, filename)
                    Image.open(img_input).convert("RGB").save(save_path)
                    # Clear DeepFace cache
                    for f in os.listdir(FACES_DB):
                        if f.endswith(".pkl"): os.remove(os.path.join(FACES_DB, f))
                    st.success(f"Registered {new_name}")
                    st.rerun()

    with col2:
        st.subheader("📋 Registered Personnel")
        if os.path.exists(FACES_DB):
            files = [f for f in os.listdir(FACES_DB) if f.lower().endswith(('.jpg', '.png'))]
            for f in files:
                c_name, c_del = st.columns([5, 1])
                c_name.write(f"👤 {f.split('.')[0]}")
                if c_del.button("🗑️", key=f"del_{f}"):
                    os.remove(os.path.join(FACES_DB, f))
                    st.rerun()

# --- TAB: LIVE MONITORING ---
elif menu == "🎥 Live Monitoring":
    st.header("🎥 Live Feed")
    rtc_config = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    webrtc_streamer(
        key="live-monitor",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: VideoProcessor(target_email),
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

# --- TAB: BATCH PROCESSING ---
elif menu == "📁 Batch Processing":
    st.header("Batch Detection")
    files = st.file_uploader("Upload Images", accept_multiple_files=True)
    if files:
        for file in files:
            img = np.asarray(bytearray(file.read()), dtype=np.uint8)
            frame = cv2.imdecode(img, 1)
            result = run_detection(frame, target_email)
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
