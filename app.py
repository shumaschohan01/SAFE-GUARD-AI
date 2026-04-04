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

# --- CONFIGURATION ---
API_URL = "https://shumaschohan-safeguard-ai.hf.space/predict/"
N8N_URL = "https://eom4pk834n2y9tj.m.pipedream.net"
# Safe folder handling
FACES_DB = "worker_faces"

if not os.path.exists(FACES_DB): os.makedirs(FACES_DB)

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect("safety_violations.db")
    conn.execute('''CREATE TABLE IF NOT EXISTS violations 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                     timestamp TEXT, type TEXT, status TEXT,
                     equipment TEXT, worker_name TEXT,
                     worker_id TEXT, confidence REAL)''')
    conn.commit(); conn.close()

init_db()

# --- HELPER FUNCTIONS ---
def identify_worker(face_img):
    try:
        if not os.listdir(FACES_DB): return "Unknown_N/A"
        results = DeepFace.find(img_path=face_img, db_path=FACES_DB, enforce_detection=False, silent=True)
        if len(results) > 0 and not results[0].empty:
            full_path = results[0].iloc[0]['identity']
            return os.path.basename(full_path).split('.')[0]
    except: pass
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

        # 🔴 CHANGED: Pipedream alert improved
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
                face_crop = frame[max(0,y1):y2, max(0,x1):x2]
                if face_crop.size > 0: worker_info = identify_worker(face_crop)
                save_to_report(label, conf, True, worker_info, user_email)
            
            color = (0, 0, 255) if is_unsafe else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"{worker_info.split('_')[0]}: {label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    except: pass
    return frame

class VideoProcessor(VideoProcessorBase):
    def __init__(self, email):
        self.email = email
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = run_detection(img, self.email)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI ---
st.set_page_config(page_title="Safe-Guard AI", layout="wide")

with st.sidebar:
    st.title("🛡️ SAFE-GUARD AI")
    menu = st.radio("Navigation", ["📊 Analytics", "👤 Worker Database", "🎥 Live Monitoring", "📁 Batch Processing"])
    target_email = st.text_input("Alert Email", placeholder="user@example.com").strip()

if menu == "📊 Analytics":
    st.header("📊 Real-Time Safety Dashboard")

    conn = sqlite3.connect("safety_violations.db")
    df = pd.read_sql_query("SELECT * FROM violations", conn)
    conn.close()

    # --- KPI METRICS ---
    total_violations = len(df)
    total_scanned = total_violations + 50 
    compliance = ((total_scanned - total_violations) / total_scanned * 100) if total_scanned else 100

    c1, c2, c3 = st.columns(3)
    # Colors for Metrics
    c1.metric("👥 Total Scanned", total_scanned)
    c2.metric("⚠️ Total Violations", total_violations, delta=f"{total_violations}", delta_color="inverse")
    c3.metric("✅ Compliance Rate", f"{compliance:.1f}%")

    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # --- ROW 1: MAIN TREND LINE ---
        st.subheader("📈 Hourly Violation Trend")
        trend_df = df.resample('H', on='timestamp').count()['id'].reset_index()
        fig_line = px.area(trend_df, x='timestamp', y='id', 
                           labels={'id': 'Violations'}, 
                           template="plotly_dark",
                           color_discrete_sequence=['#FF4B4B']) # Safety Red
        fig_line.update_layout(hovermode="x unified")
        st.plotly_chart(fig_line, use_container_width=True)

        # --- ROW 2: DETAILED TABLE ---
        st.subheader("📝 Detailed Violation Logs")
        st.dataframe(df.sort_values(by="timestamp", ascending=False), use_container_width=True)

        st.markdown("---")
        
        # --- ROW 3: PIE CHART & BAR CHART (The "Acha Look" Section) ---
        st.header("🔍 Visual Breakdown")
        col_pie, col_bar = st.columns(2)

        with col_pie:
            st.subheader("👤 Violations by Worker")
            # Custom Color Palette (Sunset/Safety Theme)
            custom_colors = ['#FF4B4B', '#FF8C00', '#FFD700', '#C0C0C0', '#808080']
            
            fig_pie = px.pie(df, names='worker_name', hole=0.5, 
                             template="plotly_dark",
                             color_discrete_sequence=custom_colors)
            
            fig_pie.update_traces(textinfo='percent+label', pull=[0.05, 0, 0, 0]) # Pehla slice thora bahar
            fig_pie.update_layout(showlegend=False) # Legend hata di taaki saaf dikhe
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_bar:
            st.subheader("🛠️ Equipment Analysis")
            eq_counts = df['equipment'].value_counts().reset_index()
            eq_counts.columns = ['Equipment', 'Count']
            
            # Gradient Bar Chart
            fig_bar = px.bar(eq_counts, x='Equipment', y='Count', 
                             color='Count',
                             color_continuous_scale='Reds', # Red gradient
                             template="plotly_dark")
            fig_bar.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        # --- NEW: TIME HEATMAP (Analysis Table ke neeche) ---
        st.subheader("⏰ Peak Violation Hours (Heatmap)")
        df['hour'] = df['timestamp'].dt.hour
        hour_counts = df.groupby('hour').size().reset_index(name='Count')
        
        fig_hour = px.bar(hour_counts, x='hour', y='Count', 
                          labels={'hour': 'Hour of Day (24h)'},
                          template="plotly_dark",
                          color_discrete_sequence=['#FFA500']) # Orange Theme
        st.plotly_chart(fig_hour, use_container_width=True)

    else:
        st.info("Abhi tak koi data record nahi hua.")
        
elif menu == "👤 Worker Database":
    st.header("👤 Worker Registration")
    col1, col2 = st.columns(2)
    with col1:
        method = st.radio("Method", ["Camera", "Upload"])
        name = st.text_input("Name")
        emp_id = st.text_input("ID")
        img_file = st.camera_input("Photo") if method == "Camera" else st.file_uploader("Photo", type=['jpg', 'png'])
        if st.button("Register") and img_file and name and emp_id:
            Image.open(img_file).convert('RGB').save(os.path.join(FACES_DB, f"{name}_{emp_id}.jpg"))
            st.success("Registered!")
            st.rerun()
    with col2:
        st.subheader("Personnel List")
        for f in os.listdir(FACES_DB):
            if "_" in f:
                c1, c2 = st.columns([4, 1])
                c1.write(f"✅ {f.split('.')[0]}")
                if c2.button("🗑️", key=f):
                    os.remove(os.path.join(FACES_DB, f))
                    st.rerun()

elif menu == "🎥 Live Monitoring":
    st.header("🎥 Live Feed")
    # Multiple STUN servers for better connectivity
    rtc_config = {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]}
    ]}

    webrtc_streamer(
        key=f"live-monitoring-{target_email}", # Dynamic key fixes conflicts
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: VideoProcessor(target_email),
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

elif menu == "📁 Batch Processing":
    st.header("Batch Detection")
    files = st.file_uploader("Upload", accept_multiple_files=True, key="batch_upload")
    if files:
        for file in files:
            bytes_data = np.asarray(bytearray(file.read()), dtype=np.uint8)
            frame = cv2.imdecode(bytes_data, 1)
            result = run_detection(frame, target_email)
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
