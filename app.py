import streamlit as st
from PIL import Image
import requests
import numpy as np
import pandas as pd
import sqlite3
import os
import time
import av
import cv2
import io
import matplotlib.pyplot as plt
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# --- CONFIGURATION ---
API_URL = "https://shumaschohan-safeguard-api.hf.space/predict/"
N8N_URL = "https://eom4pk834n2y9tj.m.pipedream.net"
FACES_DB = "worker_faces"

if not os.path.exists(FACES_DB): 
    os.makedirs(FACES_DB)

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

# --- DETECTION & ANALYTICS HELPER ---
def identify_worker(face_img):
    try:
        from deepface import DeepFace
        if not os.path.exists(FACES_DB) or not os.listdir(FACES_DB): return "Unknown_N/A"
        temp_path = "temp_face.jpg"
        cv2.imwrite(temp_path, face_img)
        results = DeepFace.find(img_path=temp_path, db_path=FACES_DB, enforce_detection=False, silent=True)
        if len(results) > 0 and not results[0].empty:
            full_path = results[0].iloc[0]['identity']
            return os.path.basename(full_path).split('.')[0]
    except: pass
    return "Unknown_N/A"

def run_detection(frame, user_email):
    try:
        _, img_encoded = cv2.imencode('.jpg', frame)
        response = requests.post(API_URL, files={'file': img_encoded.tobytes()}, timeout=5)
        if response.status_code == 200:
            detections = response.json().get('detections', [])
            for det in detections:
                label, conf = det['class'], det['conf']
                x1, y1, x2, y2 = map(int, det['bbox'])
                is_unsafe = any(w in label.lower() for w in ["no", "missing", "unsafe", "without"])
                
                worker_info = "Unknown_N/A"
                if is_unsafe:
                    face_crop = frame[max(0,y1):y2, max(0,x1):x2]
                    if face_crop.size > 0: worker_info = identify_worker(face_crop)
                    # Logic to save to DB (Calling save_to_report function)
                
                color = (0, 0, 255) if is_unsafe else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f"{label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    except: pass
    return frame

# --- UI SETUP ---
st.set_page_config(page_title="Safe-Guard AI", layout="wide")

with st.sidebar:
    st.title("🛡️ SAFE-GUARD AI")
    menu = st.radio("Navigation", ["📊 Analytics", "👤 Worker Database", "🎥 Live Monitoring"])
    target_email = st.text_input("Alert Email", placeholder="example@gmail.com")

# --- PAGES ---
if menu == "📊 Analytics":
    st.header("📊 Violation Insights & Visualizations")
    conn = sqlite3.connect("safety_violations.db")
    df = pd.read_sql_query("SELECT * FROM violations", conn)
    conn.close()

    if not df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Equipment-wise Violations")
            # Bar Chart
            equip_counts = df['equipment'].value_counts()
            st.bar_chart(equip_counts)
        
        with col2:
            st.subheader("Violation Distribution")
            # Pie Chart using Matplotlib
            fig, ax = plt.subplots()
            df['type'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
            st.pyplot(fig)

        st.subheader("Recent Activity Log")
        st.dataframe(df.sort_values(by='timestamp', ascending=False), use_container_width=True)
    else:
        st.info("No violation data available yet.")

elif menu == "👤 Worker Database":
    # (Puran wala registration code yahan rakhien...)
    st.info("Worker registration module active.")

elif menu == "🎥 Live Monitoring":
    st.header("Live AI Safety Guard")
    webrtc_streamer(
        key="cam", 
        mode=WebRtcMode.SENDRECV, 
        video_processor_factory=lambda: VideoProcessor(target_email) if 'VideoProcessor' in globals() else None,
        async_processing=False 
    )
