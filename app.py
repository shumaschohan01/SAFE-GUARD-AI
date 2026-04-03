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
from PIL import Image
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# --- CONFIGURATION ---
API_URL = "https://shumaschohan-safeguard-ai.hf.space/predict/"
N8N_URL = "https://eom4pk834n2y9tj.m.pipedream.net"
FACES_DB = "worker_faces"

# 🔴 FIX 1: Cooldown Tracker (Session State use karein taaki refresh par reset na ho)
if 'worker_cooldowns' not in st.session_state:
    st.session_state.worker_cooldowns = {}

if not os.path.exists(FACES_DB):
    os.makedirs(FACES_DB)

# --- DATABASE ---
def init_db():
    conn = sqlite3.connect("safety_violations.db")
    conn.execute('''CREATE TABLE IF NOT EXISTS violations 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, type TEXT, 
                     status TEXT, equipment TEXT, worker_name TEXT, worker_id TEXT, confidence REAL)''')
    conn.commit()
    conn.close()

init_db()

# --- HELPER FUNCTIONS ---
def identify_worker(face_crop):
    try:
        from deepface import DeepFace
        if not os.path.exists(FACES_DB) or not os.listdir(FACES_DB):
            return "Unknown_N/A"

        # 🔴 FIX 2: DeepFace Cache Clear (For instant recognition)
        pkl_path = os.path.join(FACES_DB, "representations_vgg_face.pkl")
        if os.path.exists(pkl_path):
            os.remove(pkl_path)

        temp_path = "temp_face.jpg"
        cv2.imwrite(temp_path, face_crop)
        
        results = DeepFace.find(
            img_path=temp_path, 
            db_path=FACES_DB, 
            model_name='VGG-Face', 
            enforce_detection=False, 
            silent=True
        )

        if len(results) > 0 and not results[0].empty:
            full_path = results[0].iloc[0]['identity']
            if os.path.exists(temp_path): os.remove(temp_path)
            return os.path.basename(full_path).split('.')[0]
        
        if os.path.exists(temp_path): os.remove(temp_path)
    except: pass
    return "Unknown_N/A"

# 🔴 FIX 3: Optimized save_to_report (Lag-Free)
def save_to_report(v_type, v_conf, is_unsafe, worker_info, user_email):
    if not is_unsafe: return

    now = time.time()
    # Timer: Registered = 60s, Unknown = 5s (taaki stream smooth rahe)
    gap = 60 if worker_info != "Unknown_N/A" else 5
    
    last_seen = st.session_state.worker_cooldowns.get(worker_info, 0)
    if now - last_seen < gap:
        return 

    try:
        # Info Split
        name, wid = worker_info.split("_") if "_" in worker_info else (worker_info, "N/A")
        clean_eq = v_type.lower().replace("no", "").replace("-", "").strip().capitalize()

        # Database Write with Timeout
        conn = sqlite3.connect("safety_violations.db", timeout=5)
        conn.execute('''INSERT INTO violations (timestamp, type, status, equipment, worker_name, worker_id, confidence) 
                        VALUES (?, ?, ?, ?, ?, ?, ?)''', 
                     (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), v_type, "⚠️ Unsafe", clean_eq, name, wid, float(v_conf)))
        conn.commit()
        conn.close()

        # Update Timer
        st.session_state.worker_cooldowns[worker_info] = now

        # Webhook Alert (Async-like timeout)
        if v_conf > 0.75 and user_email and "@" in user_email:
            payload = {"worker": name, "worker_id": wid, "violation": clean_eq, "confidence": f"{v_conf:.2f}",
                       "time": datetime.now().strftime("%I:%M %p"), "email": user_email}
            try: requests.post(N8N_URL, json=payload, timeout=0.5) # Bahut kam timeout taaki freeze na ho
            except: pass
    except Exception as e:
        print(f"Reporting Error: {e}") # Yahan 'e' ab error nahi dega

def run_detection(frame, user_email):
    try:
        _, img_encoded = cv2.imencode('.jpg', frame)
        response = requests.post(API_URL, files={'file': img_encoded.tobytes()}, timeout=5)
        if response.status_code == 200:
            data = response.json()
            for det in data.get('detections', []):
                label, conf = str(det.get('class', '')).lower(), float(det.get('conf', 0))
                if conf < 0.35: continue
                
                x1, y1, x2, y2 = map(int, det.get('bbox', [0,0,0,0]))
                is_unsafe = any(word in label for word in ["no", "missing", "unsafe", "without", "off"])
                
                worker_info = "Unknown_N/A"
                if is_unsafe:
                    face_crop = frame[max(0, y1):y2, max(0, x1):x2]
                    if face_crop.size > 0: worker_info = identify_worker(face_crop)
                    save_to_report(label, conf, True, worker_info, user_email)
                    color = (0, 0, 255)
                else: color = (0, 255, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
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
    menu = st.radio("Navigation", ["📊 Analytics", "👤 Worker Database", "🎥 Live Monitoring"])
    target_email = st.text_input("Alert Email", placeholder="user@example.com").strip()

if menu == "📊 Analytics":
    st.header("📊 Dashboard")
    conn = sqlite3.connect("safety_violations.db")
    df = pd.read_sql_query("SELECT * FROM violations", conn)
    conn.close()
    if not df.empty:
        st.metric("⚠️ Total Violations", len(df))
        st.dataframe(df.sort_values(by="timestamp", ascending=False), use_container_width=True)
        st.plotly_chart(px.pie(df, names='worker_name', hole=0.5, template="plotly_dark"), use_container_width=True)
    else: st.info("No logs found.")

elif menu == "👤 Worker Database":
    st.header("👤 Registration")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Name")
        wid = st.text_input("ID")
        img_file = st.camera_input("Capture")
        if st.button("Register") and name and wid and img_file:
            path = os.path.join(FACES_DB, f"{name.replace(' ', '_')}_{wid}.jpg")
            Image.open(img_file).convert("RGB").save(path)
            # Clear cache
            pkl = os.path.join(FACES_DB, "representations_vgg_face.pkl")
            if os.path.exists(pkl): os.remove(pkl)
            st.success(f"Registered {name}!")
            st.rerun()
    with col2:
        st.subheader("Registered List")
        if os.path.exists(FACES_DB):
            for f in os.listdir(FACES_DB):
                if f.endswith(".jpg"): st.text(f"✅ {f.split('.')[0].replace('_', ' ')}")

elif menu == "🎥 Live Monitoring":
    st.header("🎥 Monitoring")
    webrtc_streamer(key="live", mode=WebRtcMode.SENDRECV, 
                    video_processor_factory=lambda: VideoProcessor(target_email),
                    async_processing=True)
