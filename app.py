import streamlit as st
import cv2
import requests
import numpy as np
import pandas as pd
import sqlite3
import os
import time
import av
import matplotlib.pyplot as plt
from PIL import Image
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
    conn.commit()
    conn.close()

init_db()

# --- HELPER FUNCTIONS ---
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

def save_to_report(v_type, v_conf, is_unsafe, worker_info, user_email):
    if not is_unsafe: return
    try:
        name, wid = worker_info.split("_") if "_" in worker_info else (worker_info, "N/A")
        clean_eq = v_type.lower().replace("no", "").replace("-", "").strip().capitalize()
        conn = sqlite3.connect("safety_violations.db")
        conn.execute('''INSERT INTO violations (timestamp, type, status, equipment, worker_name, worker_id, confidence) 
                        VALUES (?, ?, ?, ?, ?, ?, ?)''',
                     (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), v_type, "⚠️ Unsafe", clean_eq, name, wid, float(v_conf)))
        conn.commit()
        conn.close()
        
        # Email Alert Logic (Pipedream)
        if v_conf > 0.6:
            payload = {"worker": name, "equipment": clean_eq, "email": user_email}
            requests.post(N8N_URL, json=payload, timeout=1)
    except: pass

def run_detection(frame, user_email):
    try:
        _, img_encoded = cv2.imencode('.jpg', frame)
        response = requests.post(API_URL, files={'file': img_encoded.tobytes()}, timeout=4)
        if response.status_code == 200:
            detections = response.json().get('detections', [])
            for det in detections:
                label, conf = det['class'], det['conf']
                x1, y1, x2, y2 = map(int, det['bbox'])
                is_unsafe = any(w in label.lower() for w in ["no", "missing", "unsafe"])
                
                worker_info = "Unknown"
                if is_unsafe:
                    face_crop = frame[max(0,y1):y2, max(0,x1):x2]
                    if face_crop.size > 0: worker_info = identify_worker(face_crop)
                    save_to_report(label, conf, True, worker_info, user_email)

                color = (0, 0, 255) if is_unsafe else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f"{label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    except: pass
    return frame

# --- WEBRTC PROCESSOR ---
class VideoProcessor(VideoProcessorBase):
    def __init__(self, user_email):
        self.user_email = user_email
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        return av.VideoFrame.from_ndarray(run_detection(img, self.user_email), format="bgr24")

# --- UI SETUP ---
st.set_page_config(page_title="Safe-Guard AI", layout="wide")

with st.sidebar:
    st.title("🛡️ SAFE-GUARD AI")
    menu = st.radio("Navigation", ["📊 Analytics", "👤 Worker Database", "🎥 Live Monitoring"])
    target_email = st.text_input("Alert Email", placeholder="user@example.com")

# --- PAGES ---
if menu == "📊 Analytics":
    st.header("📊 Violation Insights")
    conn = sqlite3.connect("safety_violations.db")
    df = pd.read_sql_query("SELECT * FROM violations", conn)
    conn.close()

    if not df.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Equipment Violations Count")
            st.bar_chart(df['equipment'].value_counts())
        with c2:
            st.subheader("Violation Distribution (%)")
            fig, ax = plt.subplots()
            df['equipment'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
            st.pyplot(fig)
        
        st.subheader("Detailed Logs")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Abhi tak koi violation data nahi mila.")

elif menu == "👤 Worker Database":
    st.header("👤 Register New Worker")
    name = st.text_input("Worker Name")
    wid = st.text_input("Worker ID")
    img_file = st.camera_input("Take Photo")
    if st.button("Register") and img_file and name:
        Image.open(img_file).save(os.path.join(FACES_DB, f"{name}_{wid}.jpg"))
        st.success("Worker Registered!")

 elif menu == "🎥 Live Monitoring":
    st.header("Live AI Safety Feed")
    elif menu == "🎥 Live Monitoring":
    st.header("Live AI Safety Feed")
    
    # STUN servers add karne se connection fast aur stable ho jata hai
    webrtc_streamer(
        key="cam", 
        mode=WebRtcMode.SENDRECV, 
        video_processor_factory=lambda: VideoProcessor(target_email),
        rtc_configuration={  # Ye hissa connection error khatam karega
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={
            "video": True, 
            "audio": False
        },
        async_processing=True, # Ise True kar dein taaki interface lag na kare
    )
