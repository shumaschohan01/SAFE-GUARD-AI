import streamlit as st
import cv2
import requests
import numpy as np
import pandas as pd
import sqlite3
import os
import time
import av
from PIL import Image
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from deepface import DeepFace # DeepFace ko requirements.txt mein lazmi rakhein

# --- CONFIGURATION ---
API_URL = st.secrets.get("https://shumaschohan-safeguard-api.hf.space/predict/")
N8N_URL = st.secrets.get("N8N_URL", "http://localhost:5678/webhook/safety-alert")
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

# --- FACE IDENTIFICATION LOGIC ---
def identify_worker(face_img):
    try:
        if not os.listdir(FACES_DB): return "Unknown_N/A"
        # Face match check
        results = DeepFace.find(img_path=face_img, db_path=FACES_DB, enforce_detection=False, silent=True)
        if len(results) > 0 and not results[0].empty:
            # File name se Name aur ID nikalna (e.g., Shumas_123.jpg)
            full_path = results[0].iloc[0]['identity']
            return os.path.basename(full_path).split('.')[0]
    except: pass
    return "Unknown_N/A"

# --- DUPLICATE & ALERT LOGIC ---
def is_duplicate_violation(worker_info, v_type):
    if "violation_cache" not in st.session_state: st.session_state.violation_cache = {}
    current_time = time.time()
    cache_key = f"{worker_info}_{v_type}"
    if cache_key in st.session_state.violation_cache:
        if current_time - st.session_state.violation_cache[cache_key] < 30: return True
    st.session_state.violation_cache[cache_key] = current_time
    return False

def save_to_report(v_type, v_conf, is_unsafe, worker_info):
    if not is_unsafe or is_duplicate_violation(worker_info, v_type): return 
    try:
        name, wid = worker_info.split("_") if "_" in worker_info else (worker_info, "N/A")
        clean_eq = v_type.lower().replace("no", "").replace("-", "").strip().capitalize()
        conn = sqlite3.connect("safety_violations.db")
        conn.execute('''INSERT INTO violations (timestamp, type, status, equipment, worker_name, worker_id, confidence) 
                        VALUES (?, ?, ?, ?, ?, ?, ?)''',
                     (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), v_type, "⚠️ Unsafe", clean_eq, name, wid, float(v_conf)))
        conn.commit(); conn.close()
        if float(v_conf) > 0.60:
            payload = {"worker": name, "equipment": clean_eq, "confidence": f"{v_conf:.2%}", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            requests.post(N8N_URL, json=payload, timeout=2)
    except: pass

# --- DETECTION ENGINE ---
def run_detection(frame):
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
                # Face crop karke identify karna
                face_crop = frame[max(0,y1):y2, max(0,x1):x2]
                if face_crop.size > 0: worker_info = identify_worker(face_crop)
                save_to_report(label, conf, True, worker_info)
            
            color = (0, 0, 255) if is_unsafe else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            display_name = worker_info.split('_')[0]
            cv2.putText(frame, f"{display_name}: {label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    except: pass
    return frame

# --- WEBRTC ---
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        return av.VideoFrame.from_ndarray(run_detection(img), format="bgr24")

# --- UI ---
st.set_page_config(page_title="Safe-Guard AI", layout="wide")
menu = st.sidebar.radio("Navigation", ["📊 Analytics", "👤 Worker Database", "🎥 Live Monitoring", "📁 Batch"])

if menu == "📊 Analytics":
    st.header("📊 Violation Insights")
    conn = sqlite3.connect("safety_violations.db")
    df = pd.read_sql_query("SELECT * FROM violations", conn)
    if not df.empty:
        st.metric("Total Violations", len(df))
        st.dataframe(df.sort_values(by='timestamp', ascending=False), use_container_width=True)
    else: st.info("No data recorded.")

elif menu == "👤 Worker Database":
    st.header("👤 Worker Registration")
    col1, col2 = st.columns(2)
    with col1:
        method = st.radio("Registration Method", ["Camera", "Upload"])
        name = st.text_input("Worker Name")
        emp_id = st.text_input("Employee ID")
        img_file = st.camera_input("Take Photo") if method == "Camera" else st.file_uploader("Upload Photo", type=['jpg', 'png'])
        
        if st.button("Register Worker") and img_file and name and emp_id:
            img = Image.open(img_file).convert('RGB')
            img.save(os.path.join(FACES_DB, f"{name}_{emp_id}.jpg"))
            st.success(f"Registered {name} successfully!")
            st.rerun()

    with col2:
        st.subheader("Registered Personnel")
        for f in os.listdir(FACES_DB):
            if "_" in f:
                c1, c2 = st.columns([4, 1])
                c1.write(f"✅ {f.split('.')[0]}")
                if c2.button("🗑️", key=f):
                    os.remove(os.path.join(FACES_DB, f))
                    st.rerun()

elif menu == "🎥 Live Monitoring":
    st.header("Live AI Guard")
    webrtc_streamer(key="cam", mode=WebRtcMode.SENDRECV, video_processor_factory=VideoProcessor, media_stream_constraints={"video": True, "audio": False}, async_processing=True)

elif menu == "📁 Batch":
    st.header("Media Analysis")
    f = st.file_uploader("Select Image", type=['jpg','png'])
    if f:
        img_np = cv2.cvtColor(np.array(Image.open(f)), cv2.COLOR_RGB2BGR)
        st.image(cv2.cvtColor(run_detection(img_np), cv2.COLOR_BGR2RGB), use_column_width=True)
