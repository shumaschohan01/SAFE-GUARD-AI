import streamlit as st
import cv2
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
FACES_DB = "worker_faces"

# Safe folder handling
if os.path.exists(FACES_DB):
    if not os.path.isdir(FACES_DB):
        os.remove(FACES_DB)
        os.makedirs(FACES_DB)
else:
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
def identify_worker(face_img):
    try:
        from deepface import DeepFace
        if not os.path.exists(FACES_DB) or not os.listdir(FACES_DB):
            return "Unknown_N/A"

        temp_path = "temp_face.jpg"
        cv2.imwrite(temp_path, face_img)
        results = DeepFace.find(img_path=temp_path, db_path=FACES_DB, enforce_detection=False, silent=True)

        if len(results) > 0 and not results[0].empty:
            identity = results[0].iloc[0]['identity']
            if os.path.exists(temp_path): os.remove(temp_path)
            return os.path.basename(identity).split('.')[0]
        
        if os.path.exists(temp_path): os.remove(temp_path)
    except Exception as e:
        print("Face Error:", e)
    return "Unknown_N/A"

def save_to_report(v_type, v_conf, is_unsafe, worker_info, user_email):
    if not is_unsafe: return
    try:
        name, wid = worker_info.split("_") if "_" in worker_info else (worker_info, "N/A")
        clean_eq = v_type.lower().replace("no", "").replace("-", "").strip().capitalize()
        conn = sqlite3.connect("safety_violations.db")
        conn.execute('INSERT INTO violations (timestamp, type, status, equipment, worker_name, worker_id, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)',
                     (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), v_type, "⚠️ Unsafe", clean_eq, name, wid, float(v_conf)))
        conn.commit()
        conn.close()

        if v_conf > 0.75 and user_email and "@" in user_email:
            payload = {"worker": name, "worker_id": wid, "violation": clean_eq, "confidence": f"{v_conf:.2f}",
                       "time": datetime.now().strftime("%I:%M %p"), "email": user_email, "subject": f"⚠️ SAFETY ALERT: {clean_eq}"}
            try: requests.post(N8N_URL, json=payload, timeout=3)
            except: pass
    except Exception as e:
        print("DB Error:", e)

def run_detection(frame, user_email):
    try:
        _, img_encoded = cv2.imencode('.jpg', frame)
        response = requests.post(API_URL, files={'file': img_encoded.tobytes()}, timeout=8)
        if response.status_code == 200:
            data = response.json()
            detections = data.get('detections', [])
            for det in detections:
                label = str(det.get('class', '')).lower()
                conf = float(det.get('conf', 0))
                bbox = det.get('bbox', [0, 0, 0, 0])
                if conf < 0.40: continue
                x1, y1, x2, y2 = map(int, bbox)
                is_unsafe = any(word in label for word in ["no", "missing", "unsafe", "without", "off"])
                worker_info = "Unknown_N/A"
                if is_unsafe:
                    face_crop = frame[max(0, y1):y2, max(0, x1):x2]
                    if face_crop.size > 0: worker_info = identify_worker(face_crop)
                    save_to_report(label, conf, True, worker_info, user_email)
                    color = (0, 0, 255)
                else: color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
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
    st.header("📊 Dashboard")

    conn = sqlite3.connect("safety_violations.db")
    df = pd.read_sql_query("SELECT * FROM violations", conn)
    conn.close()

    total_violations = len(df)
    total_scanned = total_violations + 50  # 🔴 CHANGED dummy logic

    compliance = ((total_scanned - total_violations) / total_scanned * 100) if total_scanned else 100

    c1, c2, c3 = st.columns(3)
    c1.metric("👥 Scanned", total_scanned)
    c2.metric("⚠️ Violations", total_violations)
    c3.metric("✅ Compliance", f"{compliance:.1f}%")

    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        st.line_chart(df.resample('H', on='timestamp').count()['id'])

        fig, ax = plt.subplots()
        df['worker_name'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)

        st.dataframe(df)

elif menu == "👤 Worker Database":
    st.header("👤 Worker Registration")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Name")
        wid = st.text_input("Worker ID")
        reg_mode = st.radio("Method", ["Upload Photo", "Live Camera"], key="reg_mode")
        
        # Unique keys are CRITICAL here
        if reg_mode == "Upload Photo":
            img_file = st.file_uploader("Choose Image", type=['jpg','png','jpeg'], key="upload_key")
        else:
            img_file = st.camera_input("Capture Face", key="worker_cam_key")

        if st.button("Register Now") and name and wid and img_file:
            img_path = os.path.join(FACES_DB, f"{name.replace(' ', '_')}_{wid}.jpg")
            img = Image.open(img_file)
            if img.mode != "RGB": img = img.convert("RGB")
            img.save(img_path)
            st.success(f"Registered {name}!")

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
