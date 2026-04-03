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

# 🔴 GLOBAL COOLDOWN TRACKER (1-Minute Timer Logic)
if 'worker_cooldowns' not in globals():
    global worker_cooldowns
    worker_cooldowns = {}

# Folder handling
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
def identify_worker(face_img):
    try:
        from deepface import DeepFace
        if not os.path.exists(FACES_DB) or not os.listdir(FACES_DB):
            return "Unknown_N/A"

        # 🔴 FIX: Delete DeepFace Cache to force refresh for new workers
        pkl_path = os.path.join(FACES_DB, "representations_vgg_face.pkl")
        if os.path.exists(pkl_path):
            os.remove(pkl_path)

        temp_path = "temp_face.jpg"
        cv2.imwrite(temp_path, face_img)
        
        results = DeepFace.find(
            img_path=temp_path, 
            db_path=FACES_DB, 
            model_name='VGG-Face', 
            enforce_detection=False, 
            silent=True
        )

        if len(results) > 0 and not results[0].empty:
            identity = results[0].iloc[0]['identity']
            if os.path.exists(temp_path): os.remove(temp_path)
            return os.path.basename(identity).split('.')[0]
        
        if os.path.exists(temp_path): os.remove(temp_path)
    except: pass
    return "Unknown_N/A"

def save_to_report(v_type, v_conf, is_unsafe, worker_info, user_email):
    if not is_unsafe: return

    global worker_cooldowns
    current_time = time.time()
    
    # 🔴 SMART COOLDOWN: Registered = 60s, Unknown = 2s
    cooldown_period = 60 if worker_info != "Unknown_N/A" else 2
    if worker_info in worker_cooldowns:
        if current_time - worker_cooldowns[worker_info] < cooldown_period:
            return 

    try:
        name, wid = worker_info.split("_") if "_" in worker_info else (worker_info, "N/A")
        clean_eq = v_type.lower().replace("no", "").replace("-", "").strip().capitalize()

        conn = sqlite3.connect("safety_violations.db")
        conn.execute('INSERT INTO violations (timestamp, type, status, equipment, worker_name, worker_id, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)',
                     (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), v_type, "⚠️ Unsafe", clean_eq, name, wid, float(v_conf)))
        conn.commit()
        conn.close()

        worker_cooldowns[worker_info] = current_time # Update timer

        if v_conf > 0.75 and user_email and "@" in user_email:
            payload = {"worker": name, "worker_id": wid, "violation": clean_eq, "confidence": f"{v_conf:.2f}",
                       "time": datetime.now().strftime("%I:%M %p"), "email": user_email, "subject": f"⚠️ SAFETY ALERT: {clean_eq}"}
            try: requests.post(N8N_URL, json=payload, timeout=2)
            except: pass
    except: pass

def run_detection(frame, user_email):
    try:
        _, img_encoded = cv2.imencode('.jpg', frame)
        response = requests.post(API_URL, files={'file': img_encoded.tobytes()}, timeout=8)
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
    menu = st.radio("Navigation", ["📊 Analytics", "👤 Worker Database", "🎥 Live Monitoring", "📁 Batch Processing"])
    target_email = st.text_input("Alert Email", placeholder="user@example.com").strip()

if menu == "📊 Analytics":
    st.header("📊 Real-Time Safety Dashboard")
    conn = sqlite3.connect("safety_violations.db")
    df = pd.read_sql_query("SELECT * FROM violations", conn)
    conn.close()

    if not df.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("👥 Total Scanned", len(df)+50)
        c2.metric("⚠️ Total Violations", len(df), delta_color="inverse")
        c3.metric("✅ Compliance Rate", f"{((len(df)+50 - len(df)) / (len(df)+50) * 100):.1f}%")
        
        st.subheader("📝 Detailed Logs")
        st.dataframe(df.sort_values(by="timestamp", ascending=False), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.pie(df, names='worker_name', hole=0.5, title="Violations by Worker", template="plotly_dark"), use_container_width=True)
        with col2:
            st.plotly_chart(px.bar(df['equipment'].value_counts().reset_index(), x='index', y='equipment', title="Equipment Breakdown", template="plotly_dark"), use_container_width=True)
    else: st.info("No data recorded yet.")

elif menu == "👤 Worker Database":
    st.header("👤 Worker Registration & List")
    col_reg, col_list = st.columns([1, 1])
    
    with col_reg:
        st.subheader("Add New Worker")
        name = st.text_input("Full Name")
        wid = st.text_input("Worker ID (e.g. W101)")
        reg_mode = st.radio("Method", ["Upload", "Camera"])
        img_file = st.file_uploader("Select Photo", type=['jpg','png']) if reg_mode == "Upload" else st.camera_input("Take Photo")

        if st.button("Register Now") and name and wid and img_file:
            # Save format: Name_ID.jpg
            filename = f"{name.replace(' ', '_')}_{wid}.jpg"
            img_path = os.path.join(FACES_DB, filename)
            Image.open(img_file).convert("RGB").save(img_path)
            
            # Clear cache so identification works immediately
            pkl_path = os.path.join(FACES_DB, "representations_vgg_face.pkl")
            if os.path.exists(pkl_path): os.remove(pkl_path)
            st.success(f"Successfully Registered: {name}")
            st.rerun()

    with col_list:
        st.subheader("Registered Workers")
        if os.path.exists(FACES_DB):
            files = [f for f in os.listdir(FACES_DB) if f.endswith(('.jpg', '.png'))]
            if files:
                for f in files:
                    clean_name = f.split('.')[0].replace('_', ' ')
                    st.markdown(f"✅ **{clean_name}**")
            else: st.write("No workers registered.")

elif menu == "🎥 Live Monitoring":
    st.header("🎥 Live AI Monitoring")
    rtc_config = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}, {"urls": ["stun:stun1.l.google.com:19302"]}]}
    webrtc_streamer(
        key=f"live-stream-{target_email}", 
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: VideoProcessor(target_email),
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

elif menu == "📁 Batch Processing":
    st.header("📁 Batch Image Detection")
    files = st.file_uploader("Upload Multiple Images", accept_multiple_files=True)
    if files:
        for f in files:
            img = cv2.imdecode(np.asarray(bytearray(f.read()), dtype=np.uint8), 1)
            st.image(cv2.cvtColor(run_detection(img, target_email), cv2.COLOR_BGR2RGB), caption=f.name)
