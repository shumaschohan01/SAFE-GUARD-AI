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
from deepface import DeepFace

# --- CONFIGURATION ---
API_URL = "https://shumaschohan-safeguard-ai.hf.space/predict/"
N8N_URL = "https://eom4pk834n2y9tj.m.pipedream.net"
FACES_DB = "worker_faces"

# Emergency Folder Cleanup & Ensure Directory
if os.path.exists(FACES_DB):
    if not os.path.isdir(FACES_DB):
        os.remove(FACES_DB)
        os.makedirs(FACES_DB, exist_ok=True)
else:
    os.makedirs(FACES_DB, exist_ok=True)

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect("safety_violations.db", check_same_thread=False)
    conn.execute('''CREATE TABLE IF NOT EXISTS violations 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                     timestamp TEXT, type TEXT, status TEXT,
                     equipment TEXT, worker_name TEXT,
                     worker_id TEXT, confidence REAL)''')
    conn.commit()
    conn.close()

init_db()

# --- ALERT & IDENTIFICATION LOGIC ---
def identify_worker(face_img):
    try:
        if not os.listdir(FACES_DB): return "Unknown_N/A"
        
        # Temp save for DeepFace processing
        temp_path = "current_scan.jpg"
        cv2.imwrite(temp_path, face_img)
        
        # Clear DeepFace cache to pick up new registrations
        for f in os.listdir(FACES_DB):
            if f.endswith(".pkl"): os.remove(os.path.join(FACES_DB, f))

        results = DeepFace.find(
            img_path=temp_path, 
            db_path=FACES_DB, 
            model_name='Facenet', 
            distance_metric='cosine',
            enforce_detection=False, 
            detector_backend='mtcnn', 
            silent=True
        )
        
        if len(results) > 0 and not results[0].empty:
            best_match = results[0].iloc[0]
            if best_match['distance'] < 0.55: # Cosine threshold
                full_path = best_match['identity']
                return os.path.basename(full_path).split('.')[0]
    except Exception as e:
        print(f"ID Error: {e}")
    return "Unknown_N/A"

def send_to_n8n(name, eq, conf, user_email):
    if not user_email or "@" not in user_email: return 
    payload = {
        "worker": name, "equipment": eq, "confidence": f"{conf:.2%}",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "receiver_email": user_email 
    }
    try: requests.post(N8N_URL, json=payload, timeout=2)
    except: pass

def save_to_report(v_type, v_conf, is_unsafe, worker_info, user_email):
    if "violation_cache" not in st.session_state: st.session_state.violation_cache = {}
    cache_key = f"{worker_info}_{v_type}"
    
    if cache_key in st.session_state.violation_cache:
        if time.time() - st.session_state.violation_cache[cache_key] < 30: return

    if not is_unsafe: return 
    try:
        name, wid = worker_info.split("_") if "_" in worker_info else (worker_info, "N/A")
        clean_eq = v_type.lower().replace("no", "").replace("-", "").strip().capitalize()
        conn = sqlite3.connect("safety_violations.db")
        conn.execute('''INSERT INTO violations (timestamp, type, status, equipment, worker_name, worker_id, confidence) 
                        VALUES (?, ?, ?, ?, ?, ?, ?)''',
                     (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), v_type, "⚠️ Unsafe", clean_eq, name, wid, float(v_conf)))
        conn.commit(); conn.close()
        
        st.session_state.violation_cache[cache_key] = time.time()
        if float(v_conf) > 0.60: send_to_n8n(name, clean_eq, float(v_conf), user_email)
    except: pass

# --- DETECTION ENGINE ---
def run_detection(frame, user_email):
    try:
        _, img_encoded = cv2.imencode('.jpg', frame)
        response = requests.post(API_URL, files={'file': img_encoded.tobytes()}, timeout=5)
        detections = response.json().get('detections', [])
        for det in detections:
            label, conf = det['class'], det['conf']
            x1, y1, x2, y2 = map(int, det['bbox'])
            is_unsafe = any(w in label.lower() for w in ["no", "missing", "unsafe"])
            
            worker_info = "Scanning..."
            if is_unsafe:
                # Add padding for better face capture
                y_top = max(0, y1 - 40)
                y_bot = min(frame.shape[0], y1 + int((y2-y1)*0.6))
                face_crop = frame[y_top:y_bot, max(0, x1-20):min(frame.shape[1], x2+20)]
                
                if face_crop.size > 0: 
                    worker_info = identify_worker(face_crop)
                save_to_report(label, conf, True, worker_info, user_email)
            
            color = (0, 0, 255) if is_unsafe else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            display_name = worker_info.split('_')[0]
            cv2.putText(frame, f"{display_name}: {label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    except: pass
    return frame

# --- WEBRTC ---
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
    menu = st.radio("Navigation", ["📊 Analytics", "👤 Worker Database", "🎥 Live Monitoring", "📁 Batch"])
    st.markdown("---")
    st.subheader("⚙️ Alert Settings")
    target_email = st.text_input("Alert Email", placeholder="example@gmail.com")
    if not target_email:
        st.warning("⚠️ Email likhein alerts ke liye.")

# --- PAGES ---
if menu == "📊 Analytics":
    st.header("📊 Violation Insights")
    conn = sqlite3.connect("safety_violations.db")
    df = pd.read_sql_query("SELECT * FROM violations", conn)
    conn.close()
    if not df.empty:
        st.metric("Total Violations", len(df))
        st.dataframe(df.sort_values(by='timestamp', ascending=False), use_container_width=True)
    else: st.info("No data found.")

elif menu == "👤 Worker Database":
    st.header("👤 Personnel Management")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Register New Worker")
        reg_method = st.radio("Source", ["Camera", "Upload"])
        new_name = st.text_input("Name")
        new_id = st.text_input("Worker ID")
        img_input = st.camera_input("Take Photo") if reg_method == "Camera" else st.file_uploader("Upload Image", type=['jpg', 'png'])

        if st.button("Register Now"):
            if new_name and new_id and img_input:
                clean_name = new_name.strip().replace(" ", "_")
                clean_id = new_id.strip()
                filename = f"{clean_name}_{clean_id}.jpg"
                save_path = os.path.join(FACES_DB, filename)

                try:
                    img = Image.open(img_input).convert("RGB")
                    img.save(save_path)
                    st.success(f"✅ {new_name} registered!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Please fill all details.")

    with col2:
        st.subheader("Current Database")
        for f in os.listdir(FACES_DB):
            if f.endswith(('.jpg', '.png')):
                c1, c2 = st.columns([4, 1])
                c1.write(f"👤 {f.split('.')[0]}")
                if c2.button("🗑️", key=f):
                    os.remove(os.path.join(FACES_DB, f))
                    st.rerun()

elif menu == "🎥 Live Monitoring":
    st.header("🎥 Real-Time Safety Feed")
    webrtc_streamer(
        key="safety-cam", 
        mode=WebRtcMode.SENDRECV, 
        video_processor_factory=lambda: VideoProcessor(target_email),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

elif menu == "📁 Batch":
    st.header("📁 Media Analysis")
    f = st.file_uploader("Select Image", type=['jpg','png','jpeg'])
    if f:
        img_np = cv2.cvtColor(np.array(Image.open(f)), cv2.COLOR_RGB2BGR)
        processed = run_detection(img_np, target_email)
        st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True)
