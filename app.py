import streamlit as st
import cv2
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

if not os.path.exists(FACES_DB):
    os.makedirs(FACES_DB, exist_ok=True)

# --- DATABASE ---
def init_db():
    conn = sqlite3.connect("safety_violations.db", check_same_thread=False)
    conn.execute('''CREATE TABLE IF NOT EXISTS violations 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, type TEXT, 
                     status TEXT, equipment TEXT, worker_name TEXT, worker_id TEXT, confidence REAL)''')
    conn.commit()
    conn.close()

init_db()

# --- HELPER FUNCTIONS ---
def get_registered_workers():
    """Returns a list of worker names from the FACES_DB folder"""
    files = [f for f in os.listdir(FACES_DB) if f.endswith(('.jpg', '.png', '.jpeg'))]
    return [f.split('_')[0] for f in files]

def identify_worker(face_img):
    try:
        from deepface import DeepFace
        temp_path = "current_frame.jpg"
        cv2.imwrite(temp_path, face_img)
        
        # Representations file refresh
        pkl_path = os.path.join(FACES_DB, "representations_vgg_face.pkl")
        if os.path.exists(pkl_path):
            os.remove(pkl_path)

        results = DeepFace.find(
            img_path=temp_path, 
            db_path=FACES_DB, 
            model_name='VGG-Face', 
            distance_metric='cosine',
            enforce_detection=False, 
            detector_backend='opencv', # Faster for real-time
            silent=True
        )

        if len(results) > 0 and not results[0].empty:
            best_match = results[0].iloc[0]
            # Threshold adjust: 0.4 se kam distance matlab solid match
            if best_match['distance'] < 0.4: 
                identity = best_match['identity']
                return os.path.basename(identity).split('.')[0]
                
    except Exception as e:
        print(f"Recognition Error: {e}")
    return "Unknown"

def run_detection(frame, user_email):
    try:
        _, img_encoded = cv2.imencode('.jpg', frame)
        response = requests.post(API_URL, files={'file': img_encoded.tobytes()}, timeout=5)
        
        if response.status_code == 200:
            detections = response.json().get('detections', [])
            for det in detections:
                label = str(det.get('class', '')).lower()
                conf = float(det.get('conf', 0))
                if conf < 0.45: continue
                
                x1, y1, x2, y2 = map(int, det.get('bbox', [0,0,0,0]))
                is_unsafe = any(word in label for word in ["no", "missing", "unsafe"])
                
                name_display = "Scanning..."
                color = (0, 255, 0)

                if is_unsafe:
                    color = (0, 0, 255)
                    # Face Identification Logic
                    # Hum box ko upar se thoda aur expand kar rahe hain taaki face pura aaye
                    face_h = int((y2 - y1) * 0.5)
                    face_crop = frame[max(0, y1-20):y1+face_h, x1:x2]
                    
                    if face_crop.size > 0:
                        name_display = identify_worker(face_crop)
                    
                    # Log to DB
                    # (Optional: isko thoda delay karke save karein taaki duplicate logs na banein)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name_display}: {label}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    except:
        pass
    return frame

class VideoProcessor(VideoProcessorBase):
    def __init__(self, email):
        self.email = email
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = run_detection(img, self.email)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI INTERFACE ---
st.set_page_config(page_title="Safe-Guard AI", layout="wide")

with st.sidebar:
    st.title("🛡️ SAFE-GUARD AI")
    menu = st.radio("Go to", ["📊 Analytics", "👤 Worker Database", "🎥 Live Monitoring"])
    
    st.markdown("---")
    st.subheader("📋 Registered Workers")
    workers = get_registered_workers()
    if workers:
        for w in set(workers): # set() for unique names
            st.write(f"✅ {w}")
    else:
        st.write("No workers registered yet.")
    
    st.markdown("---")
    target_email = st.text_input("Alert Email", placeholder="manager@site.com")

if menu == "📊 Analytics":
    st.header("📊 Safety Dashboard")
    conn = sqlite3.connect("safety_violations.db")
    df = pd.read_sql_query("SELECT * FROM violations", conn)
    conn.close()

    if not df.empty:
        c1, c2 = st.columns(2)
        c1.metric("Total Violations", len(df))
        st.plotly_chart(px.pie(df, names='worker_name', title="Violations by Worker", hole=0.4), use_container_width=True)
        st.dataframe(df.sort_values(by="timestamp", ascending=False), use_container_width=True)
    else:
        st.info("No logs in database yet.")

elif menu == "👤 Worker Database":
    st.header("👤 Worker Registration")
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Worker Name (e.g. Ali)")
        wid = st.text_input("Worker ID (e.g. 101)")
        source = st.radio("Registration Source", ["Upload", "Camera"])
        img_file = st.file_uploader("Upload Face Image", type=['jpg','png']) if source == "Upload" else st.camera_input("Capture Face")

    if st.button("Register & Save"):
        if name and wid and img_file:
            clean_name = name.strip().replace(" ", "_")
            file_path = os.path.join(FACES_DB, f"{clean_name}_{wid}.jpg")
            img = Image.open(img_file).convert("RGB")
            img.save(file_path)
            
            # Reset DeepFace Cache
            pkl = os.path.join(FACES_DB, "representations_vgg_face.pkl")
            if os.path.exists(pkl): os.remove(pkl)
            
            st.success(f"Worker {name} has been registered!")
            st.rerun() # Refresh to update sidebar list
        else:
            st.warning("Please provide Name, ID and Image.")

elif menu == "🎥 Live Monitoring":
    st.header("🎥 Real-Time Worker Recognition")
    if not workers:
        st.error("⚠️ Pehle 'Worker Database' mein ja kar kisi ko register karein, warna recognition kaam nahi karegi.")
    
    webrtc_streamer(
        key="safety-stream",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: VideoProcessor(target_email),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
