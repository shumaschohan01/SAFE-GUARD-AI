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

# --- CONFIGURATION ---
FACES_DB = "worker_faces"

# Robust folder handling
if os.path.exists(FACES_DB):
    if not os.path.isdir(FACES_DB):
        # Agar worker_faces naam ki koi file pehle se maujood hai jo folder nahi hai
        os.remove(FACES_DB)
        os.makedirs(FACES_DB)
else:
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
def identify_worker(face_img):
    try:
        from deepface import DeepFace
        temp_path = "current_frame.jpg"
        cv2.imwrite(temp_path, face_img)
        
        # Representations file ko delete karein taaki har baar naya data scan ho
        pkl_path = os.path.join(FACES_DB, "representations_vgg_face.pkl")
        if os.path.exists(pkl_path):
            os.remove(pkl_path)

        results = DeepFace.find(
            img_path=temp_path, 
            db_path=FACES_DB, 
            model_name='VGG-Face', 
            distance_metric='cosine', # Cosine similarity behtar kaam karti hai
            enforce_detection=False, 
            detector_backend='retinaface', # 'opencv' fast hai magar 'retinaface' zyada accurate hai
            silent=True
        )

        if len(results) > 0 and not results[0].empty:
            best_match = results[0].iloc[0]
            # Agar distance 0.4 se zyada hai toh matlab match weak hai
            if best_match['distance'] < 0.4: 
                identity = best_match['identity']
                return os.path.basename(identity).split('.')[0]
                
    except Exception as e:
        print(f"Recognition Error: {e}")
    return "Unknown_N/A"

def save_to_report(v_type, v_conf, worker_info, user_email):
    try:
        name, wid = worker_info.split("_") if "_" in worker_info else (worker_info, "N/A")
        clean_eq = v_type.lower().replace("no", "").replace("-", "").strip().capitalize()

        conn = sqlite3.connect("safety_violations.db")
        conn.execute('''
            INSERT INTO violations (timestamp, type, status, equipment, worker_name, worker_id, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), v_type, "⚠️ Unsafe", clean_eq, name, wid, float(v_conf)))
        conn.commit()
        conn.close()

        # Alert if confidence is high
        if v_conf > 0.70 and user_email and "@" in user_email:
            payload = {"worker": name, "worker_id": wid, "violation": clean_eq, 
                       "confidence": f"{v_conf:.2f}", "time": datetime.now().strftime("%I:%M %p"), "email": user_email}
            requests.post(N8N_URL, json=payload, timeout=2)
    except Exception as e:
        print(f"DB/Alert Error: {e}")

def run_detection(frame, user_email):
    try:
        _, img_encoded = cv2.imencode('.jpg', frame)
        response = requests.post(API_URL, files={'file': img_encoded.tobytes()}, timeout=5)
        
        if response.status_code == 200:
            detections = response.json().get('detections', [])
            for det in detections:
                label = str(det.get('class', '')).lower()
                conf = float(det.get('conf', 0))
                if conf < 0.40: continue
                
                x1, y1, x2, y2 = map(int, det.get('bbox', [0,0,0,0]))
                is_unsafe = any(word in label for word in ["no", "missing", "unsafe"])
                
                worker_name = "Scanning..."
                color = (0, 255, 0) # Default Green

                if is_unsafe:
                    color = (0, 0, 255) # Red
                    head_height = int((y2 - y1) * 0.4)
                    face_crop = frame[y1:y1+head_height, x1:x2]
                    if face_crop.size > 0:
                        worker_name = identify_worker(face_crop)
                    
                    save_to_report(label, conf, worker_name, user_email)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{worker_name.split('_')[0]}: {label}", (x1, y1-10), 
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
    menu = st.radio("Go to", ["📊 Analytics", "👤 Worker Database", "🎥 Live Monitoring", "📁 Batch Processing"])
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
        st.info("No data found.")

elif menu == "👤 Worker Database":
    st.header("👤 Register New Worker")
    name = st.text_input("Full Name")
    wid = st.text_input("Worker ID")
    source = st.radio("Source", ["Upload Image", "Use Camera"])
    
    img_file = st.file_uploader("Upload Face", type=['jpg','png']) if source == "Upload Image" else st.camera_input("Take Photo")

    if st.button("Register Now"):
        if name and wid and img_file:
            # Save file with clean name format
            clean_name = name.replace(" ", "_")
            file_path = os.path.join(FACES_DB, f"{clean_name}_{wid}.jpg")
            img = Image.open(img_file).convert("RGB")
            img.save(file_path)
            
            # Clear DeepFace cache to recognize new person immediately
            pkl = os.path.join(FACES_DB, "representations_vgg_face.pkl")
            if os.path.exists(pkl): os.remove(pkl)
            
            st.success(f"✅ {name} registered successfully!")
        else:
            st.error("Please fill all fields.")

elif menu == "🎥 Live Monitoring":
    st.header("🎥 Live Safety Feed")
    webrtc_streamer(
        key="safety-stream",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: VideoProcessor(target_email),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

elif menu == "📁 Batch Processing":
    st.header("📁 Process Images")
    uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True)
    if uploaded_files:
        for f in uploaded_files:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            processed = run_detection(frame, target_email)
            st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), caption=f.name)
