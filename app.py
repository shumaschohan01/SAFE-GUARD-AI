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

# --- CONFIGURATION & DIRECTORY FIX ---
API_URL = "https://shumaschohan-safeguard-ai.hf.space/predict/"
N8N_URL = "https://eom4pk834n2y9tj.m.pipedream.net"
FACES_DB = "worker_faces"

# Sakht check: Agar FACES_DB file hai toh delete karo, warna folder banao
if os.path.exists(FACES_DB):
    if not os.path.isdir(FACES_DB):
        os.remove(FACES_DB)
        os.makedirs(FACES_DB, exist_ok=True)
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
def get_registered_workers():
    """Registered workers ki unique list nikalta hai"""
    try:
        if not os.path.isdir(FACES_DB): return []
        files = [f for f in os.listdir(FACES_DB) if f.endswith(('.jpg', '.png', '.jpeg'))]
        return sorted(list(set([f.split('_')[0] for f in files])))
    except:
        return []

def identify_worker(face_img):
    try:
        from deepface import DeepFace
        import os

        # 1. Temp image save karein high quality mein
        temp_path = "debug_face.jpg"
        cv2.imwrite(temp_path, face_img)
        
        # 2. Representations cache ko lazmi delete karein har baar
        pkl_path = os.path.join(FACES_DB, "representations_vgg_face.pkl")
        if os.path.exists(pkl_path):
            os.remove(pkl_path)

        # 3. DeepFace Find logic
        # Model 'Facenet' ya 'VGG-Face' dono mein se koi bhi use kar sakte hain
        results = DeepFace.find(
            img_path=temp_path, 
            db_path=FACES_DB, 
            model_name='VGG-Face', 
            distance_metric='cosine',
            enforce_detection=False, 
            detector_backend='opencv', # 'retinaface' slow hai magar accurate hai, 'opencv' fast hai
            silent=True
        )

        if len(results) > 0 and not results[0].empty:
            best_match = results[0].iloc[0]
            # VGG-Face ke liye cosine distance 0.40 se kam hona chahiye match ke liye
            if best_match['distance'] < 0.40: 
                identity_path = best_match['identity']
                # File name se worker ka naam nikaalein (e.g., Ali_101.jpg -> Ali)
                worker_filename = os.path.basename(identity_path).split('_')[0]
                return worker_filename
                
    except Exception as e:
        print(f"DeepFace Match Error: {e}")
    
    return "Unknown"

def run_detection(frame, user_email, save_log=True):
    """YOLO Detection + Face Recognition ka main engine"""
    try:
        _, img_encoded = cv2.imencode('.jpg', frame)
        response = requests.post(API_URL, files={'file': img_encoded.tobytes()}, timeout=8)
        
        if response.status_code == 200:
            detections = response.json().get('detections', [])
            for det in detections:
                label = str(det.get('class', '')).lower()
                conf = float(det.get('conf', 0))
                if conf < 0.40: continue
                
                x1, y1, x2, y2 = map(int, det.get('bbox', [0,0,0,0]))
                is_unsafe = any(word in label for word in ["no", "missing", "unsafe"])
                
                worker_name = "Scanning..."
                color = (0, 255, 0) # Safe

                if is_unsafe:
                        color = (0, 0, 255) # Red for violation
                        # 🔴 Bounding box ko thoda upar aur side se barhaein (Padding)
                        padding = 20
                        y_start = max(0, y1 - padding)
                        y_end = min(frame.shape[0], y1 + int((y2-y1)*0.5)) # Box ka top 50% hissa
                        x_start = max(0, x1 - padding)
                        x_end = min(frame.shape[1], x2 + padding)
                        face_crop = frame[y_start:y_end, x_start:x_end]
    
                        if face_crop.size > 0:
                            worker_name = identify_worker(face_crop)
                    
                    if save_log:
                        # Database logging logic (optional: add rate limiting)
                        pass

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f"{worker_name}: {label}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
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

# --- STREAMLIT UI ---
st.set_page_config(page_title="Safe-Guard AI", layout="wide")

with st.sidebar:
    st.title("🛡️ SAFE-GUARD AI")
    menu = st.radio("Navigation", ["📊 Analytics", "👤 Worker Database", "🎥 Live Monitoring", "📁 Batch Processing"])
    
    st.markdown("---")
    st.subheader("📋 Registered Workers")
    worker_list = get_registered_workers()
    if worker_list:
        for w in worker_list:
            st.write(f"✅ {w}")
    else:
        st.caption("No workers registered.")
    
    st.markdown("---")
    target_email = st.text_input("Alert Email", placeholder="manager@site.com")

# --- MENU LOGIC ---

if menu == "📊 Analytics":
    st.header("📊 Violation Analytics")
    conn = sqlite3.connect("safety_violations.db")
    df = pd.read_sql_query("SELECT * FROM violations", conn)
    conn.close()
    if not df.empty:
        st.plotly_chart(px.bar(df, x='worker_name', color='equipment', title="Violations per Worker"))
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No data available.")

elif menu == "👤 Worker Database":
    st.header("👤 Register Worker")
    name = st.text_input("Name")
    wid = st.text_input("ID")
    img_file = st.file_uploader("Face Image", type=['jpg','png'])
    if st.button("Register") and name and wid and img_file:
        path = os.path.join(FACES_DB, f"{name.strip()}_{wid}.jpg")
        Image.open(img_file).convert("RGB").save(path)
        st.success(f"{name} Registered!")
        st.rerun()

elif menu == "🎥 Live Monitoring":
    st.header("🎥 Live Recognition Feed")
    webrtc_streamer(
        key="live",
        video_processor_factory=lambda: VideoProcessor(target_email),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )

elif menu == "📁 Batch Processing":
    st.header("📁 Batch Image Analysis")
    st.write("Multiple images upload karein aur system har kisi ka PPE aur Worker ID check karega.")
    
    uploaded_files = st.file_uploader("Upload Images", type=['jpg','png','jpeg'], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Start Processing"):
            cols = st.columns(2)
            for idx, file in enumerate(uploaded_files):
                # Convert to CV2 format
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, 1)
                
                with st.spinner(f"Processing {file.name}..."):
                    # Detection aur Recognition chalayein
                    processed_img = run_detection(frame, target_email, save_log=True)
                    
                    # Display in grid
                    with cols[idx % 2]:
                        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption=f"Result: {file.name}")
            st.success("Batch Processing Complete!")
