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

# --- 1. CONFIGURATION & DIRECTORY SETUP ---
API_URL = "https://shumaschohan-safeguard-ai.hf.space/predict/"
N8N_URL = "https://eom4pk834n2y9tj.m.pipedream.net"
FACES_DB = "worker_faces"

# Ensure FACES_DB is a directory
if os.path.exists(FACES_DB):
    if not os.path.isdir(FACES_DB):
        os.remove(FACES_DB)
        os.makedirs(FACES_DB, exist_ok=True)
else:
    os.makedirs(FACES_DB, exist_ok=True)

# --- 2. DATABASE INITIALIZATION ---
def init_db():
    conn = sqlite3.connect("safety_violations.db", check_same_thread=False)
    conn.execute('''CREATE TABLE IF NOT EXISTS violations 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, type TEXT, 
                     status TEXT, equipment TEXT, worker_name TEXT, worker_id TEXT, confidence REAL)''')
    conn.commit()
    conn.close()

init_db()

# --- 3. HELPER FUNCTIONS ---

def get_registered_workers():
    """Returns a unique list of registered worker names from folder."""
    try:
        files = [f for f in os.listdir(FACES_DB) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        return sorted(list(set([f.split('_')[0] for f in files])))
    except Exception:
        return []

def identify_worker(face_img):
    """DeepFace matching with Facenet & MTCNN for high accuracy."""
    try:
        from deepface import DeepFace
        
        # Save temp crop for analysis
        temp_path = "current_face_scan.jpg"
        cv2.imwrite(temp_path, face_img)
        
        # Clear DeepFace cache to ensure new registrations are picked up
        pkl_path = os.path.join(FACES_DB, "representations_facenet.pkl")
        if os.path.exists(pkl_path):
            os.remove(pkl_path)

        # Facenet is more robust for side-faces and low light
        results = DeepFace.find(
            img_path=temp_path, 
            db_path=FACES_DB, 
            model_name='Facenet', 
            distance_metric='cosine',
            enforce_detection=False, 
            detector_backend='mtcnn', # Best for finding faces in complex backgrounds
            silent=True
        )

        if len(results) > 0 and not results[0].empty:
            best_match = results[0].iloc[0]
            # 0.50 - 0.60 is the sweet spot for Facenet cosine distance
            if best_match['distance'] < 0.55: 
                identity_path = best_match['identity']
                return os.path.basename(identity_path).split('_')[0]
                
    except Exception as e:
        print(f"Recognition Error: {e}")
    return "Unknown"

def save_violation(v_type, conf, worker_name, user_email):
    """Saves violation to SQLite and sends alert."""
    try:
        clean_eq = v_type.lower().replace("no", "").replace("-", "").strip().capitalize()
        conn = sqlite3.connect("safety_violations.db")
        conn.execute('''INSERT INTO violations (timestamp, type, status, equipment, worker_name, worker_id, confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?)''', 
                     (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), v_type, "⚠️ Unsafe", clean_eq, worker_name, "N/A", float(conf)))
        conn.commit()
        conn.close()

        # Webhook Alert
        if conf > 0.70 and user_email and "@" in user_email:
            payload = {"worker": worker_name, "violation": clean_eq, "time": datetime.now().strftime("%I:%M %p"), "email": user_email}
            requests.post(N8N_URL, json=payload, timeout=2)
    except Exception as e:
        print(f"Log Error: {e}")

def run_detection(frame, user_email, save_log=True):
    """Main Detection Logic for both Live and Batch."""
    try:
        _, img_encoded = cv2.imencode('.jpg', frame)
        response = requests.post(API_URL, files={'file': img_encoded.tobytes()}, timeout=10)
        
        if response.status_code == 200:
            detections = response.json().get('detections', [])
            for det in detections:
                label = str(det.get('class', '')).lower()
                conf = float(det.get('conf', 0))
                if conf < 0.40: continue
                
                x1, y1, x2, y2 = map(int, det.get('bbox', [0,0,0,0]))
                is_unsafe = any(word in label for word in ["no", "missing", "unsafe"])
                
                name_tag = "Scanning..."
                color = (0, 255, 0) # Green for safe

                if is_unsafe:
                    color = (0, 0, 255) # Red for danger
                    # Expand crop to ensure face is captured
                    y_top = max(0, y1 - 50)
                    y_bottom = min(frame.shape[0], y1 + int((y2 - y1) * 0.6))
                    face_crop = frame[y_top:y_bottom, max(0, x1-20):min(frame.shape[1], x2+20)]
                    
                    if face_crop.size > 0:
                        name_tag = identify_worker(face_crop)
                    
                    if save_log and name_tag != "Scanning...":
                        save_violation(label, conf, name_tag, user_email)

                # Draw UI
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f"{name_tag}: {label}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    except Exception as e:
        print(f"Detection Error: {e}")
    return frame

# --- 4. VIDEO STREAMING CLASS ---
class VideoProcessor(VideoProcessorBase):
    def __init__(self, email):
        self.email = email
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = run_detection(img, self.email)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 5. STREAMLIT UI ---
st.set_page_config(page_title="Safe-Guard AI | PPE & Face ID", layout="wide")

with st.sidebar:
    st.title("🛡️ SAFE-GUARD AI")
    menu = st.radio("Navigation", ["📊 Analytics", "👤 Worker Database", "🎥 Live Monitoring", "📁 Batch Processing"])
    
    st.markdown("---")
    st.subheader("📋 Registered Workers")
    current_workers = get_registered_workers()
    if current_workers:
        for w in current_workers:
            st.write(f"✅ {w}")
    else:
        st.caption("No workers in database.")
    
    st.markdown("---")
    target_email = st.text_input("Alert Email", placeholder="manager@site.com")

# --- 6. PAGE LOGIC ---

if menu == "📊 Analytics":
    st.header("📊 Safety Analytics Dashboard")
    conn = sqlite3.connect("safety_violations.db")
    df = pd.read_sql_query("SELECT * FROM violations", conn)
    conn.close()
    
    if not df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.pie(df, names='worker_name', title="Violations by Worker", hole=0.4), use_container_width=True)
        with col2:
            st.plotly_chart(px.bar(df, x='equipment', title="Violation Type Count"), use_container_width=True)
        st.subheader("Detailed Logs")
        st.dataframe(df.sort_values(by="timestamp", ascending=False), use_container_width=True)
    else:
        st.info("Database is empty. Start monitoring to collect data.")

elif menu == "👤 Worker Database":
    st.header("👤 Worker Registration")
    c1, c2 = st.columns(2)
    with c1:
        new_name = st.text_input("Full Name (e.g., Ali)")
        new_id = st.text_input("Worker ID (e.g., 101)")
        reg_source = st.radio("Photo Source", ["Upload File", "Take Picture"])
        img_input = st.file_uploader("Upload Image", type=['jpg','png']) if reg_source == "Upload File" else st.camera_input("Capture")

    if st.button("Register Worker"):
        if new_name and new_id and img_input:
            clean_name = new_name.strip().replace(" ", "_")
            save_path = os.path.join(FACES_DB, f"{clean_name}_{new_id}.jpg")
            Image.open(img_input).convert("RGB").save(save_path)
            
            # Reset Cache
            for f in os.listdir(FACES_DB):
                if f.endswith(".pkl"): os.remove(os.path.join(FACES_DB, f))
                
            st.success(f"Registered {new_name} successfully!")
            st.rerun()
        else:
            st.error("Please provide all details.")

elif menu == "🎥 Live Monitoring":
    st.header("🎥 Real-Time PPE & Face Recognition")
    if not current_workers:
        st.warning("⚠️ Database empty! Recognition Unknown dikhayega.")
        
    webrtc_streamer(
        key="safety-live",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: VideoProcessor(target_email),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

elif menu == "📁 Batch Processing":
    st.header("📁 Batch Image Analysis")
    uploaded_images = st.file_uploader("Upload multiple images", type=['jpg','png','jpeg'], accept_multiple_files=True)
    
    if uploaded_images:
        if st.button("Analyze All"):
            grid = st.columns(2)
            for i, img_file in enumerate(uploaded_images):
                file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
                opencv_img = cv2.imdecode(file_bytes, 1)
                
                with st.spinner(f"Analyzing {img_file.name}..."):
                    result_img = run_detection(opencv_img, target_email, save_log=True)
                    with grid[i % 2]:
                        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption=f"Processed: {img_file.name}")
            st.success("Batch Analysis Complete!")
