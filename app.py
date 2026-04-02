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
API_URL = "https://shumaschohan-safeguard-ai.hf.space/predict/"
N8N_URL = "https://eom4pk834n2y9tj.m.pipedream.net"
FACES_DB = "worker_faces"

if not os.path.exists(FACES_DB):
    os.makedirs(FACES_DB)
elif os.path.isfile(FACES_DB):
    os.remove(FACES_DB)
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
        
        # 1. Database mein save karein
        conn = sqlite3.connect("safety_violations.db")
        conn.execute('''INSERT INTO violations (timestamp, type, status, equipment, worker_name, worker_id, confidence) 
                        VALUES (?, ?, ?, ?, ?, ?, ?)''',
                     (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), v_type, "⚠️ Unsafe", clean_eq, name, wid, float(v_conf)))
        conn.commit()
        conn.close()
        
        # 2. Email Alert Logic
        # Sirf tab email bhejein jab confidence high ho aur email address provide kiya gaya ho
        if v_conf > 0.75 and user_email and "@" in user_email:
            # Payload jo Pipedream ko jayega
            payload = {
                "worker": name,
                "worker_id": wid,
                "violation": clean_eq,
                "confidence": f"{v_conf:.2f}",
                "time": datetime.now().strftime("%I:%M %p"),
                "email": user_email,
                "subject": f"⚠️ SAFETY ALERT: {clean_eq} Violation detected!"
            }
            
            # Email bhejne ki request
            # Note: Isay 'background' mein hona chahiye taaki app hang na ho
            try:
                requests.post(N8N_URL, json=payload, timeout=2)
                # Success message sirf console/sidebar mein dikhane ke liye
                print(f"Alert sent for {name}")
            except:
                print("Email service unreachable")
                
    except Exception as e:
        print(f"Reporting Error: {e}")

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
                violation_keywords = ["no", "missing", "unsafe", "without", "off", "violation"]
                is_unsafe = any(word in label for word in violation_keywords)
                worker_info = "Unknown_N/A"
                if is_unsafe:
                    face_crop = frame[max(0, y1):y2, max(0, x1):x2]
                    if face_crop.size > 0: worker_info = identify_worker(face_crop)
                    save_to_report(label, conf, True, worker_info, user_email)
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                label_text = f"ALERT: {label}" if is_unsafe else label
                cv2.putText(frame, f"{label_text} ({conf:.2f})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    except: pass
    return frame

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
    menu = st.radio("Navigation", ["📊 Analytics", "👤 Worker Database", "🎥 Live Monitoring", "📁 Batch Processing"])
    target_email = st.text_input("Alert Email", placeholder="user@example.com")

# --- PAGES ---
if menu == "📊 Analytics":
    st.header("📊 Real-Time Safety Dashboard")
    
    # Database se data uthana
    conn = sqlite3.connect("safety_violations.db")
    df = pd.read_sql_query("SELECT * FROM violations", conn)
    conn.close()

    # --- CALCULATIONS FOR TILES ---
    # Note: 'Total Scanned' ko accurately track karne ke liye hum total detections ya logs use karte hain
    total_violations = len(df)
    
    # Dummy logic for 'Total Scanned' (agar aapne scan logs alag nahi banaye toh hum assume karte hain)
    # Behtar FYP ke liye hum isay total frames ya registered workers se relate kar sakte hain
    total_scanned = total_violations + 50  # Sirf example ke liye, aap isay frame count se link kar sakte hain
    
    if total_scanned > 0:
        compliance_rate = ((total_scanned - total_violations) / total_scanned) * 100
    else:
        compliance_rate = 100

    # --- LIVE COUNTER TILES (Top Row) ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="👥 Total Scanned", value=total_scanned, delta="Overall Activity")
    
    with col2:
        # Delta mein hum dikhate hain ke violations barh rahi hain ya kam (Red color for increase)
        st.metric(label="⚠️ Total Violations", value=total_violations, delta=f"{total_violations} Detected", delta_color="inverse")
    
    with col3:
        st.metric(label="✅ Safety Compliance", value=f"{compliance_rate:.1f}%", delta="Target: 95%+")

    st.markdown("---") # Divider line

    # --- GRAPHS SECTION ---
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("📈 Violation Trend")
            # Hourly trend line graph
            trend_df = df.resample('H', on='timestamp').count()['id']
            st.line_chart(trend_df)
            
        with c2:
            st.subheader("🍕 Equipment Distribution")
            fig, ax = plt.subplots(figsize=(5, 5))
            df['equipment'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=['#ff9999','#66b3ff','#99ff99'])
            st.pyplot(fig)

        st.subheader("📝 Recent Violation Logs")
        st.dataframe(df.sort_values(by="timestamp", ascending=False), use_container_width=True)
    else:
        st.info("Abhi tak koi violation data record nahi hua. Live Monitoring start karein!")

elif menu == "👤 Worker Database":
    st.header("👤 Register New Worker")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Worker Name")
        wid = st.text_input("Worker ID")
        
        reg_mode = st.radio("Registration Mode", ["Upload Photo", "Take Photo"])
        img_file = None
        if reg_mode == "Upload Photo":
            img_file = st.file_uploader("Upload Worker Image", type=['jpg', 'png', 'jpeg'])
        else:
            img_file = st.camera_input("Take Photo")

        if st.button("Register Worker") and img_file and name:
            safe_name = name.replace(" ", "_")
            img_path = os.path.join(FACES_DB, f"{safe_name}_{wid}.jpg")
            Image.open(img_file).save(img_path)
            st.success(f"Worker {name} Registered Successfully!")
    with col2:
        st.subheader("Registered Workers List")
        if os.path.exists(FACES_DB):
            for f in os.listdir(FACES_DB):
                if f.endswith(('.jpg', '.png', '.jpeg')):
                    st.write(f"✅ {f.split('.')[0]}")

elif menu == "🎥 Live Monitoring":
    st.header("Live AI Safety Feed")
    webrtc_streamer(
        key="cam", 
        mode=WebRtcMode.SENDRECV, 
        video_processor_factory=lambda: VideoProcessor(target_email),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

elif menu == "📁 Batch Processing":
    st.header("📁 Image Batch Analysis")
    uploaded_files = st.file_uploader("Upload Images for Detection", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            with st.spinner(f"Analyzing {uploaded_file.name}..."):
                processed_frame = run_detection(frame, target_email)
                st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption=f"Result: {uploaded_file.name}", use_container_width=True)
