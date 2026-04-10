import streamlit as st
import time
import requests
import numpy as np
import pandas as pd
import sqlite3
import os
import av
import plotly.express as px
import io
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from deepface import DeepFace 
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# --- CONFIGURATION ---
API_URL = "https://shumaschohan-safeguard-ai.hf.space/run/predict" 
N8N_URL = "https://eom4pk834n2y9tj.m.pipedream.net"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_DB = os.path.join(BASE_DIR, "worker_faces")

if not os.path.exists(FACES_DB):
    os.makedirs(FACES_DB, exist_ok=True)

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
def identify_worker(face_img_array):
    try:
        if not os.path.exists(FACES_DB) or not os.listdir(FACES_DB): 
            return "Unknown_N/A"
        
        results = DeepFace.find(
            img_path=face_img_array, 
            db_path=FACES_DB, 
            model_name='ArcFace', 
            distance_metric='cosine',
            enforce_detection=False, 
            silent=True
        )
        
        if len(results) > 0 and not results[0].empty:
            best_match = results[0].iloc[0]
            if best_match['distance'] < 0.55:  
                full_path = best_match['identity']
                return os.path.basename(full_path).split('.')[0]
                
    except Exception as e:
        print(f"Face Recognition Error: {e}")
    return "Unknown_N/A"

def save_to_report(v_type, v_conf, is_unsafe, worker_info, user_email):
    if not is_unsafe: return
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

        if v_conf > 0.75 and user_email and "@" in user_email:
            payload = {"worker": name, "worker_id": wid, "violation": clean_eq, "confidence": f"{v_conf:.2f}", "email": user_email}
            requests.post(N8N_URL, json=payload, timeout=2)
    except Exception as e:
        print(f"DB/Alert Error: {e}")

def run_detection(img_array, user_email):
    try:
        # Convert NumPy array to PIL Image for processing
        pil_img = Image.fromarray(img_array)
        
        # Prepare image for API call without CV2
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()
        
        files = {'file': ('image.jpg', img_bytes, 'image/jpeg')}
        response = requests.post(API_URL, files=files, timeout=5)
        
        if response.status_code == 200:
            draw = ImageDraw.Draw(pil_img)
            detections = response.json().get('detections', [])
            
            for det in detections:
                label, conf = det['class'], det['conf']
                x1, y1, x2, y2 = map(int, det['bbox'])
                is_unsafe = any(w in label.lower() for w in ["no", "missing", "unsafe"])
                
                worker_info = "Unknown_N/A"
                if is_unsafe:
                    # Face crop using PIL
                    face_crop = pil_img.crop((max(0, x1-20), max(0, y1-20), x2+20, y2+20))
                    worker_info = identify_worker(np.array(face_crop))
                    save_to_report(label, conf, True, worker_info, user_email)
                
                color = "red" if is_unsafe else "green"
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                draw.text((x1, y1-15), f"{worker_info.split('_')[0]}: {label}", fill=color)
            
            return np.array(pil_img)
    except Exception as e:
        print(f"Detection Error: {e}")
    return img_array

class VideoProcessor(VideoProcessorBase):
    def __init__(self, email):
        self.email = email
    def recv(self, frame):
        # Streamlit-webrtc gives RGB by default
        img = frame.to_ndarray(format="rgb24")
        img = run_detection(img, self.email)
        return av.VideoFrame.from_ndarray(img, format="rgb24")

# --- UI INTERFACE ---
st.set_page_config(page_title="Safe-Guard AI", layout="wide", page_icon="🛡️")

with st.sidebar:
    st.title("🛡️ SAFE-GUARD AI")
    menu = st.radio("Navigation", ["📊 Analytics", "👤 Worker Database", "🎥 Live Monitoring", "📁 Batch Processing"])
    target_email = st.text_input("Alert Email", placeholder="manager@safety.com").strip()

if menu == "📊 Analytics":
    st.header("📊 Safety Dashboard")
    conn = sqlite3.connect("safety_violations.db")
    df = pd.read_sql_query("SELECT * FROM violations", conn)
    conn.close()

    total_v = len(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("⚠️ Total Violations", total_v)
    c2.metric("✅ Compliance Rate", f"{max(0, 100-total_v)}%")
    c3.metric("📅 Active Site", "Phase 1")

    if not df.empty:
        col_pie, col_bar = st.columns(2)
        with col_pie:
            fig_pie = px.pie(df, names='worker_name', hole=0.4, title="Violations by Personnel", template="plotly_dark")
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_bar:
            # Fixed bar chart syntax
            counts = df['equipment'].value_counts().reset_index()
            counts.columns = ['Equipment', 'Count']
            fig_bar = px.bar(counts, x='Equipment', y='Count', title="Equipment Issues", template="plotly_dark")
            st.plotly_chart(fig_bar, use_container_width=True)
        st.dataframe(df.sort_values(by="timestamp", ascending=False), use_container_width=True)
    else:
        st.info("No data recorded yet.")

elif menu == "👤 Worker Database":
    st.header("👤 Personnel Registration")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.form("registration_form"):
            name = st.text_input("Full Name")
            wid = st.text_input("Worker ID")
            img_file = st.file_uploader("Upload Face Image", type=['jpg', 'png'])
            submit = st.form_submit_button("Register Worker")
            
            if submit and name and wid and img_file:
                path = os.path.join(FACES_DB, f"{name.replace(' ', '_')}_{wid}.jpg")
                Image.open(img_file).convert("RGB").save(path)
                for f in os.listdir(FACES_DB):
                    if f.endswith(".pkl"): os.remove(os.path.join(FACES_DB, f))
                st.success(f"Registered {name}")

    with col2:
        st.subheader("Current Database")
        if os.path.exists(FACES_DB):
            for f in os.listdir(FACES_DB):
                if f.lower().endswith(('.jpg', '.png')):
                    st.text(f"✅ {f.split('.')[0]}")

elif menu == "🎥 Live Monitoring":
    st.header("🎥 Active Monitoring Feed")
    webrtc_streamer(
        key="safety-stream",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: VideoProcessor(target_email),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

elif menu == "📁 Batch Processing":
    st.header("📁 Image Batch Analysis")
    uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True)
    if uploaded_files:
        for uf in uploaded_files:
            img = Image.open(uf).convert("RGB")
            processed_array = run_detection(np.array(img), target_email)
            st.image(processed_array, caption=uf.name)
