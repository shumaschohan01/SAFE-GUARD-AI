import streamlit as st
import cv2
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import sqlite3
import os
import time
from PIL import Image
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# --- CONFIGURATION ---
# Deployment ke waqt ye URLs badal kar public URLs (e.g. Render/Railway) kar dein
API_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000/predict/")
N8N_URL = os.getenv("N8N_URL", "http://localhost:5678/webhook/safety-alert")

FACES_DB = "worker_faces"
if not os.path.exists(FACES_DB): os.makedirs(FACES_DB)

# --- DATABASE & N8N LOGIC ---
def send_to_n8n(worker_name, equipment, confidence):
    payload = {
        "worker": worker_name, "equipment": equipment,
        "confidence": f"{confidence:.2%}",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    try: requests.post(N8N_URL, json=payload, timeout=2)
    except: pass

def save_to_report(v_type, v_conf, is_unsafe, worker_info):
    if not is_unsafe: return 
    try:
        name, wid = worker_info.split("_") if "_" in worker_info else (worker_info, "N/A")
        clean_eq = v_type.lower().replace("no", "").replace("-", "").strip().capitalize()
        conn = sqlite3.connect("safety_violations.db")
        conn.execute('''INSERT INTO violations (timestamp, type, status, equipment, worker_name, worker_id, confidence) 
                        VALUES (?, ?, ?, ?, ?, ?, ?)''',
                     (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), v_type, "⚠️ Unsafe", clean_eq, name, wid, float(v_conf)))
        conn.commit(); conn.close()
        if float(v_conf) > 0.60: send_to_n8n(name, clean_eq, float(v_conf))
    except: pass

# --- WEBRTC VIDEO PROCESSOR ---
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Har 5th frame process karein taaki speed bani rahe
        processed_img = run_detection(img) 
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# --- UI SETUP ---
st.set_page_config(page_title="Safe-Guard AI", layout="wide")

with st.sidebar:
    st.title("🛡️ SAFE-GUARD AI")
    menu = st.radio("Navigation", ["📊 Analytics", "👤 Worker Database", "🎥 Live Monitoring", "📁 Batch"])

if menu == "🎥 Live Monitoring":
    st.header("Live Web Monitoring")
    # Ye component browser ka camera use karega
    webrtc_streamer(
        key="safety-check",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
elif menu == "📁 Batch Processing":
    st.header("📁 Media Analysis")
    f = st.file_uploader("Upload Image or Video", type=['jpg','png','mp4'])
    if f:
        if f.type.startswith('image'):
            st.image(cv2.cvtColor(run_detection(cv2.cvtColor(np.array(Image.open(f)), cv2.COLOR_RGB2BGR)), cv2.COLOR_BGR2RGB), use_column_width=True)
        else:
            t = tempfile.NamedTemporaryFile(delete=False); t.write(f.read())
            vf = cv2.VideoCapture(t.name); sf = st.empty()
            while vf.isOpened():
                r, fr = vf.read()
                if not r: break
                sf.image(cv2.cvtColor(run_detection(fr), cv2.COLOR_BGR2RGB), use_column_width=True)
            vf.release()
