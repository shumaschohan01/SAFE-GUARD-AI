import streamlit as st
import cv2
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import sqlite3
import tempfile
import os
import time
from PIL import Image
from datetime import datetime
from deepface import DeepFace
import requests


# --- DIRECTORY SETUP ---
FACES_DB = "worker_faces"
if not os.path.exists(FACES_DB):
    os.makedirs(FACES_DB)

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect("safety_violations.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS violations 
                      (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                       timestamp TEXT, type TEXT, status TEXT,
                       equipment TEXT, worker_name TEXT,
                       worker_id TEXT, confidence REAL)''')
    conn.commit()
    conn.close()

init_db()

# --- UTILITY FUNCTIONS ---
def is_duplicate_violation(worker_info, v_type):
    if "violation_cache" not in st.session_state:
        st.session_state.violation_cache = {}
    current_time = time.time()
    cache_key = f"{worker_info}_{v_type}"
    if cache_key in st.session_state.violation_cache:
        if current_time - st.session_state.violation_cache[cache_key] < 30:
            return True
    st.session_state.violation_cache[cache_key] = current_time
    return False

def save_to_report(v_type, v_conf, is_unsafe, worker_info):
    if not is_unsafe: return 
    try:
        if is_duplicate_violation(worker_info, v_type): return 
        name, wid = worker_info.split("_") if "_" in worker_info else (worker_info, "N/A")
        clean_eq = v_type.lower().replace("no", "").replace("-", " ").strip().capitalize()
        conn = sqlite3.connect("safety_violations.db")
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO violations (timestamp, type, status, equipment, worker_name, worker_id, confidence) 
                          VALUES (?, ?, ?, ?, ?, ?, ?)''',
                       (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), v_type, "⚠️ Unsafe", 
                        clean_eq, name, wid, float(v_conf)))
        conn.commit(); conn.close()
    except: pass

def identify_worker(face_img):
    try:
        if not os.listdir(FACES_DB): return "Unknown_N/A"
        results = DeepFace.find(img_path=face_img, db_path=FACES_DB, enforce_detection=False, silent=True)
        if len(results) > 0 and not results[0].empty:
            return os.path.basename(results[0].iloc[0]['identity']).split('.')[0]
    except: pass
    return "Unknown_N/A"

def run_detection(frame):
    API_URL = "http://127.0.0.1:8000/predict/"
    try:
        _, img_encoded = cv2.imencode('.jpg', frame)
        response = requests.post(API_URL, files={'file': img_encoded.tobytes()}, timeout=3)
        detections = response.json().get('detections', [])
        for det in detections:
            label, conf = det['class'], det['conf']
            box = det['bbox'][0] if isinstance(det['bbox'][0], list) else det['bbox']
            x1, y1, x2, y2 = map(int, box)
            is_unsafe = any(w in label.lower() for w in ["no", "missing", "unsafe", "without", "off"])
            color = (0, 0, 255) if is_unsafe else (0, 255, 0)
            worker_info = "Unknown_N/A"
            if is_unsafe:
                face_crop = frame[max(0,y1):y2, max(0,x1):x2]
                if face_crop.size > 0: worker_info = identify_worker(face_crop)
            save_to_report(label, conf, is_unsafe, worker_info)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            cv2.putText(frame, f"{worker_info.split('_')[0]}: {label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    except: pass
    return frame

# --- 1. N8N SENDING FUNCTION ---
def send_to_n8n(worker_name, equipment, confidence):
    # n8n se copy kiya hua TEST URL yahan paste karein
    N8N_WEBHOOK_URL = "http://localhost:5678/webhook-test/safety-alert" 
    
    payload = {
        "worker": worker_name,
        "equipment": equipment,
        "confidence": f"{confidence:.2%}",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "🚨 HIGH RISK ALERT"
    }
    
    try:
        # Timeout 2 seconds rakha hai taaki app slow na ho
        response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=2)
        if response.status_code == 200:
            print(f"✅ Alert sent to n8n for {worker_name}")
    except Exception as e:
        print(f"❌ n8n Connection Error: {e}")

# --- 2. UPDATED SAVE TO REPORT ---
def save_to_report(v_type, v_conf, is_unsafe, worker_info):
    if not is_unsafe: 
        return 
        
    try:
        if is_duplicate_violation(worker_info, v_type): 
            return 
        
        # Worker info splitting
        name, wid = worker_info.split("_") if "_" in worker_info else (worker_info, "N/A")
        
        # Cleaning equipment name (e.g., "no-helmet" -> "Helmet")
        clean_eq = v_type.lower().replace("no", "").replace("-", " ").strip().capitalize()
        
        # Current Timestamp
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # --- DATABASE MEIN SAVE ---
        conn = sqlite3.connect("safety_violations.db")
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO violations (timestamp, type, status, equipment, worker_name, worker_id, confidence) 
                          VALUES (?, ?, ?, ?, ?, ?, ?)''',
                       (now_str, v_type, "⚠️ Unsafe", 
                        clean_eq, name, wid, float(v_conf)))
        conn.commit()
        conn.close()

        # --- N8N KO BHEJNA (Automation Trigger) ---
        # Confidence threshold ko aap check kar sakte hain (0.60 = 60% accuracy)
        if float(v_conf) > 0.60: 
            send_to_n8n(name, clean_eq, float(v_conf))
            
    except Exception as e:
        # VS Code ke terminal mein error nazar ayega agar kuch galat hua
        print(f"❗ Error in save_to_report: {e}")
# --- UI SETUP ---
st.set_page_config(page_title="Safe-Guard AI", layout="wide")

# Sidebar Logic
with st.sidebar:
    st.markdown("""
        <div style="text-align: center;">
            <h1 style='color: #FF4B4B; font-size: 28px; margin-bottom: 0;'>🛡️ SAFE-GUARD AI</h1>
            <p style='color: #808495; font-size: 14px;'>Smart Safety Monitoring System</p>
        </div>
        <hr style="margin-top: 5px; margin-bottom: 20px; border-color: #30363d;">
    """, unsafe_allow_html=True)

    menu = st.radio(
        "Main Navigation", 
        ["📊 Analytics",  "👤 Worker Database","🎥 Live Monitoring", "📁 Batch Processing"],
        index=0
    )
    
    st.markdown("---")
    st.info("⚡ **System Status:** Online")
    st.caption("Developed by Shumas Chohan")

# --- PAGE LOGIC ---

if menu == "📊 Analytics":
    st.header("📊 Violation Insights")
    conn = sqlite3.connect("safety_violations.db")
    df = pd.read_sql_query("SELECT * FROM violations", conn)
    conn.close()

    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Violations", len(df))
        m2.metric("Most Missing", df['equipment'].mode()[0])
        m3.metric("Peak Hour", f"{df['hour'].mode()[0]}:00")
        m4.metric("Risk Level", "🔴 High" if len(df) > 15 else "🟡 Moderate")

        st.markdown("---")
        
        st.subheader("🕸️ Equipment Risk Hierarchy")
        fig_sun = px.sunburst(
            df, path=['equipment', 'worker_name'], 
            template="plotly_dark",
            color_discrete_sequence=px.colors.sequential.Reds_r
        )
        st.plotly_chart(fig_sun, use_container_width=True)

        st.subheader("📈 Hourly Trend")
        trend_data = df.groupby('hour').size().reset_index(name='count')
        st.plotly_chart(px.line(trend_data, x='hour', y='count', markers=True, template="plotly_dark"), use_container_width=True)

        st.subheader("📑 Detailed Audit Log")
        st.dataframe(
                 df.sort_values(by='timestamp', ascending=False), 
                 use_container_width=True)
   
    else:
        st.info("No data recorded yet.")

        

elif menu == "🎥 Live Monitoring":
    st.header("🛡️ Live AI Guard")
    if "active" not in st.session_state: st.session_state.active = False
    c1, c2 = st.columns(2)
    if c1.button("🎥 Start Monitoring"): st.session_state.active = True
    if c2.button("🛑 Stop Monitoring"): st.session_state.active = False; st.rerun()
    
    ph = st.empty()
    if st.session_state.active:
        cap = cv2.VideoCapture(0)
        cnt = 0
        while st.session_state.active:
            ret, frame = cap.read()
            if not ret: break
            if cnt % 3 == 0: frame = run_detection(frame)
            ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
            cnt += 1
        cap.release()

elif menu == "👤 Worker Database":
    st.header("👤 Worker Registration")
    c1, c2 = st.columns(2)
    with c1:
        m = st.radio("Method", ["Camera", "File"])
        n, i = st.text_input("Name"), st.text_input("ID")
        img_f = st.camera_input("Capture") if m == "Camera" else st.file_uploader("Upload Image", type=['jpg','png'])
        if st.button("Save Worker") and img_f and n and i:
            Image.open(img_f).convert('RGB').save(os.path.join(FACES_DB, f"{n}_{i}.jpg"))
            st.success(f"Registered {n}!"); st.rerun()
    with c2:
        st.subheader("Current Database")
        for f in [f for f in os.listdir(FACES_DB) if "_" in f]:
            ca, cb = st.columns([3,1])
            ca.write(f"👤 {f}")
            if cb.button("🗑️", key=f): 
                os.remove(os.path.join(FACES_DB, f)); st.rerun()

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