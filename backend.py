from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np
import io
from PIL import Image
import sqlite3
from datetime import datetime

app = FastAPI()
model = YOLO("model/PEP-DETECTION.pt")

def init_db():
    conn = sqlite3.connect("safety_violations.db")
    cursor = conn.cursor()
    # Ensure all columns exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS violations 
                      (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                       timestamp TEXT, 
                       type TEXT, 
                       equipment TEXT,
                       confidence REAL)''')
    conn.commit()
    conn.close()

init_db()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(img)
    
    results = model(img_np)
    detections = []
    
    conn = sqlite3.connect("safety_violations.db")
    cursor = conn.cursor()

    for r in results:
        for box in r.boxes:
            cls_name = model.names[int(box.cls)]
            conf = float(box.conf)
            detections.append({"class": cls_name, "conf": conf, "bbox": box.xyxy.tolist()})

            # Logic: Agar class name "No-" se shuru ho rahi hai (Violation)
            # Example: "No-Helmet", "No-Vest"
            if "no" in cls_name.lower() or "missing" in cls_name.lower():
                # Extract Equipment Name: "No-Helmet" -> "Helmet"
                equipment = cls_name.lower().replace("no", "").replace("-", "").replace("missing", "").strip().capitalize()
                
                cursor.execute("INSERT INTO violations (timestamp, type, equipment, confidence) VALUES (?, ?, ?, ?)",
                               (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), cls_name, equipment, conf))
    
    conn.commit()
    conn.close()
    return {"detections": detections}