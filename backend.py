from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import io
from PIL import Image
import sqlite3
from datetime import datetime

app = FastAPI()

# CORS allow karna zaroori hai web deployment ke liye
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("PEP-DETECTION.pt")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(img)
    
    results = model(img_np)
    detections = []
    
    for r in results:
        for box in r.boxes:
            cls_name = model.names[int(box.cls)]
            conf = float(box.conf)
            detections.append({
                "class": cls_name, 
                "conf": conf, 
                "bbox": box.xyxy.tolist()[0]
            })
    
    return {"detections": detections}
