from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

# Load YOLO model
model = YOLO("yolov8n.pt")

@app.get("/")
def root():
    return {"message": "YOLOv8 API is running!"}

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    # Read image
    image = await file.read()
    npimg = np.frombuffer(image, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Run detection
    results = model(frame)
    boxes = results[0].boxes.xyxy.tolist()  # bounding boxes
    classes = results[0].boxes.cls.tolist()  # class ids
    scores = results[0].boxes.conf.tolist()  # confidence

    response = []
    for box, cls, score in zip(boxes, classes, scores):
        response.append({
            "box": box,
            "class": model.names[int(cls)],
            "confidence": float(score)
        })

    return JSONResponse(content={"detections": response})