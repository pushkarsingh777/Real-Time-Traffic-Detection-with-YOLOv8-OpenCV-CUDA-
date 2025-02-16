from ultralytics import YOLO
import torch
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = YOLO("yolov8n.pt").to(device)

cap = cv2.VideoCapture("traffic.mp4")

frame_width = 640
frame_height = 480

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    frame = cv2.resize(frame, (frame_width, frame_height))

    results = model(frame, device=device)  
    vehicle_count = len(results[0].boxes)

    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Traffic Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
