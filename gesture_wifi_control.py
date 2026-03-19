from ultralytics import YOLO
import cv2
import requests
from collections import deque
import time

# ==========================
# SETTINGS
# ==========================
MODEL_PATH = r"C:\Users\hp\OneDrive\Desktop\gesture_data\runs_pose_2\gesture_pose\weights\best.pt"
ESP32_IP = "10.199.12.109"   # CHANGE THIS
CONFIDENCE_THRESHOLD = 0.60
STABLE_FRAMES = 5
SEND_INTERVAL = 0.5  # seconds between repeated sends

# Gesture to endpoint mapping
COMMAND_MAP = {
    "forward": "forward",
    "left": "left",
    "right": "right",
    "back": "back"
}

# ==========================
# LOAD MODEL
# ==========================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

history = deque(maxlen=STABLE_FRAMES)
last_sent_command = None
last_send_time = 0


def send_command(command: str):
    global last_sent_command, last_send_time

    current_time = time.time()

    # avoid spamming same command too fast
    if command == last_sent_command and (current_time - last_send_time) < SEND_INTERVAL:
        return

    url = f"http://{ESP32_IP}/{command}"

    try:
        response = requests.get(url, timeout=0.3)
        print(f"Sent: {command}, Response: {response.text}")
        last_sent_command = command
        last_send_time = current_time
    except requests.exceptions.RequestException:
        print(f"Failed to send command: {command}")


stable_gesture = "stop"
send_command("stop")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, imgsz=640, conf=CONFIDENCE_THRESHOLD, verbose=False)
    annotated = results[0].plot()

    detected_gesture = None
    detected_conf = 0.0

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        cls_id = int(results[0].boxes.cls[0].item())
        detected_conf = float(results[0].boxes.conf[0].item())
        detected_gesture = model.names[cls_id]
        history.append(detected_gesture)

        # stable decision
        if history.count(detected_gesture) >= STABLE_FRAMES:
            stable_gesture = detected_gesture

    else:
        history.clear()
        stable_gesture = "stop"

    # send command if stable gesture exists in command map
    if stable_gesture in COMMAND_MAP:
        send_command(COMMAND_MAP[stable_gesture])

    # display text
    cv2.putText(
        annotated,
        f"Detected: {detected_gesture if detected_gesture else 'None'}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated,
        f"Stable: {stable_gesture}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.putText(
        annotated,
        "Press Q to quit",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.imshow("Gesture WiFi Robot Control", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        send_command("stop")
        break

cap.release()
cv2.destroyAllWindows()