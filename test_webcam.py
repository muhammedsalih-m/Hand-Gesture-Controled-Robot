from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\hp\OneDrive\Desktop\gesture_data\runs_pose_2\gesture_pose\weights\best.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.5, imgsz=640, verbose=False)
    annotated = results[0].plot()

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        cls_id = int(results[0].boxes.cls[0].item())
        conf = float(results[0].boxes.conf[0].item())
        class_name = model.names[cls_id]

        cv2.putText(
            annotated,
            f"{class_name} {conf:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Gesture Test", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()