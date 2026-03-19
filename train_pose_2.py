from ultralytics import YOLO

# load YOLO pose model
model = YOLO("yolov8n-pose.pt")

# train
model.train(
    data=r"C:\Users\hp\OneDrive\Desktop\gesture_data\final_dataset_2\data.yaml",
    epochs=150,
    imgsz=640,
    batch=8,
    device="cpu",   # change to 0 if you have NVIDIA GPU
    workers=2,
    fliplr=0.0,
    mosaic=0.0,
    project=r"C:\Users\hp\OneDrive\Desktop\gesture_data\runs_pose_2",
    name="gesture_pose"
)