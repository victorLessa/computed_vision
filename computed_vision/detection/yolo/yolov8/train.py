from ultralytics import YOLO

model=YOLO('yolov8n.yaml').load('yolov8n.pt')

results=model.train(data='./config.yaml', epochs=100, resume=True, iou=0.5, conf=0.001)
# ae1d1a2198005e99d4a9a45aab616fc2f7778322