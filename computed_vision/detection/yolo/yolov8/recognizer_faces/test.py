import copy
import matplotlib.pyplot as plt
import cv2
import os
from ultralytics import YOLO

img = cv2.imread('./walking-6830897_960_720.jpg')

model = YOLO("../face_yolov8.pt")

image_copy=copy.deepcopy(img)

results = model(image_copy)

for r in results:
  annotated_frame = r.plot()
  r.save(filename="result.jpg")
