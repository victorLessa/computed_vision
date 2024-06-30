
import cv2
import os
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("../face_yolov8.pt")


cap = cv2.VideoCapture(0)
W = 1024
H = 968
cap.set(3, W)
cap.set(4, H)

face_id = input('\n enter user name end press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")

dataset_path = './dataset'

count = 188
while True:
  ret, img = cap.read()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
  results = model(img)

  count+=1
  for r in results:
    # print(classes)
    boxes = r.boxes.cpu().numpy()
    # print(boxes)
    for box in boxes:
      for x1, y1, x2,y2 in box.xyxy.astype(int): 
        # print(x1, y1, x2,y2)
        # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        if not os.path.isdir(dataset_path):
          os.makedirs(dataset_path, exist_ok=True)
        cv2.imwrite(os.path.join(dataset_path, "User." + str(count) + "." + str(face_id) + ".jpg"), gray[y1:y2,x1:x2])
  
    # Visualize the results on the frame
    annotated_frame = r.plot()
    # print(annotated_frame)
    # Display the annotated frame
    
    cv2.imshow("YOLOv8 Inference", annotated_frame)

  # cv2.imshow('video',img)
  if cv2.waitKey(1) == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
