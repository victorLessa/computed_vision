
import cv2

from ultralytics import YOLO

font = cv2.FONT_HERSHEY_SIMPLEX
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('./recognizer.yml')
# Load the YOLOv8 model
model = YOLO("best.pt")

# Open the video file
cap = cv2.VideoCapture(0)
W = 640
H = 480
cap.set(3, W)
cap.set(4, H)

names=["Victor"]

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # classes = results.names[results.pred[0].cpu().numpy()]
        for r in results:
          # print(classes)
          boxes = r.boxes.cpu().numpy()
          # print(boxes)
          for box in boxes:
            for x1, y1, x2,y2 in box.xyxy.astype(int): 
              
              # print(x1, y1, x2,y2)
            
              id, confidence = recognizer.predict(gray[y1:y2,x1:x2])
              print(id, confidence)
              if (confidence < 50):
                id = names[id]
                print(id)
                confidence = "  {0}%".format(round(100 - confidence))
              else:
                id = "Unrecognized"
                confidence = "  {0}%".format(round(100 - confidence))
              # cv2.imwrite("./User.jpg", gray[y1:y2,x1:x2]) 
              cv2.putText(frame, str(id), (x2+5,y2-5), 
                font,
                1, 
                (255,255,0), 
                1
                )  
          # Visualize the results on the frame
          annotated_frame = r.plot()
          # print(annotated_frame)
          # Display the annotated frame
          
          cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()