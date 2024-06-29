import cv2
import numpy as np
import os 
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('./trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
#iniciate id counter
id = 0
# names related to ids: example ==> Marcelo: id=1,  etc
names = ['victor'] 
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 1080) # set video widht
cam.set(4, 960) # set video height
# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
faceCount = 0
while True:
    ret, img =cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (40,40),
       )
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
        # If confidence is less them 100 ==> "0" : perfect match
        print(confidence)
        if (confidence < 50):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "Unrecognized"
            confidence = "  {0}%".format(round(100 - confidence))
            faceCount += 1
            print('salvando')
            cv2.imwrite("./danger/User."+ str(faceCount) + '.jpg', gray[y:y+h,x:x+w]) 

        cv2.putText(
                    img, 
                    str(id), 
                    (x+5,y-5), 
                    font, 
                    1, 
                    (255,255,255), 
                    2
                   )
        cv2.putText(
                    img, 
                    str(confidence), 
                    (x+5,y+h-5), 
                    font, 
                    1, 
                    (255,255,0), 
                    1
                   )  
    frame = cv2.resize(img, (960, 1080))
    cv2.imshow('camera',img) 
    if cv2.waitKey(1) == ord('q'):
      break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()