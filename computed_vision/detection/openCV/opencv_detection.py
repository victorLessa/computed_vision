import cv2 as cv
import numpy as np
import time


faceCascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

cap = cv.VideoCapture(0)
cap.set(3, 1024)
cap.set(4, 768)

face_id = input('\n enter user name end press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")


count = 0
while True:
  ret, img = cap.read()
  # img = cv.flip(img, -1)
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  faces = faceCascade.detectMultiScale(
      gray,
      scaleFactor=1.2,
      minNeighbors=5,     
      minSize=(40, 40)
  )
  for (x,y,w,h) in faces:
    count += 1
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    cv.imwrite("./dataset/User." + str(count) + "." + str(face_id) + ".jpg" , gray[y:y+h,x:x+w])
  cv.imshow('video',img)
  if cv.waitKey(1) == ord('q'):
    break

cap.release()
cv.destroyAllWindows()