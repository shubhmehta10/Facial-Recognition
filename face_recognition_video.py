import cv2 
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    _,img = cap.read()
    img =cv2.flip(img,1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    f = face.detectMultiScale(gray,scaleFactor = 1.2,minNeighbors = 2)

    for x,y,w,h in f:
        img = cv2.rectangle(img,(x,y),(x+h,y+w),(0,255,0),2)
    cv2.imshow('Face Recognized',img)

    if cv2.waitKey(1) == 13:
        break

cap.release()    
cv2.destroyAllWindows()
