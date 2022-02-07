import cv2
import numpy as np
cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
i =0
while(1):
    # get a frame
    ret, frame = cap.read()
    ret2,frame2 = cap2.read()
    if ret!=None and ret2!=None:
        assert frame.shape == frame2.shape
    # show a frameqqq
    if ret!=None:
        # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame,dsize=(600,400))
        cv2.imshow("Right", frame)
        
    if ret2!=None:
        # frame2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        frame2 = cv2.resize(frame2,dsize=(600,400))
        cv2.imshow("Left",frame2)
    

    k=cv2.waitKey(1)
    if k==27:
        break
    elif k==ord('s'):
        cv2.imwrite('right/'+str(i)+'.jpg',frame)
        cv2.imwrite('left/'+str(i)+'.jpg',frame2)
        i+=1
cap.release()
cv2.destroyAllWindows() 
