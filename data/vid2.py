import cv2
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*"xvid") # xvid, divx, MJPG
output = cv2.VideoWriter("output.avi",fourcc, 20.0,(600,600))
# print("cap", cap)
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        # frame = cv2.resize(frame, (600,400))
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #cv2.imshow("frame", frame)
        cv2.imshow("gray", gray)
        output.write(gray)   # writing frame in to output.avi
        k = cv2.waitKey(30)
        if k == ord("q"):
             break
cap.release()
output.release()   # important step so that the container should be released
cv2.destroyAllWindows()


