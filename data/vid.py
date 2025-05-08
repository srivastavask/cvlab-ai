# reading mp4 file and performing simple operations
import cv2
vid = cv2.VideoCapture('triveni.mp4')
# print("video",vid)
while True:
    time, frame = vid.read()
    frame = cv2.resize(frame, (600,400))
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame", frame)
    cv2.imshow("gray", gray)
    k = cv2.waitKey(3)
    if k == ord("q"):
        break
vid.release()   # release
cv2.destroyAllWindows()


