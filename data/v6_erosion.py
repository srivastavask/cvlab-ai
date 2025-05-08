import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([20,30,0])
    upper_red = np.array([180,255,150])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    
    kernel = np.ones((5,5), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=1)
    dilation = cv2.dilate(mask, kernel, iterations=1)
    
    cv2.imshow('frame', frame)
    cv2.imshow('res', res)
    cv2.imshow('erode', erosion)
    cv2.imshow('dilation', dilation)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()