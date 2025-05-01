import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./images/watch.jpg', cv2.IMREAD_GRAYSCALE)
# 1 way of doing
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 2 way of doing
plt.imshow(img, cmap='gray', interpolation='bicubic')
# plt.plot([50,100], [80,100], 'r', linewidth=5)
plt.show()

