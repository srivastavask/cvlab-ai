import cv2
import matplotlib.pyplot as plt
import numpy as np

imgpath = "C:/Users/Yash Ganodiya/OneDrive/Pictures/Screenshots/Screenshot (144).png"
image = cv2.imread(imgpath)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Original Image")
plt.show()

height, width, channels = image.shape
print(f"Image Dimensions: {width}x{height}")
print(f"Number of Channels: {channels}")

total_pixels = height * width
print(f"Total Pixels in Image: {total_pixels}")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(gray_image, cmap="gray")
plt.axis("off")
plt.title("Grayscale Image")
plt.show()

threshold_value = 128  
_, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

plt.imshow(binary_image, cmap="gray")
plt.axis("off")
plt.title("Binary Image")
plt.show()

height, width = gray_image.shape
image_size = height * width
print(f"Image Size (Total Pixels): {image_size}")

black_pixel_count = np.sum(binary_image == 0)
print(f"Black Pixel Area: {black_pixel_count}")

#sobel
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
sobel_x_abs = cv2.convertScaleAbs(sobel_x)
sobel_y_abs = cv2.convertScaleAbs(sobel_y)
sobel_combined = cv2.addWeighted(sobel_x_abs, 0.5, sobel_y_abs, 0.5, 0)

plt.imshow(sobel_combined, cmap='gray')
plt.axis('off')
plt.title('Sobel Edge Detection')
plt.show()

#Prewitt
prewitt_x_kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_y_kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
prewitt_x = cv2.filter2D(gray_image, cv2.CV_64F, prewitt_x_kernel)
prewitt_y = cv2.filter2D(gray_image, cv2.CV_64F, prewitt_y_kernel)
prewitt_x_abs = cv2.convertScaleAbs(prewitt_x)
prewitt_y_abs = cv2.convertScaleAbs(prewitt_y)
prewitt_combined = cv2.addWeighted(prewitt_x_abs, 0.5, prewitt_y_abs, 0.5, 0)

plt.imshow(prewitt_combined, cmap='gray')
plt.axis('off')
plt.title('Prewitt Edge Detection')
plt.show()

#Roberts Cross Operator
roberts_cross_x_kernel = np.array([[1, 0], [0, -1]])
roberts_cross_y_kernel = np.array([[0, 1], [-1, 0]])
roberts_x = cv2.filter2D(gray_image, cv2.CV_64F, roberts_cross_x_kernel)
roberts_y = cv2.filter2D(gray_image, cv2.CV_64F, roberts_cross_y_kernel)
roberts_x_abs = cv2.convertScaleAbs(roberts_x)
roberts_y_abs = cv2.convertScaleAbs(roberts_y)
roberts_combined = cv2.addWeighted(roberts_x_abs, 0.5, roberts_y_abs, 0.5, 0)

plt.imshow(roberts_combined, cmap='gray')
plt.axis('off')
plt.title('Roberts Cross Edge Detection')
plt.show()

#Canny Edge Detector
canny_edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)

plt.imshow(canny_edges, cmap='gray')
plt.axis('off')
plt.title('Canny Edge Detection')
plt.show()

#Global
_, global_thresh = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

#Adaptive
adaptive_thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)

#Canny
#canny_edges = cv2.Canny(gray_image, 100, 200)

#Watershed
_, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Noise removal (Morphological operations)
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# Sure background (dilation)
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Sure foreground (distance transform)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labeling
_, markers = cv2.connectedComponents(sure_fg)

# Add 1 to all labels so background is 1, not 0
markers = markers + 1
markers[unknown == 255] = 0

# Apply watershed algorithm
markers = cv2.watershed(image, markers)
image[markers == -1] = [0, 0, 255]  # Mark boundaries in red

# ========== Display Results ==========
plt.figure(figsize=(12, 8))

plt.subplot(231), plt.imshow(gray_image, cmap="gray"), plt.title("Grayscale Image"), plt.axis("off")
plt.subplot(232), plt.imshow(global_thresh, cmap="gray"), plt.title("Global Thresholding"), plt.axis("off")
plt.subplot(233), plt.imshow(adaptive_thresh, cmap="gray"), plt.title("Adaptive Thresholding"), plt.axis("off")
plt.subplot(234), plt.imshow(canny_edges, cmap="gray"), plt.title("Canny Edge Detection"), plt.axis("off")
plt.subplot(235), plt.imshow(markers, cmap="jet"), plt.title("Watershed Segmentation"), plt.axis("off")

plt.tight_layout()
plt.show()












