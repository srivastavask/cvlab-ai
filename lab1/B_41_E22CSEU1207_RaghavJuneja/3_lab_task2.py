import cv2
import numpy as np
from skimage.filters import sobel, prewitt, roberts

class ImageAnalyzer:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None
        self.gray_image = None

    def read_image(self):
        """Reads the image from the specified path."""
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image not found at {self.image_path}")
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        print("Image read and converted to grayscale.")

    def display_image(self, window_name, image):
        """Displays the given image in a window."""
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def sobel_operator(self):
        """Applies the Sobel operator for edge detection."""
        sobel_edges = sobel(self.gray_image)
        sobel_edges = (sobel_edges * 255).astype(np.uint8)
        cv2.imwrite("sobel_edges.jpg", sobel_edges)
        print("Sobel edges saved as 'sobel_edges.jpg'.")
        return sobel_edges

    def prewitt_operator(self):
        """Applies the Prewitt operator for edge detection."""
        prewitt_edges = prewitt(self.gray_image)
        prewitt_edges = (prewitt_edges * 255).astype(np.uint8)
        cv2.imwrite("prewitt_edges.jpg", prewitt_edges)
        print("Prewitt edges saved as 'prewitt_edges.jpg'.")
        return prewitt_edges

    def roberts_operator(self):
        """Applies the Roberts Cross operator for edge detection."""
        roberts_edges = roberts(self.gray_image)
        roberts_edges = (roberts_edges * 255).astype(np.uint8)
        cv2.imwrite("roberts_edges.jpg", roberts_edges)
        print("Roberts edges saved as 'roberts_edges.jpg'.")
        return roberts_edges

    def canny_edge_detection(self):
        """Applies the Canny edge detector."""
        canny_edges = cv2.Canny(self.gray_image, 100, 200)
        cv2.imwrite("canny_edges.jpg", canny_edges)
        print("Canny edges saved as 'canny_edges.jpg'.")
        return canny_edges

    def global_thresholding(self):
        """Applies global thresholding for segmentation."""
        _, thresh_image = cv2.threshold(self.gray_image, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite("global_threshold.jpg", thresh_image)
        print("Global thresholding image saved as 'global_threshold.jpg'.")
        return thresh_image

    def adaptive_thresholding(self):
        """Applies adaptive thresholding for segmentation."""
        adaptive_thresh = cv2.adaptiveThreshold(
            self.gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        cv2.imwrite("adaptive_threshold.jpg", adaptive_thresh)
        print("Adaptive thresholding image saved as 'adaptive_threshold.jpg'.")
        return adaptive_thresh

    def edge_detection_segmentation(self):
        """Uses Canny edge detection for segmentation."""
        return self.canny_edge_detection()

    def watershed_segmentation(self):
        """Applies the Watershed algorithm for region-based segmentation."""
        ret, thresh = cv2.threshold(self.gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(thresh, kernel, iterations=2)
        dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(self.image, markers)
        self.image[markers == -1] = [255, 0, 0]
        cv2.imwrite("watershed_segmentation.jpg", self.image)
        print("Watershed segmentation result saved as 'watershed_segmentation.jpg'.")
        return self.image

if __name__ == "__main__":

    image_path = "images/img.png"  
    analyzer = ImageAnalyzer(image_path)


    analyzer.read_image()


    analyzer.display_image("Sobel Edges", analyzer.sobel_operator())
    analyzer.display_image("Prewitt Edges", analyzer.prewitt_operator())
    analyzer.display_image("Roberts Edges", analyzer.roberts_operator())
    analyzer.display_image("Canny Edges", analyzer.canny_edge_detection())


    analyzer.display_image("Global Thresholding", analyzer.global_thresholding())
    analyzer.display_image("Adaptive Thresholding", analyzer.adaptive_thresholding())
    analyzer.display_image("Edge Detection Segmentation", analyzer.edge_detection_segmentation())
    analyzer.display_image("Watershed Segmentation", analyzer.watershed_segmentation())
