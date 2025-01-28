import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None
        self.gray_image = None
        self.binary_image = None

    def read_image(self):
        """Reads the image from the specified path."""
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image not found at {self.image_path}")
        print("Image read successfully.")

    def display_image(self, window_name, image):
        """Displays the given image in a window."""
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def extract_image_size(self):
        """Extracts and returns the size of the image."""
        if self.image is None:
            raise ValueError("Image not loaded. Call read_image() first.")
        height, width, channels = self.image.shape
        print(f"Image Size: Height={height}, Width={width}, Channels={channels}")
        return height, width, channels

    def calculate_image_pixels(self):
        """Calculates and returns the total number of pixels in the image."""
        height, width, _ = self.extract_image_size()
        total_pixels = height * width
        print(f"Total Pixels: {total_pixels}")
        return total_pixels

    def convert_to_grayscale(self):
        """Converts the image to grayscale and saves it."""
        if self.image is None:
            raise ValueError("Image not loaded. Call read_image() first.")
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("gray_image.jpg", self.gray_image)
        print("Grayscale image saved as 'gray_image.jpg'.")
        return self.gray_image

    def convert_to_binary(self, threshold=128):
        """Converts the grayscale image to binary and saves it."""
        if self.gray_image is None:
            raise ValueError("Grayscale image not available. Call convert_to_grayscale() first.")
        _, self.binary_image = cv2.threshold(self.gray_image, threshold, 255, cv2.THRESH_BINARY)
        cv2.imwrite("binary_image.jpg", self.binary_image)
        print("Binary image saved as 'binary_image.jpg'.")
        return self.binary_image

    def count_black_pixels(self):
        """Counts the number of black pixels in the binary image."""
        if self.binary_image is None:
            raise ValueError("Binary image not available. Call convert_to_binary() first.")
        black_pixels = np.sum(self.binary_image == 0)
        print(f"Black Pixels Count: {black_pixels}")
        return black_pixels

if __name__ == "__main__":

    image_path = "images/img.png"  
    processor = ImageProcessor(image_path)


    processor.read_image()


    processor.display_image("Original Image", processor.image)


    processor.extract_image_size()


    processor.calculate_image_pixels()

    grayscale = processor.convert_to_grayscale()
    processor.display_image("Grayscale Image", grayscale)

    
    binary = processor.convert_to_binary()
    processor.display_image("Binary Image", binary)
    processor.count_black_pixels()
