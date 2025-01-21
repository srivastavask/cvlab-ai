import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, IMG_PATH):
        self.IMG_PATH = IMG_PATH
        self.img = cv2.imread(IMG_PATH)
        if self.img is None:
            raise FileNotFoundError("No image was found")

    @staticmethod
    def method_to_resize():
        methods = {
            'linear': cv2.INTER_LINEAR,
            'nearest': cv2.INTER_NEAREST,
            'polynomial': cv2.INTER_CUBIC,
        }
        print("Available interpolation methods:")
        for key, value in methods.items():
            print(f"  - {key.capitalize()}: OpenCV code {value}")

    def resize_to_dimensions(self, width, height, method='linear'):
        # this method changes to a specific dimension sya 40 by 40 to 80 by 80 these
        # are to be specified in height and width
        # so if u try to save this u will see new image only
        """
        Resize the image to a specific width and height.
        
        :param width: The desired width of the image.
        :param height: The desired height of the image.
        :param method: The interpolation method (default is 'linear').
        """
        print(f"Original dimensions: {self.get_dimensions()}")
        methods = {
            'linear': cv2.INTER_LINEAR,
            'nearest': cv2.INTER_NEAREST,
            'polynomial': cv2.INTER_CUBIC,
        }
        if method not in methods:
            raise ValueError(f"Invalid method. Choose from {list(methods.keys())}.")
        
    
        self.img = cv2.resize(self.img, (width, height), interpolation=methods[method])
        print(f"Resized image dimensions: {self.get_dimensions()}")
        
    

    def resize_by_scale(self, fx, fy, method='linear'):
        # it scales along the axis, so dimensions change
        # example if fx=10,fy=10 then 500 by 500 will change to 5000 by 5000 when u display
        # on saving it appears like it isnt resized but the factors are scaled
        """
        Resize the image by scaling factors (fx and fy).
        
        :param fx: The scaling factor for the width.
        :param fy: The scaling factor for the height.
        :param method: The interpolation method (default is 'linear').
        """
        print(f"Original dimensions: {self.get_dimensions()}")
        methods = {
            'linear': cv2.INTER_LINEAR,
            'nearest': cv2.INTER_NEAREST,
            'polynomial': cv2.INTER_CUBIC,
        }
        if method not in methods:
            raise ValueError(f"Invalid method. Choose from {list(methods.keys())}.")
        
       
        self.img = cv2.resize(self.img, None, fx=fx, fy=fy, interpolation=methods[method])
        print(f"Resized image dimensions: {self.get_dimensions()}")
        
    

        

    def blur_image(self, blur_type='box', ksize=(13, 13)):
        """
        Blur the image using different techniques.
        
        :param blur_type: Type of blurring ('box', 'gaussian', 'adaptive')
        :param ksize: Kernel size, tuple (width, height)
        """
        if blur_type == 'box':
            self.img = cv2.blur(self.img, ksize)
            print("Applied Box Blurring")
        elif blur_type == 'gaussian':
            self.img = cv2.GaussianBlur(self.img, ksize, 0)
            print("Applied Gaussian Blurring")
        elif blur_type == 'adaptive':
            self.img = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            print("Applied Adaptive Blurring")
        else:
            raise ValueError("Invalid blur type. Choose from 'box', 'gaussian', or 'adaptive'.")


    def display_image(self, window_name='Image'):
        cv2.imshow(window_name, self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_image(self, output_path):
        try:
            cv2.imshow('Image',self.img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite(output_path, self.img)
            print(f"Image saved to {output_path}")
        except Exception as e:
            print(f"Error saving image: {e}")

    def get_dimensions(self):
        return self.img.shape


if __name__ == '__main__':
    IMG_PATH = 'images/img.png'
    # processor = ImageProcessor(IMG_PATH)
    # print(f"Original image dimensions: {processor.get_dimensions()}")
    # processor.resize_image(10, 10)
    # processor2=ImageProcessor(IMG_PATH)
    # processor2.resize_image(5,5,method='polynomial')
    # processor2.display_image()
    
    # processor.blur_image()
    # processor.display_image()
    # processor.save_image('edited_images/asdf.png')


    processor = ImageProcessor(IMG_PATH)
    processor.resize_image(5,5)
    