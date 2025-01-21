import cv2
import numpy as np
import os
class Perform:
    def __init__(self, width, height):
        """
        Constructor initializes a blank canvas using np.zeros.
        :param width: Width of the canvas.
        :param height: Height of the canvas.
        """
        self.blank = np.zeros((height, width, 3), dtype=np.uint8)

    @staticmethod
    def available_shapes():
        return ['rectangle', 'triangle', 'square', 'rhombus']  

    def make_shape(self, startX, startY, endX, endY, shape_type, color=(255, 255, 255), thickness=2):
        if shape_type == 'rectangle':
            cv2.rectangle(self.blank, (startX, startY), (endX, endY), color, thickness)
        elif shape_type == 'triangle':
            pts = np.array([[startX, endY], [(startX + endX) // 2, startY], [endX, endY]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(self.blank, [pts], isClosed=True, color=color, thickness=thickness)
        elif shape_type == 'square':
            side_length = min(abs(endX - startX), abs(endY - startY))
            endX = startX + side_length
            endY = startY + side_length
            cv2.rectangle(self.blank, (startX, startY), (endX, endY), color, thickness)
        elif shape_type == 'rhombus':  
            centerX, centerY = (startX + endX) // 2, (startY + endY) // 2
            pts = np.array([
                [centerX, startY],
                [startX, centerY],
                [centerX, endY],
                [endX, centerY]
            ], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(self.blank, [pts], isClosed=True, color=color, thickness=thickness)
        else:
            raise ValueError("Unsupported shape type, Use rectangle, triangle, square, or rhombus.")

        return self.blank  
    # since all of them modify in place so i will be performing translation on blank only
    def translate(self,tx,ty):
        M=np.float32([[1,0,tx],[0,1,ty]])
        translated=cv2.warpAffine(self.blank,M,(self.blank.shape[1],self.blank.shape[0]))
        return translated
    def remove_shape(self,startX,startY,endX,endY,shape_type):
        """_summary_

        Args:
            startX (_type_): _description_
            startY (_type_): _description_
            endX (_type_): _description_
            endY (_type_): _description_
            shape_type (_type_): _description_
        """
        if shape_type=='rectangle':
            cv2.rectangle(self.blank, (startX+10, startY+10), (endX+10, endY+10), (0, 0, 0), -1)  
        elif shape_type=='triangle':
            pts = np.array([[startX, endY], [(startX + endX) // 2, startY], [endX, endY]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(self.blank, [pts], (0, 0, 0))
        elif shape_type=='square':
            side_length = min(abs(endX - startX), abs(endY - startY))
            endX = startX + side_length
            endY = startY + side_length
            cv2.rectangle(self.blank, (startX, startY), (endX, endY), (0, 0, 0), -1) 
        elif shape_type=='rhombus':
            
            centerX, centerY = (startX + endX) // 2, (startY + endY) // 2
            pts = np.array([
                [centerX, startY],
                [startX, centerY],
                [centerX, endY],
                [endX, centerY]
            ], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(self.blank, [pts], (0, 0, 0))
        else:
            raise ValueError("Unsupported shape type")
        return self.blank
    def scale(self, scale_x, scale_y):
        """_summary_
        

        Args:
            scale_x (_type_): _description_
            scale_y (_type_): _description_

        Returns:
            _type_: _description_
        """
        center = (self.blank.shape[1] // 2, self.blank.shape[0] // 2)

        # scaling matrix
        M = cv2.getRotationMatrix2D(center, 0, scale_x)  # Here rotation angle is 0, just scaling
        scaled_image = cv2.warpAffine(self.blank, M, (self.blank.shape[1], self.blank.shape[0]))
        
        return scaled_image

    def rotate(self, angle):
        """Rotate the entire image by the given angle."""
        # Get the center of the canvas
        center = (self.blank.shape[1] // 2, self.blank.shape[0] // 2)

        # Create a rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1)  # The 1 here means no scaling during rotation
        rotated_image = cv2.warpAffine(self.blank, M, (self.blank.shape[1], self.blank.shape[0]))
        
        return rotated_image

    def reflect(self, axis='horizontal'):
        """Reflect the image either horizontally or vertically."""
        if axis == 'horizontal':
            reflected_image = cv2.flip(self.blank, 1)  # Horizontal flip
        elif axis == 'vertical':
            reflected_image = cv2.flip(self.blank, 0)  # Vertical flip
        else:
            raise ValueError("Axis must be either 'horizontal' or 'vertical'.")
        
        return reflected_image
    def shear(self, shear_x, shear_y):
        """ Shear the image along both the X and Y axes. """
        M = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])  # Shear matrix
        sheared_image = cv2.warpAffine(self.blank, M, (self.blank.shape[1], self.blank.shape[0]))
        return sheared_image
    def crop(self, startX, startY, endX, endY):
        """
        Crop a specific region from the canvas.
        :param startX: Top-left X-coordinate of the cropping area.
        :param startY: Top-left Y-coordinate of the cropping area.
        :param endX: Bottom-right X-coordinate of the cropping area.
        :param endY: Bottom-right Y-coordinate of the cropping area.
        :return: Cropped region of the canvas.
        """
        return self.blank[startY:endY, startX:endX]
    def save_image(self, filepath):
        cv2.imwrite(filepath, self.blank)
        print(f"Image saved: {filepath}")

if __name__ == '__main__':
    output_dir = 'lab1/B_41_E22CSEU1207_RaghavJuneja/results/lab1comp/edited_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    blank_width, blank_height = 500, 500

    # rectangle = Perform(blank_width, blank_height)
    # canvas1 = rectangle.make_shape(50, 50, 200, 150, shape_type='rectangle', color=(0, 255, 0)) 
    # cv2.imshow("Rectangle Canvas", canvas1)
    # # rectangle.save_image(f"{output_dir}/rectangle_canvas.png")

    # triangle = Perform(blank_width, blank_height)
    # canvas2 = triangle.make_shape(250, 50, 400, 200, shape_type='triangle', color=(255, 0, 0))
    # cv2.imshow('Triangle Canvas', canvas2)
    # # triangle.save_image(f"{output_dir}/triangle_canvas.png")

    # square = Perform(blank_width, blank_height)
    # canvas3 = square.make_shape(50, 250, 200, 400, shape_type='square', color=(0, 0, 255))
    # cv2.imshow('Square Canvas', canvas3)
    # # square.save_image(f"{output_dir}/square_canvas.png")

    # rhombus = Perform(blank_width, blank_height)
    # canvas4 = rhombus.make_shape(250, 250, 400, 400, shape_type='rhombus', color=(255, 255, 0))
    # cv2.imshow('Rhombus Canvas', canvas4)
    # # rhombus.save_image(f"{output_dir}/rhombus_canvas.png")

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # combined_canvas = Perform(blank_width, blank_height)
    # combined_canvas.make_shape(50, 50, 200, 150, shape_type='rectangle', color=(0, 255, 0))
    # combined_canvas.make_shape(250, 50, 400, 200, shape_type='triangle', color=(255, 0, 0))
    # combined_canvas.make_shape(50, 250, 200, 400, shape_type='square', color=(0, 0, 255))
    # combined_canvas.make_shape(250, 250, 400, 400, shape_type='rhombus', color=(255, 255, 0))
    # combined_canvas.save_image(f"{output_dir}/allshapes.png")
    
    
    # canvas = Perform(blank_width, blank_height)
    # canvas.make_shape(50, 50, 200, 150, shape_type='rectangle', color=(0, 255, 0))
    # canvas.make_shape(250, 50, 400, 200, shape_type='triangle', color=(255, 0, 0))
    # canvas.make_shape(50, 250, 200, 400, shape_type='square', color=(0, 0, 255))
    # canvas.make_shape(250, 250, 400, 400, shape_type='rhombus', color=(255, 255, 0))

    # Save original canvas
    # canvas.save_image(f"{output_dir}/original_canvas.png")

    # # Perform scaling
    # scaled_canvas = canvas.scale(1.5, 1.5)
    # cv2.imshow("Scaled Canvas", scaled_canvas)
    # cv2.imwrite(f"{output_dir}/scaled_canvas.png", scaled_canvas)

    # # Perform rotation
    # rotated_canvas = canvas.rotate(45)
    # cv2.imshow("Rotated Canvas", rotated_canvas)
    # cv2.imwrite(f"{output_dir}/rotated_canvas.png", rotated_canvas)
    
    # translated_canvas = canvas.translate(50, 50)  
    # cv2.imshow("Translated Canvas", translated_canvas)
    # cv2.imwrite(f"{output_dir}/translated_canvas.png", translated_canvas)

    # if we just want to move only one shape in canva-> we can black oriinal area
    # create new shape and then perform
    # rectangle = Perform(blank_width, blank_height)
    # canvas1 = rectangle.make_shape(50, 50, 200, 150, shape_type='rectangle', color=(0, 255, 0))
    # cv2.imshow("Rectangle Canvas", canvas1)
    # translated_canvas1 = rectangle.translate(50, 50)  # Apply translation
    # cv2.imshow("Translated Rectangle", translated_canvas1)
    # cv2.imwrite(f"{output_dir}/translated_rectangle_canvas.png", translated_canvas1)

    # triangle = Perform(blank_width, blank_height)
    # canvas2 = triangle.make_shape(250, 50, 400, 200, shape_type='triangle', color=(255, 0, 0))
    # cv2.imshow('Triangle Canvas', canvas2)
    # translated_canvas2 = triangle.translate(50, 50)  # Apply translation
    # cv2.imshow("Translated Triangle", translated_canvas2)
    # cv2.imwrite(f"{output_dir}/translated_triangle_canvas.png", translated_canvas2)

    # square = Perform(blank_width, blank_height)
    # canvas3 = square.make_shape(50, 250, 200, 400, shape_type='square', color=(0, 0, 255))
    # cv2.imshow('Square Canvas', canvas3)
    # translated_canvas3 = square.translate(50, 50)  # Apply translation
    # cv2.imshow("Translated Square", translated_canvas3)
    # cv2.imwrite(f"{output_dir}/translated_square_canvas.png", translated_canvas3)

    # rhombus = Perform(blank_width, blank_height)
    # canvas4 = rhombus.make_shape(250, 250, 400, 400, shape_type='rhombus', color=(255, 255, 0))
    # cv2.imshow('Rhombus Canvas', canvas4)
    # translated_canvas4 = rhombus.translate(50, 50)  # Apply translation
    # cv2.imshow("Translated Rhombus", translated_canvas4)
    # cv2.imwrite(f"{output_dir}/translated_rhombus_canvas.png", translated_canvas4)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # canvas.save_image(f"{output_dir}/rotated.png")
    
    # reflected_canvas = canvas.reflect('horizontal')
    # cv2.imshow("Reflected Canvas (Horizontal)", reflected_canvas)

    # reflected_canvas_vertical = canvas.reflect('vertical')
    # cv2.imshow("Reflected Canvas (Vertical)", reflected_canvas_vertical)
    
    # sheared_canvas = canvas.shear(0.5, 0.2)  # Example shear factors
    # cv2.imshow("Sheared Canvas", sheared_canvas)
    # # translated_canvas = canvas.translate(100, 100)
    # # cv2.imshow("Translated Canvas", translated_canvas)
    # cropped_region = canvas.crop(50, 50, 200, 150)
    # cv2.imshow("Cropped Region", cropped_region)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # rectangle = Perform(blank_width, blank_height)
    # canvas1 = rectangle.make_shape(50, 50, 200, 150, shape_type='rectangle', color=(0, 255, 0))
    # translated_canvas1 = rectangle.translate(50, 50)
    # rotated_canvas1 = rectangle.rotate(45)
    # reflected_canvas1 = rectangle.reflect('horizontal')
    # sheared_canvas1 = rectangle.shear(0.5, 0.2)

    # # Save transformed images
    # cv2.imwrite(f"{output_dir}/translated_rectangle_canvas.png", translated_canvas1)
    # cv2.imwrite(f"{output_dir}/rotated_rectangle_canvas.png", rotated_canvas1)
    # cv2.imwrite(f"{output_dir}/reflected_rectangle_canvas.png", reflected_canvas1)
    # cv2.imwrite(f"{output_dir}/sheared_rectangle_canvas.png", sheared_canvas1)

    # # Create and apply transformations for Triangle
    # triangle = Perform(blank_width, blank_height)
    # canvas2 = triangle.make_shape(250, 50, 400, 200, shape_type='triangle', color=(255, 0, 0))
    # translated_canvas2 = triangle.translate(50, 50)
    # rotated_canvas2 = triangle.rotate(45)
    # reflected_canvas2 = triangle.reflect('vertical')
    # sheared_canvas2 = triangle.shear(0.5, 0.2)

    # # Save transformed images
    # cv2.imwrite(f"{output_dir}/translated_triangle_canvas.png", translated_canvas2)
    # cv2.imwrite(f"{output_dir}/rotated_triangle_canvas.png", rotated_canvas2)
    # cv2.imwrite(f"{output_dir}/reflected_triangle_canvas.png", reflected_canvas2)
    # cv2.imwrite(f"{output_dir}/sheared_triangle_canvas.png", sheared_canvas2)

    # # Create and apply transformations for Square
    # square = Perform(blank_width, blank_height)
    # canvas3 = square.make_shape(50, 250, 200, 400, shape_type='square', color=(0, 0, 255))
    # translated_canvas3 = square.translate(50, 50)
    # rotated_canvas3 = square.rotate(45)
    # reflected_canvas3 = square.reflect('horizontal')
    # sheared_canvas3 = square.shear(0.5, 0.2)

    
    # cv2.imwrite(f"{output_dir}/translated_square_canvas.png", translated_canvas3)
    # cv2.imwrite(f"{output_dir}/rotated_square_canvas.png", rotated_canvas3)
    # cv2.imwrite(f"{output_dir}/reflected_square_canvas.png", reflected_canvas3)
    # cv2.imwrite(f"{output_dir}/sheared_square_canvas.png", sheared_canvas3)

    
    # rhombus = Perform(blank_width, blank_height)
    # canvas4 = rhombus.make_shape(250, 250, 400, 400, shape_type='rhombus', color=(255, 255, 0))
    # translated_canvas4 = rhombus.translate(50, 50)
    # rotated_canvas4 = rhombus.rotate(45)
    # reflected_canvas4 = rhombus.reflect('vertical')
    # sheared_canvas4 = rhombus.shear(0.5, 0.2)

    # cv2.imwrite(f"{output_dir}/translated_rhombus_canvas.png", translated_canvas4)
    # cv2.imwrite(f"{output_dir}/rotated_rhombus_canvas.png", rotated_canvas4)
    # cv2.imwrite(f"{output_dir}/reflected_rhombus_canvas.png", reflected_canvas4)
    # cv2.imwrite(f"{output_dir}/sheared_rhombus_canvas.png", sheared_canvas4)


    # cv2.imshow("Original Rectangle", canvas1)
    # cv2.imshow("Translated Rectangle", translated_canvas1)
    # cv2.imshow("Rotated Rectangle", rotated_canvas1)
    # cv2.imshow("Reflected Rectangle", reflected_canvas1)
    # cv2.imshow("Sheared Rectangle", sheared_canvas1)

    # cv2.imshow("Original Triangle", canvas2)
    # cv2.imshow("Translated Triangle", translated_canvas2)
    # cv2.imshow("Rotated Triangle", rotated_canvas2)
    # cv2.imshow("Reflected Triangle", reflected_canvas2)
    # cv2.imshow("Sheared Triangle", sheared_canvas2)

    # cv2.imshow("Original Square", canvas3)
    # cv2.imshow("Translated Square", translated_canvas3)
    # cv2.imshow("Rotated Square", rotated_canvas3)
    # cv2.imshow("Reflected Square", reflected_canvas3)
    # cv2.imshow("Sheared Square", sheared_canvas3)

    # cv2.imshow("Original Rhombus", canvas4)
    # cv2.imshow("Translated Rhombus", translated_canvas4)
    # cv2.imshow("Rotated Rhombus", rotated_canvas4)
    # cv2.imshow("Reflected Rhombus", reflected_canvas4)
    # cv2.imshow("Sheared Rhombus", sheared_canvas4)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    canvas = Perform(blank_width, blank_height)
    canvas.make_shape(50, 50, 200, 150, shape_type='rectangle', color=(0, 255, 0))
    canvas.make_shape(250, 50, 400, 200, shape_type='triangle', color=(255, 0, 0))
    canvas.make_shape(50, 250, 200, 400, shape_type='square', color=(0, 0, 255))
    canvas.make_shape(250, 250, 400, 400, shape_type='rhombus', color=(255, 255, 0))

    rotated_canvas = canvas.rotate(45)
    cv2.imwrite(f"{output_dir}/rotated_canvas.png", rotated_canvas)

    
    reflected_canvas = canvas.reflect('horizontal')
    cv2.imwrite(f"{output_dir}/reflected_canvas.png", reflected_canvas)

    
    sheared_canvas = canvas.shear(0.5, 0.2)
    cv2.imwrite(f"{output_dir}/sheared_canvas.png", sheared_canvas)
