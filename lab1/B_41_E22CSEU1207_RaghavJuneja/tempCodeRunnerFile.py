as = Perform(blank_width, blank_height)
    canvas.make_shape(50, 50, 200, 150, shape_type='rectangle', color=(0, 255, 0))
    canvas.make_shape(250, 50, 400, 200, shape_type='triangle', color=(255, 0, 0))
    canvas.make_shape(50, 250, 200, 400, shape_type='square', color=(0, 0, 255))
    canvas.make_shape(250, 250, 400, 400, shape_type='rhombus', color=(255, 255, 0))


    cropped_canvas = canvas.crop(100, 100, 400, 400)
    cv2.imwrite(f"{output_dir}/cropped_canvas.png", cropped_canvas)

    #
    rectangle_cropped = canvas.crop(50, 50, 200, 150)
    cv2.imwrite(f"{output_dir}/rectangle_cropped.png", rectangle_cropped)

    triangle_cropped = canvas.crop(250, 50, 400, 200)
    cv2.imwrite(f"{output_dir}/triangle_cropped.png", triangle_cropped)


    square_cropped = canvas.crop(50, 250, 200, 400)
    cv2.imwrite(f"{output_dir}/square_cropped.png", square_cropped)


    rhombus_cropped = canvas.crop(250, 250, 400, 400)
    cv2.imwrite(f"{output_dir}/rhombus_cropped.png", rhombus_cropped)

