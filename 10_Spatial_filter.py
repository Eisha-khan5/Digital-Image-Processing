import cv2 as cv
import numpy as np
import os

# Read the input image
input_image_path = "cat.jpg"  # Change this to your image path
output_dir = "output"  # Directory to save the output image
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

input_image = cv.imread(input_image_path)

# Define a 3x3 averaging filter kernel
kernel = np.ones((3, 3), np.float32) / 9

# Apply the filter using OpenCV's filter2D function
result_image = cv.filter2D(input_image, -1, kernel)

# Save the resultant image
output_image_path = os.path.join(output_dir, "result_image.jpg")
cv.imwrite(output_image_path, result_image)

# Display the original and resultant images
cv.imshow("Original Image", input_image)
cv.imshow("Resultant Image", result_image)
cv.waitKey(0)
cv.destroyAllWindows()
