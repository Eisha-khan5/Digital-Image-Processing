import cv2 as cv
import numpy as np

# Read input image
img = cv.imread("moon.jpg", cv.IMREAD_GRAYSCALE)

# Apply Sobel filter for horizontal gradient
edges = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)

# Convert the output image to uint8
edges = cv.convertScaleAbs(edges)

# Display the resultant image
cv.imshow("Edge Detection Result (Horizontal Gradient)", edges)
cv.waitKey(0)
cv.destroyAllWindows()