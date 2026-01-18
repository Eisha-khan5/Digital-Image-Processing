import cv2 as cv
import numpy as np

# Read input image
img = cv.imread("moon.jpg", cv.IMREAD_GRAYSCALE)

# Apply Sobel filter for horizontal gradient
horizontal_edges = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)

# Apply Sobel filter for vertical gradient
vertical_edges = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)

# Combine the horizontal and vertical edges to get the magnitude of the gradient
edges = cv.magnitude(horizontal_edges, vertical_edges)

# Convert the output image to uint8
edges = cv.convertScaleAbs(edges)

# Display the resultant image
cv.imshow("Combined Edge Detection Result", edges)
cv.waitKey(0)
cv.destroyAllWindows()
