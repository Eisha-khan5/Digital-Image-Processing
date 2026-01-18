import cv2 as cv
import numpy as np

# Read input image
img = cv.imread("moon.jpg", cv.IMREAD_GRAYSCALE)

# Apply Canny edge detection
edges = cv.Canny(img, 100, 200)

# Display the resultant image
cv.imshow("Edge-Based Segmentation Result", edges)
cv.waitKey(0)
cv.destroyAllWindows()
