import cv2 as cv
import numpy as np

# Read input image
img = cv.imread("moon.jpg", cv.IMREAD_GRAYSCALE)

# Apply Gaussian blur
blurred = cv.GaussianBlur(img, (3, 3), 0)

# Apply Laplacian filter
edges = cv.Laplacian(blurred, cv.CV_64F)

# Convert the output image to uint8
edges = cv.convertScaleAbs(edges)

# Display the resultant image
cv.imshow("Edge Detection Result (Laplacian of Gaussian)", edges)
cv.waitKey(0)
cv.destroyAllWindows()
