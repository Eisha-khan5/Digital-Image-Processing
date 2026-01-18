import cv2 as cv
import numpy as np

# Read input image
img = cv.imread("moon.jpg", cv.IMREAD_GRAYSCALE)

# Define the Robert filter
robert_filter = np.array([[-1, 0],
                          [0, 1]])

# Apply 2D convolution with the Robert filter
edges = cv.filter2D(img, -1, robert_filter)

# Display the resultant image
cv.imshow("Edge Detection Result (Robert Filter)", edges)
cv.waitKey(0)
cv.destroyAllWindows()
