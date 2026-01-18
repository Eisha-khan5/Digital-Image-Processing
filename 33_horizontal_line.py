import cv2 as cv
import numpy as np

# Read input image
img = cv.imread("moon.jpg", cv.IMREAD_GRAYSCALE)

# Apply Canny edge detection
edges = cv.Canny(img, 100, 200)

# Define the horizontal line detection mask
kernel = np.array([[-1, -1, -1],
                   [ 2,  2,  2],
                   [-1, -1, -1]], dtype=np.float32)

# Apply 2D convolution with the kernel only on the edges
result = cv.filter2D(edges, -1, kernel)

# Display the resultant image
cv.imshow("Horizontal Line Detection Result on Edges", result)
cv.waitKey(0)
cv.destroyAllWindows()