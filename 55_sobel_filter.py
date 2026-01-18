import cv2 as cv
import numpy as np

# Read input image
img = cv.imread("moon.jpg", cv.IMREAD_GRAYSCALE)

# Define the Sobel filter for horizontal edges
sobel_horizontal = np.array([[-1, -2, -1],
                              [ 0,  0,  0],
                              [ 1,  2,  1]])

# Define the Sobel filter for vertical edges
sobel_vertical = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

# Apply 2D convolution with the Sobel filter for horizontal edges
horizontal_edges = cv.filter2D(img, -1, sobel_horizontal)

# Apply 2D convolution with the Sobel filter for vertical edges
vertical_edges = cv.filter2D(img, -1, sobel_vertical)

# Convert images to float32
horizontal_edges = horizontal_edges.astype(np.float32)
vertical_edges = vertical_edges.astype(np.float32)

# Compute the magnitude of the gradient (edge strength)
edge_strength = cv.magnitude(horizontal_edges, vertical_edges)

# Convert the edge strength image to uint8 for display
edge_strength = cv.convertScaleAbs(edge_strength)

# Display the resultant image
cv.imshow("Edge Detection Result (Sobel Filter)", edge_strength)
cv.waitKey(0)
cv.destroyAllWindows()
