import cv2 as cv
import numpy as np

# Read input image
img = cv.imread("moon.jpg", cv.IMREAD_GRAYSCALE)

# Define the Prewitt filter for horizontal edges
prewitt_horizontal = np.array([[-1, -1, -1],
                                [ 0,  0,  0],
                                [ 1,  1,  1]], dtype=np.float32)

# Define the Prewitt filter for vertical edges
prewitt_vertical = np.array([[-1, 0, 1],
                              [-1, 0, 1],
                              [-1, 0, 1]], dtype=np.float32)

# Apply 2D convolution with the Prewitt filter for horizontal edges
horizontal_edges = cv.filter2D(img, -1, prewitt_horizontal)

# Apply 2D convolution with the Prewitt filter for vertical edges
vertical_edges = cv.filter2D(img, -1, prewitt_vertical)

# Convert the images to the same data type for cv.magnitude
horizontal_edges = horizontal_edges.astype(np.float32)
vertical_edges = vertical_edges.astype(np.float32)

# Compute the magnitude of the gradient (edge strength)
edge_strength = cv.magnitude(horizontal_edges, vertical_edges)

# Convert the edge strength image to uint8 for display
edge_strength = cv.convertScaleAbs(edge_strength)

# Display the resultant image
cv.imshow("Edge Detection Result", edge_strength)
cv.waitKey(0)
cv.destroyAllWindows()
