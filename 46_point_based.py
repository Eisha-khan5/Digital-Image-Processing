import cv2 as cv
import numpy as np

# Read input image
img = cv.imread("moon.jpg", cv.IMREAD_GRAYSCALE)

# Apply Laplacian of Gaussian (LoG) filter
log_img = cv.Laplacian(img, cv.CV_64F)

# Threshold the filtered image
threshold = 100  # Adjust threshold as needed
_, points = cv.threshold(np.abs(log_img), threshold, 255, cv.THRESH_BINARY)

# Convert to uint8 for display
points = np.uint8(points)

# Display the resultant image
cv.imshow("Point Detection Result", points)
cv.waitKey(0)
cv.destroyAllWindows()
