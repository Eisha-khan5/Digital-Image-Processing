import cv2 as cv
import numpy as np

# Read input image
img = cv.imread("moon.jpg", cv.IMREAD_GRAYSCALE)

# Apply simple thresholding to segment the image
_, segmented_img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

# Display the resultant image
cv.imshow("Pixel-Based Segmentation Result", segmented_img)
cv.waitKey(0)
cv.destroyAllWindows()
