import cv2 as cv
import numpy as np

# Read input image
img = cv.imread("moon.jpg")

# Reshape image to 2D array of pixels
pixels = img.reshape((-1, 3)).astype(np.float32)

# Define criteria and apply k-means clustering
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
k = 2  # Number of clusters (you can adjust this)
_, labels, centers = cv.kmeans(pixels, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

# Reshape the labels to the original image shape
segmented_img = labels.reshape(img.shape[0], img.shape[1])

# Convert labels to uint8 for displaying
segmented_img = np.uint8(segmented_img * 255 / (k-1))

# Display the resultant image
cv.imshow("Region-Based Segmentation Result", segmented_img)
cv.waitKey(0)
cv.destroyAllWindows()
