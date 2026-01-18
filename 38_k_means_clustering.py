import cv2 as cv
import numpy as np

# Read input image
img = cv.imread("moon.jpg")

# Reshape image into a 2D array of pixels
pixels = img.reshape((-1, 3)).astype(np.float32)

# Define criteria and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
k = 8  # Number of clusters
_, labels, centers = cv.kmeans(pixels, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

# Convert the centers back into uint8
centers = np.uint8(centers)

# Map each pixel to its corresponding centroid color
segmented_img = centers[labels.flatten()]

# Reshape the segmented image back to the original shape
segmented_img = segmented_img.reshape(img.shape)

# Display the resultant image
cv.imshow("K-means Clustering Result", segmented_img)
cv.waitKey(0)
cv.destroyAllWindows()
