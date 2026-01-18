import cv2 as cv

# Read input image in grayscale
img = cv.imread("moon.jpg", cv.IMREAD_GRAYSCALE)

# Define threshold values
thresholds = [50, 100, 150]

# Apply multilevel thresholding
_, thresh1 = cv.threshold(img, thresholds[0], 255, cv.THRESH_BINARY)
_, thresh2 = cv.threshold(img, thresholds[1], 255, cv.THRESH_BINARY)
_, thresh3 = cv.threshold(img, thresholds[2], 255, cv.THRESH_BINARY)

# Combine thresholded images
thresh_combined = cv.bitwise_or(thresh1, thresh2)
thresh_combined = cv.bitwise_or(thresh_combined, thresh3)

# Display the resultant image
cv.imshow("Multilevel Thresholding Result", thresh_combined)
cv.waitKey(0)
cv.destroyAllWindows()
