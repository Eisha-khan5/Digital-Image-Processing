import cv2 as cv
import numpy as np

# Read input image
img = cv.imread("moon.jpg")

# Convert image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Apply thresholding to segment the image
_, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

# Find contours
contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Draw contours with different colors for each object
segmented_img = np.zeros_like(img)
for i, contour in enumerate(contours):
    color = np.random.randint(0, 255, (3,)).tolist()  # Generate a random color for each object
    cv.drawContours(segmented_img, [contour], -1, color, thickness=cv.FILLED)

# Display the resultant image
cv.imshow("Instance Segmentation Result", segmented_img)
cv.waitKey(0)
cv.destroyAllWindows()
