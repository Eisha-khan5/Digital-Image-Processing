import cv2 as cv
import numpy as np

# Read input image
img = cv.imread("moon.jpg")

# Convert image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise
blurred = cv.GaussianBlur(gray, (5, 5), 0)

# Adaptive thresholding for better segmentation
mask = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 4)

# Find contours
contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Create a blank mask for the object
object_mask = np.zeros_like(gray)

# Draw contours on the object mask
cv.drawContours(object_mask, contours, -1, (255), thickness=cv.FILLED)

# Invert the object mask to get the background mask
background_mask = cv.bitwise_not(object_mask)

# Create a blank image for the segmented result
segmented_img = np.zeros_like(img)

# Apply the masks to the input image
segmented_img = cv.bitwise_and(img, img, mask=object_mask)

# Concatenate original and segmented images horizontally
combined_img = np.hstack((img, segmented_img))

# Display the combined image
cv.imshow("Original and Segmented Images", combined_img)
cv.waitKey(0)
cv.destroyAllWindows()
