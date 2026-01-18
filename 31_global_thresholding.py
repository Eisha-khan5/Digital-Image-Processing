import cv2 as cv

# Read input image in grayscale
img = cv.imread("moon.jpg", cv.IMREAD_GRAYSCALE)

# Apply global thresholding
_, thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

# Display the resultant image
cv.imshow("Global Thresholding Result", thresh)
cv.waitKey(0)
cv.destroyAllWindows()
