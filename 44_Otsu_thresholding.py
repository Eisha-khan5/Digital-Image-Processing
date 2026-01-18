import cv2 as cv

# Read input image in grayscale
img = cv.imread("moon.jpg", cv.IMREAD_GRAYSCALE)

# Apply Otsu's thresholding
_, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Display the resultant image
cv.imshow("Otsu Thresholding Result", thresh)
cv.waitKey(0)
cv.destroyAllWindows()