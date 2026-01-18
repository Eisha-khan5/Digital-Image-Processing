import cv2 as cv

# Read input image in grayscale
img = cv.imread("moon.jpg", cv.IMREAD_GRAYSCALE)

# Apply Canny edge detection with hysteresis thresholding
edges = cv.Canny(img, 50, 150)

# Display the resultant image
cv.imshow("Hysteresis Thresholding Result", edges)
cv.waitKey(0)
cv.destroyAllWindows()
