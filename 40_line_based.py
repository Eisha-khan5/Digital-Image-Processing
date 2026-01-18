import cv2 as cv
import numpy as np

# Read input image
img = cv.imread("moon.jpg", cv.IMREAD_GRAYSCALE)

# Apply Gaussian blur to reduce noise
blurred_img = cv.GaussianBlur(img, (3, 3), 0)

# Apply Canny edge detection
edges = cv.Canny(blurred_img, 50, 150)

# Perform Hough Line Transform
lines = cv.HoughLines(edges, 1, np.pi/180, 100)

# Draw detected lines on a blank image
result = np.zeros_like(img)
if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(result, (x1, y1), (x2, y2), (255), 1)

# Display the resultant image
cv.imshow("Line Detection Result", result)
cv.waitKey(0)
cv.destroyAllWindows()
