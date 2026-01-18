import cv2 as cv
import numpy as np

def object_detection(img):
    # Convert image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using Canny
    edges = cv.Canny(blurred, 30, 150)

    # Find contours in the edge-detected image
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around detected objects
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img

# Read input image
img = cv.imread("moon.jpg")

# Perform object detection
result = object_detection(img)

# Display the resultant image
cv.imshow("Object Detection Result", result)
cv.waitKey(0)
cv.destroyAllWindows()
