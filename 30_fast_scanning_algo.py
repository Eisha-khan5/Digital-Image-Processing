import cv2 as cv
import numpy as np

def fast_scanning_algorithm(img, seed):
    # Convert image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Create a copy of the input image
    segmented_img = img.copy()

    # Define the fill color (in BGR format)
    fill_color = (0, 255, 0)  # Green color

    # Define the flood fill parameters
    connectivity = 4  # 4-connected neighbors
    flags = connectivity + (255 << 8) + cv.FLOODFILL_MASK_ONLY

    # Perform flood fill starting from the seed point
    cv.floodFill(segmented_img, mask=None, seedPoint=seed, newVal=fill_color, loDiff=(0, 0, 0), upDiff=(0, 0, 0), flags=flags)

    return segmented_img

# Read input image
img = cv.imread("moon.jpg")

# Define seed point for flood fill
seed_point = (100, 100)  # Example seed point

# Perform fast scanning algorithm
segmented_img = fast_scanning_algorithm(img, seed_point)

# Display the resultant image
cv.imshow("Fast Scanning Algorithm Result", segmented_img)
cv.waitKey(0)
cv.destroyAllWindows()
