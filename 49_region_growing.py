import cv2 as cv
import numpy as np

def region_growing_segmentation(img, seed, threshold=20):
    # Convert image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Create a binary mask for segmented region
    segmented_img = np.zeros_like(gray)

    # Define a queue for pixels to be processed
    queue = [seed]

    # Process pixels in the queue
    while queue:
        # Get the current pixel
        x, y = queue.pop(0)

        # Check if pixel is within image bounds
        if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
            # Check if pixel is not already segmented
            if segmented_img[y, x] == 0:
                # Segment the pixel
                segmented_img[y, x] = 255

                # Check neighboring pixels for segmentation
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if 0 <= x + dx < gray.shape[1] and 0 <= y + dy < gray.shape[0]:
                            # Check intensity difference between neighboring pixels and seed pixel
                            if abs(int(gray[y + dy, x + dx]) - int(gray[seed[1], seed[0]])) <= threshold:
                                # Add neighboring pixel to the queue
                                queue.append((x + dx, y + dy))

    return segmented_img

# Read input image
img = cv.imread("moon.jpg")

# Define seed point for region growing
seed_point = (100, 100)  # Example seed point

# Perform region growing segmentation
segmented_img = region_growing_segmentation(img, seed_point)

# Display the resultant image
cv.imshow("Region Growing Segmentation Result", segmented_img)
cv.waitKey(0)
cv.destroyAllWindows()
