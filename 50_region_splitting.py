import cv2 as cv
import numpy as np

def region_splitting_segmentation(img):
    # Convert image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Initialize segmented image
    segmented_img = np.zeros_like(gray)

    # Define a threshold for region splitting
    threshold = 20

    # Define a queue for pixels to be processed
    queue = [(0, 0, gray.shape[1]-1, gray.shape[0]-1)]  # (x1, y1, x2, y2)

    # Process regions in the queue
    while queue:
        # Get the bounding box coordinates
        x1, y1, x2, y2 = queue.pop(0)

        # Compute mean intensity of the region
        mean_intensity = np.mean(gray[y1:y2+1, x1:x2+1])

        # Check if region needs to be split
        if np.max(gray[y1:y2+1, x1:x2+1]) - np.min(gray[y1:y2+1, x1:x2+1]) > threshold:
            # Split region horizontally
            if x2 - x1 > y2 - y1:
                mid = (x1 + x2) // 2
                queue.extend([(x1, y1, mid, y2), (mid+1, y1, x2, y2)])
            # Split region vertically
            else:
                mid = (y1 + y2) // 2
                queue.extend([(x1, y1, x2, mid), (x1, mid+1, x2, y2)])
        else:
            # Fill region with mean intensity
            segmented_img[y1:y2+1, x1:x2+1] = mean_intensity

    return segmented_img

# Read input image
img = cv.imread("moon.jpg")

# Perform region splitting segmentation
segmented_img = region_splitting_segmentation(img)

# Display the resultant image
cv.imshow("Region Splitting Segmentation Result", segmented_img)
cv.waitKey(0)
cv.destroyAllWindows()
