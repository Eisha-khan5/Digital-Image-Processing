import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load an image from file
    image_path = 'circles.png'  # Replace with your image path
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Display the original image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Define a kernel for opening
    kernel = np.ones((5, 5), np.uint8)  # You can adjust the kernel size

    # Perform opening (erosion followed by dilation)
    opened_image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)

    # Display the opened image
    plt.subplot(1, 2, 2)
    plt.imshow(opened_image, cmap='gray')
    plt.title('Opened Image')
    plt.axis('off')

    # Show both images
    plt.show()

if __name__ == "__main__":
    main()
