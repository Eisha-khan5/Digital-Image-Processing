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
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Define a kernel for closing
    kernel_size = (5, 5)  # You can adjust the kernel size
    kernel = np.ones(kernel_size, np.uint8)

    # Perform closing (dilation followed by erosion)
    closed_image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)

    # Display the closed image
    plt.subplot(1, 2, 2)
    plt.imshow(closed_image, cmap='gray')
    plt.title('Closed Image')
    plt.axis('off')

    # Show both images
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
