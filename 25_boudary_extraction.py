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

    # Define a kernel for erosion
    kernel_size = (3, 3)  # You can adjust the kernel size
    kernel = np.ones(kernel_size, np.uint8)

    # Perform erosion
    eroded_image = cv.erode(image, kernel, iterations=1)

    # Perform boundary extraction (original - eroded)
    boundary_image = cv.subtract(image, eroded_image)

    # Display the boundary image
    plt.subplot(1, 2, 2)
    plt.imshow(boundary_image, cmap='gray')
    plt.title('Boundary Image')
    plt.axis('off')

    # Show both images
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
