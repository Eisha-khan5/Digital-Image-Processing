import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

def remove_noise_geometric_mean(image_path, kernel_size):
    # Load the image
    img = cv.imread(image_path)

    # Convert image to float32 for correct calculation
    img_float = np.float32(img)

    # Apply geometric mean filter
    filtered_image = np.power(np.prod(img_float, axis=2), 1/(kernel_size*kernel_size))

    # Convert filtered image back to uint8
    filtered_image = np.uint8(filtered_image)

    return filtered_image

def main():
    # Path to the image
    image_path = "lady.jpg"

    # Kernel size for the geometric mean filter (adjust as needed)
    kernel_size = 3  # 3x3 kernel

    # Remove noise using geometric mean filter
    denoised_image = remove_noise_geometric_mean(image_path, kernel_size)

    # Display original and denoised images
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(denoised_image, cmap='gray')
    plt.title("Denoised Image")
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()
