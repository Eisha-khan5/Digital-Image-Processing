import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

def remove_noise_arithmetic_mean(image_path, kernel_size):
    # Load the image
    img = cv.imread(image_path)

    # Apply arithmetic mean filter
    filtered_image = cv.blur(img, (kernel_size, kernel_size))

    return filtered_image

def main():
    # Path to the image
    image_path = "lady.jpg"

    # Kernel size for the arithmetic mean filter (adjust as needed)
    kernel_size = 3  # 3x3 kernel

    # Remove noise using arithmetic mean filter
    denoised_image = remove_noise_arithmetic_mean(image_path, kernel_size)

    # Display original and denoised images
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(denoised_image, cv.COLOR_BGR2RGB))
    plt.title("Denoised Image")
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()
