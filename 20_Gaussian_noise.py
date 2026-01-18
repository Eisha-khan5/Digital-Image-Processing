import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

def apply_gaussian_noise(image_path):
    # Load the image
    img = cv.imread(image_path)

    # Add Gaussian noise
    mean = 0
    sigma = 25
    noisy_image = np.zeros(img.shape, np.uint8)
    cv.randn(noisy_image, mean, sigma)

    # Add noise to the original image
    noisy_image = cv.add(img, noisy_image)

    return noisy_image

def main():
    # Path to the image
    image_path = "cat.jpg"

    # Apply Gaussian noise
    noisy_image = apply_gaussian_noise(image_path)

    # Display original and noisy images
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(noisy_image, cv.COLOR_BGR2RGB))
    plt.title("Noisy Image")
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()
