import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

def remove_noise_contra_harmonic_mean(image_path, kernel_size, Q):
    # Load the image
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    # Pad the image to handle border pixels
    padded_img = cv.copyMakeBorder(img, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, cv.BORDER_REPLICATE)

    # Initialize filtered image
    filtered_image = np.zeros_like(img)

    # Apply contra-harmonic mean filter
    for i in range(kernel_size // 2, img.shape[0] + kernel_size // 2):
        for j in range(kernel_size // 2, img.shape[1] + kernel_size // 2):
            neighborhood = padded_img[i - kernel_size // 2: i + kernel_size // 2 + 1, j - kernel_size // 2: j + kernel_size // 2 + 1]
            num = np.sum(np.power(neighborhood, Q + 1))
            den = np.sum(np.power(neighborhood, Q))
            filtered_image[i - kernel_size // 2, j - kernel_size // 2] = num / (den + 1e-6)

    # Convert the filtered image to uint8
    filtered_image = np.uint8(filtered_image)

    return filtered_image

def main():
    # Path to the image
    image_path = "lady.jpg"

    # Kernel size for the contra-harmonic mean filter (adjust as needed)
    kernel_size = 3  # 3x3 kernel

    # Order parameter (Q) for the contra-harmonic mean filter
    Q = -2.5  # You can adjust this parameter as needed

    # Remove noise using contra-harmonic mean filter
    denoised_image = remove_noise_contra_harmonic_mean(image_path, kernel_size, Q)

    # Display original and denoised images
    plt.subplot(1, 2, 1)
    plt.imshow(cv.imread(image_path, cv.IMREAD_GRAYSCALE), cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(denoised_image, cmap='gray')
    plt.title("Denoised Image")
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()
