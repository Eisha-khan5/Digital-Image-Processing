import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def dualistic_sub_image_hist_equalization(img):
    # Convert image to grayscale
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Split image into two equal halves
    height, width = gray_img.shape
    half_width = width // 2

    left_half = gray_img[:, :half_width]
    right_half = gray_img[:, half_width:]

    # Apply histogram equalization on each half
    left_half_eq = cv.equalizeHist(left_half)
    right_half_eq = cv.equalizeHist(right_half)

    # Merge the equalized halves
    equalized_img = np.concatenate((left_half_eq, right_half_eq), axis=1)

    # Convert back to BGR color space
    output_img = cv.cvtColor(equalized_img, cv.COLOR_GRAY2BGR)
    
    return output_img

def main():
    # Load image
    img = cv.imread('cat.jpg')

    if img is None:
        print("Error: Image not found!")
        return

    # Apply dualistic sub-image histogram equalization
    result_img = dualistic_sub_image_hist_equalization(img)

    # Display original and resultant images
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title('Original')

    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(result_img, cv.COLOR_BGR2RGB))
    plt.title('Dualistic Sub-Image Hist Equalized')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
