import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def apply_histogram_normalization(input_image):
    # Convert input image to grayscale
    gray_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)

    # Apply Histogram Equalization
    normalized_image = cv.equalizeHist(gray_image)

    # Compute histograms
    input_hist = cv.calcHist([gray_image], [0], None, [256], [0, 256])
    output_hist = cv.calcHist([normalized_image], [0], None, [256], [0, 256])

    return normalized_image, input_hist, output_hist

def main():
    # Read input image
    input_image = cv.imread("car.jpg")

    # Apply Histogram Normalization
    normalized_image, input_hist, output_hist = apply_histogram_normalization(input_image)

    # Display input and output histograms
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(input_hist, color='blue')
    plt.title('Input Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.subplot(1, 2, 2)
    plt.plot(output_hist, color='red')
    plt.title('Output Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    # Display input and output images
    cv.imshow("Input Image", input_image)
    cv.imshow("Normalized Image", normalized_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
