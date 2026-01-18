import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def bi_histo_equalization(img):
    # Convert image to YUV color space
    yuv_img = cv.cvtColor(img, cv.COLOR_BGR2YUV)
    # Apply histogram equalization on Y channel
    yuv_img[:,:,0] = cv.equalizeHist(yuv_img[:,:,0])
    
    # Convert back to BGR color space
    output_img = cv.cvtColor(yuv_img, cv.COLOR_YUV2BGR)
    
    return output_img

def plot_histogram(img):
    hist, bins = np.histogram(img.flatten(), 256, [0,256])
    plt.plot(hist, color='gray')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

def main():
    # Load image
    img = cv.imread('cat.jpg')

    # Apply bi-histogram equalization
    result_img = bi_histo_equalization(img)

    # Display original and resultant images
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(cv.cvtColor(result_img, cv.COLOR_BGR2RGB))
    plt.title('Bi-Histogram Equalized Image')
    plt.axis('off')

    # Plot histograms
    plt.subplot(2, 2, 3)
    plot_histogram(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    plt.title('Original Image Histogram')

    plt.subplot(2, 2, 4)
    plot_histogram(cv.cvtColor(result_img, cv.COLOR_BGR2GRAY))
    plt.title('Bi-Histogram Equalized Image Histogram')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
