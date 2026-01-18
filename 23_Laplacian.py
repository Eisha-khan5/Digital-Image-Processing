import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def laplacian_filter(image):
    # Apply Laplacian filter
    laplacian = cv.Laplacian(image, cv.CV_64F)

    # Convert back to uint8
    laplacian = np.uint8(np.absolute(laplacian))

    return laplacian

def plot_histogram(image):
    # Calculate histogram
    hist = cv.calcHist([image], [0], None, [256], [0, 256])

    # Plot histogram
    plt.figure()
    plt.title("Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.plot(hist, color='black')
    plt.xlim([0, 256])
    plt.show()

def main():
    # Read input image
    image_path = "cat.jpg"  # Change this to your image path
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Unable to load the image.")
        return

    # Apply Laplacian filter
    laplacian_image = laplacian_filter(image)

    # Display the resultant image and its histogram
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(laplacian_image, cmap='gray')
    plt.title("Laplacian Filter Result")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plot_histogram(laplacian_image)

    plt.show()

if __name__ == "__main__":
    main()
