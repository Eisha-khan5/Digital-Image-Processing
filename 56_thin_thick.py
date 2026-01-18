import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def thinning(image):
    size = np.size(image)
    skeleton = np.zeros(image.shape, np.uint8)

    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv.erode(image, element)
        temp = cv.dilate(eroded, element)
        temp = cv.subtract(image, temp)
        skeleton = cv.bitwise_or(skeleton, temp)
        image = eroded.copy()

        zeros = size - cv.countNonZero(image)
        if zeros == size:
            done = True

    return skeleton

def thickening(image, kernel_size=(3, 3)):
    element = cv.getStructuringElement(cv.MORPH_CROSS, kernel_size)
    dilated = cv.dilate(image, element)
    return dilated

def main():
    # Load an image from file
    image_path = 'circles.png'  # Replace with your image path
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Binarize the image
    _, binary_image = cv.threshold(image, 127, 255, cv.THRESH_BINARY)

    # Apply thinning
    thinned_image = thinning(binary_image)

    # Apply thickening
    thickened_image = thickening(binary_image)

    # Display the images
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(binary_image, cmap='gray')
    plt.title('Binary Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(thinned_image, cmap='gray')
    plt.title('Thinned Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(thickened_image, cmap='gray')
    plt.title('Thickened Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
