import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def run_length_encode(image):
    flat_image = image.flatten()
    encoded = []
    previous_pixel = flat_image[0]
    count = 1

    for pixel in flat_image[1:]:
        if pixel == previous_pixel:
            count += 1
        else:
            encoded.append((previous_pixel, count))
            previous_pixel = pixel
            count = 1
    encoded.append((previous_pixel, count))
    return encoded

def run_length_decode(encoded, shape):
    decoded = []
    for pixel, count in encoded:
        decoded.extend([pixel] * count)
    return np.array(decoded).reshape(shape)

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

    # Apply Run-Length Encoding
    encoded = run_length_encode(image)
    print(f"Encoded: {encoded[:100]}...")  # Print first 100 encoded values for brevity

    # Decode the encoded image
    decoded_image = run_length_decode(encoded, image.shape)

    # Display the decoded image
    plt.subplot(1, 2, 2)
    plt.imshow(decoded_image, cmap='gray')
    plt.title('Decoded Image')
    plt.axis('off')

    # Show both images
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
