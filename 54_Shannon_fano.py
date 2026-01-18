import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Function to build the Shannon-Fano tree and generate codes
def shannon_fano_encode(symbols, codes, prefix=""):
    if len(symbols) == 1:
        codes[symbols[0][0]] = prefix
        return
    total = sum([symbol[1] for symbol in symbols])
    acc = 0
    split_idx = 0
    for i, symbol in enumerate(symbols):
        acc += symbol[1]
        if acc >= total / 2:
            split_idx = i
            break

    shannon_fano_encode(symbols[:split_idx + 1], codes, prefix + "0")
    shannon_fano_encode(symbols[split_idx + 1:], codes, prefix + "1")

# Function to encode data using the Shannon-Fano codes
def shannon_fano_encoding(data, codes):
    return "".join(codes[pixel] for pixel in data)

# Function to decode data using the Shannon-Fano codes
def shannon_fano_decoding(encoded_data, codes):
    reverse_codes = {v: k for k, v in codes.items()}
    current_code = ""
    decoded_data = []

    for bit in encoded_data:
        current_code += bit
        if current_code in reverse_codes:
            decoded_data.append(reverse_codes[current_code])
            current_code = ""

    return decoded_data

def main():
    # Load an image from file
    image_path = 'circles.png'  # Replace with your image path
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Flatten the image to a 1D array
    flat_image = image.flatten()

    # Calculate the frequency of each pixel value
    frequency = defaultdict(int)
    for pixel in flat_image:
        frequency[pixel] += 1

    # Sort symbols by frequency
    symbols = sorted(frequency.items(), key=lambda item: item[1], reverse=True)

    # Generate Shannon-Fano codes
    codes = {}
    shannon_fano_encode(symbols, codes)

    # Encode the image
    encoded_image = shannon_fano_encoding(flat_image, codes)
    print(f"Encoded Image (first 100 bits): {encoded_image[:100]}")

    # Decode the image
    decoded_image_flat = shannon_fano_decoding(encoded_image, codes)
    decoded_image = np.array(decoded_image_flat, dtype=np.uint8).reshape(image.shape)

    # Display the original and decoded images
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(decoded_image, cmap='gray')
    plt.title('Decoded Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
