import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop, heapify
from collections import defaultdict

# Function to build the Huffman tree and generate codes
def build_huffman_tree(data):
    frequency = defaultdict(int)
    for pixel in data:
        frequency[pixel] += 1

    heap = [[weight, [symbol, ""]] for symbol, weight in frequency.items()]
    heapify(heap)

    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    return sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

# Function to encode data using the Huffman codes
def huffman_encode(data, huffman_code):
    encoding = ""
    for pixel in data:
        encoding += huffman_code[pixel]
    return encoding

# Function to decode data using the Huffman codes
def huffman_decode(encoded_data, huffman_code):
    reverse_huffman_code = {v: k for k, v in huffman_code.items()}
    current_code = ""
    decoded_data = []

    for bit in encoded_data:
        current_code += bit
        if current_code in reverse_huffman_code:
            decoded_data.append(reverse_huffman_code[current_code])
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

    # Build the Huffman tree and get the codes
    huffman_tree = build_huffman_tree(flat_image)
    huffman_code = {item[0]: item[1] for item in huffman_tree}

    # Encode the image
    encoded_image = huffman_encode(flat_image, huffman_code)

    # Print a portion of the encoded image
    print(f"Encoded Image (first 100 bits): {encoded_image[:100]}")

    # Decode the image
    decoded_image_flat = huffman_decode(encoded_image, huffman_code)
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
