import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Function to perform LZW encoding
def lzw_encode(data):
    dictionary = {chr(i): i for i in range(256)}
    p = ""
    code = 256
    encoded_data = []
    
    for c in data:
        pc = p + c
        if pc in dictionary:
            p = pc
        else:
            encoded_data.append(dictionary[p])
            dictionary[pc] = code
            code += 1
            p = c
    if p:
        encoded_data.append(dictionary[p])
    
    return encoded_data

# Function to perform LZW decoding
def lzw_decode(encoded_data):
    dictionary = {i: chr(i) for i in range(256)}
    code = 256
    p = chr(encoded_data.pop(0))
    decoded_data = [p]
    
    for k in encoded_data:
        if k in dictionary:
            entry = dictionary[k]
        elif k == code:
            entry = p + p[0]
        decoded_data.append(entry)
        dictionary[code] = p + entry[0]
        code += 1
        p = entry
    
    return ''.join(decoded_data)

def main():
    # Load an image from file
    image_path = 'circles.png'  # Replace with your image path
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Flatten the image to a 1D array
    flat_image = image.flatten()
    flat_image_str = ''.join([chr(pixel) for pixel in flat_image])

    # Encode the image using LZW
    encoded_image = lzw_encode(flat_image_str)
    print(f"Encoded Image (first 100 codes): {encoded_image[:100]}")

    # Decode the image using LZW
    decoded_image_str = lzw_decode(encoded_image)
    decoded_image_flat = [ord(char) for char in decoded_image_str]
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
