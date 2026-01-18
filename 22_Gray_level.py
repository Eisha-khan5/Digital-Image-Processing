import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def gray_level_slicing(image_path, lower_thresh, upper_thresh):
    # Read the input image
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    
    # Check if the image is loaded successfully
    if image is None:
        print("Error: Unable to load the image.")
        return
    
    # Apply gray level slicing
    result_image = np.copy(image)
    result_image[(image >= lower_thresh) & (image <= upper_thresh)] = 255
    result_image[(image < lower_thresh) | (image > upper_thresh)] = 0
    
    # Display the original and resultant images
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(result_image, cmap='gray')
    plt.title('Resultant Image')
    plt.axis('off')
    
    plt.show()

# Example usage
if __name__ == "__main__":
    image_path = 'cat.jpg'  # Change this to the path of your image
    lower_thresh = 100  # Lower threshold for gray level slicing
    upper_thresh = 200  # Upper threshold for gray level slicing
    gray_level_slicing(image_path, lower_thresh, upper_thresh)
