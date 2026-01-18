import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image_path = "cat.jpg"  # Provide the path to your image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded successfully
if image is None:
    print("Error: Unable to load the image.")
else:
    # Apply Sobel filter
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute the gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize the gradient magnitude to the range [0, 255]
    gradient_magnitude_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Display the original and filtered images
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(gradient_magnitude_normalized, cmap='gray')
    plt.title('Gradient Magnitude')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(sobel_x, cmap='gray')
    plt.title('Sobel X')
    plt.axis('off')
    
    plt.show()
