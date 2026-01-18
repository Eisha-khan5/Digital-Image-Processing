import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('plant.jpg')
smoothed_image = cv2.GaussianBlur(image, (15, 15), 0) # Larger kernel size: 15x15
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2RGB))
plt.title('Smoothed Image')
plt.axis('off')
plt.show()