#Consider an image with 2 objects and a total of 3-pixel values (1 for each object and one 
#for the background). Add Gaussian noise to the image. Implement and test Otsu’s 
#algorithm with this image. 

import numpy as np
import cv2

#Creating image with 2 objects and total of 3 pixel value

# Image dimensions
height = 200
width = 200

#Blank image with all pixels initialized to the background value
image = np.zeros((height, width), dtype=np.uint8)

# Define the positions and shapes of the two objects
#Object 1
object1_position = (20, 20)
object1_shape = (50, 50) 

#Object 2
object2_position = (100, 100)
object2_shape = (45, 45)  

# First object- Rectangle
cv2.rectangle(image, object1_position, (object1_position[0]+object1_shape[0], object1_position[1]+object1_shape[1]), 255, -1)

# Second object - Circle
cv2.circle(image, object2_position, int(object2_shape[0]/2), 255, -1)

# Adding gaussian noise to the image
mean = 0
std_dev = 25
noise = np.zeros(image.shape, dtype=np.uint8)
cv2.randn(noise, mean, std_dev)
noisy_image = cv2.add(image, noise)

# Apply Otsu's thresholding
_, otsu_threshold_image = cv2.threshold(noisy_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Show the image
cv2.imshow("Image", image)
cv2.imshow("Gaussian Noisy Image", noisy_image)
cv2.imshow("Otsu Thresholded Image", otsu_threshold_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
