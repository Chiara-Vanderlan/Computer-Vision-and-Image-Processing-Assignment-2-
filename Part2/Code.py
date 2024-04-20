#Implement a region-growing technique for image segmentation. The basic idea is to start 
#from a set of points inside the object of interest (foreground), denoted as seeds, and 
#recursively add neighboring pixels as long as they are in a pre-defined range of the pixel 
#values of the seeds. 

import cv2
import numpy as np

#Function for region growing  
def process(image, seeds, threshold):
    height, width = image.shape
    segmented = np.zeros_like(image)
    visited = np.zeros_like(image, dtype=bool)
    stack = []

    # Define 4-connectivity
    connectivity = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    for seed in seeds:
        stack.append(seed)

    while stack:
        current_pixel = stack.pop()
        segmented[current_pixel] = 255

        for dx, dy in connectivity:
            x, y = current_pixel[0] + dx, current_pixel[1] + dy

            if 0 <= x < height and 0 <= y < width and not visited[x, y]:
                if abs(int(image[x, y]) - int(image[current_pixel])) < threshold:
                    stack.append((x, y))
                    visited[x, y] = True

    return segmented

if __name__ == "__main__":

    # Read the image
    image = cv2.imread('Part2\dog.jpg', cv2.IMREAD_GRAYSCALE)

    cv2.imshow('Original Image', image)

    #Define the seed
    seeds = [(100, 100), (250, 250)]
    #Define threhold for region growing
    threshold = 15
    #Performing the processing funtion
    segmented_image = process(image, seeds, threshold)

    cv2.imshow('Segmented Image', segmented_image)
    cv2.imwrite('Results\segmented_image.jpg', segmented_image)

    #Define the seed
    seeds_2 = [(50, 50),(100, 100), (250, 250),(150, 150)]
    #Define threhold for region growing
    threshold_2 = 11
    #Performing the processing funtion
    segmented_image_2 = process(image, seeds_2, threshold_2)

    cv2.imshow('Segmented Image 2', segmented_image_2)
    cv2.imwrite('Results\segmented_image_2.jpg', segmented_image_2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
