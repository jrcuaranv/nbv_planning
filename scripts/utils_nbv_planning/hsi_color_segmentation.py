import cv2
import numpy as np

def get_semantic_image(image, add_seg_noise = False):
    # Convert BGR to HSI
    hsi_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for red hues (fruits)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([179, 255, 255])

    # Threshold the HSV image to get only red hues
    mask1 = cv2.inRange(hsi_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsi_image, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask1, mask2)

    # Define lower and upper bounds for green hues (leaves)
    lower_green = np.array([40, 20, 20])  # Adjust these values based on the specific green hues in your images
    upper_green = np.array([80, 255, 255])  # Adjust these values based on the specific green hues in your images

    # Threshold the HSV image to get only green hues
    mask_green = cv2.inRange(hsi_image, lower_green, upper_green)

    green_color = (0, 255, 0)  # BGR color format: (B, G, R)
    red_color = (0, 0, 255)  # BGR color format: (B, G, R)
    
    # Adding noise to segmentation masks
    
    if (add_seg_noise):
        aux1 = np.random.sample()
        if (0.7 < aux1 and aux1 < 0.85):
            green_color = (0, 0, 255) # red
        if (0.85 < aux1 and aux1 < 1.0):
            green_color = (0, 0, 0) # black

        aux2 = np.random.sample()
        if (0.7 < aux2 and aux2 < 0.85):
            red_color = (0, 255, 0) # green
        if (0.85 < aux2 and aux2 < 1.0):
            red_color = (0, 0, 0) # black

    green_image = np.full(image.shape, green_color, dtype=np.uint8)
    red_image = np.full(image.shape, red_color, dtype=np.uint8)

    semantic_image = cv2.bitwise_and(green_image, green_image, mask=mask_green) + cv2.bitwise_and(red_image, red_image, mask=mask_red)
    # return image_bgr8
    return semantic_image.astype(np.uint8), mask_red, mask_green
