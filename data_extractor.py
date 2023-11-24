import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('Yasin (2022).png')

# Convert the image to RGB format
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define the RGB color ranges for red, green, and blue
red_lower = np.array([0, 0, 200])
red_upper = np.array([100, 100, 255])

green_lower = np.array([0, 200, 0])
green_upper = np.array([100, 255, 100])

blue_lower = np.array([200, 0, 0])
blue_upper = np.array([255, 100, 100])

# Create masks for each color
red_mask = cv2.inRange(image_rgb, red_lower, red_upper)
green_mask = cv2.inRange(image_rgb, green_lower, green_upper)
blue_mask = cv2.inRange(image_rgb, blue_lower, blue_upper)

# Find the contours in each mask
red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract data points
red_points = []
green_points = []
blue_points = []

for contour in red_contours:
    for point in contour:
        x, y = point[0]
        red_points.append((x, y))

for contour in green_contours:
    for point in contour:
        x, y = point[0]
        green_points.append((x, y))

for contour in blue_contours:
    for point in contour:
        x, y = point[0]
        blue_points.append((x, y))

# Visualize the extracted points
plt.figure(figsize=(8, 6))
plt.imshow(image_rgb)
plt.scatter(*zip(*red_points), color='red', s=5, label='Red Points')
plt.scatter(*zip(*green_points), color='green', s=5, label='Green Points')
plt.scatter(*zip(*blue_points), color='blue', s=5, label='Blue Points')
plt.legend()
plt.show()