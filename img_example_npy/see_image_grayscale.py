import numpy as np
import cv2

# Load the color frame
color = np.load('color_frame_0.npy')

# Convert to grayscale
gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

# Now `gray` is a 2D matrix (480x640)
print("Grayscale matrix shape:", gray.shape)
print(gray)  # This will print the entire matrix

# Optional: Display it as an image
cv2.imshow('Grayscale Frame', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

