import numpy as np
import cv2

depth = np.load('depth_frame_0.npy')

# Convert to 8-bit for display (normalize with alpha)
depth_visual = cv2.convertScaleAbs(depth, alpha=0.03)
depth_colormap = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)

# Now `depth` is a 2D matrix (480x640,3)
print("depth matrix shape:", depth_colormap.shape)
print(depth_colormap)  # This will print the entire matrix

cv2.imshow('Depth Colormap', depth_colormap)
cv2.waitKey(0)
cv2.destroyAllWindows()

