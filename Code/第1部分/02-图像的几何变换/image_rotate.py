import cv2
import numpy as np

# Load the image
img = cv2.imread('lena.jpg')

# Resize the image
resized = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation = cv2.INTER_LINEAR)

# Get the shape of the image
rows, cols = resized.shape[:2]

# Compute the center of the image
center = (cols / 2, rows / 2)

# Create the rotation matrix
rot = cv2.getRotationMatrix2D(center, -30, 1)

# Apply the rotation to the image
rotated = cv2.warpAffine(resized, rot, (cols, rows))

# Create the translation matrix
trans = np.float32([[1, 0, 50], [0, 1, 0]])

# Apply the translation to the image
translated = cv2.warpAffine(rotated, trans, (cols, rows))

# Save the result
cv2.imwrite('result.jpg', translated)

# Display the original image and the result
cv2.imshow('Original', img)
cv2.imshow('Result', translated)
cv2.waitKey(0)
cv2.destroyAllWindows()
