"""
Exercise 2: Modify to Achieve Goals — Color Mixing

Modify the channel values below to create each target color.
Run the script after each change to see your result.

Goals (see README for full descriptions):
  Goal 1: Create pure cyan
  Goal 2: Create a medium gray
  Goal 3: Create a dark purple
"""

import numpy as np
from PIL import Image

# Create a 150x150 image (starts as black)
image = np.zeros((150, 150, 3), dtype=np.uint8)

# =============================================
# MODIFY the values below to achieve each goal
# (see README for goal descriptions)
# =============================================
image[:, :, 0] = 255  # Red channel   
image[:, :, 1] = 128  # Green channel 
image[:, :, 2] = 0    # Blue channel   
# =============================================

# Save and display the result
result = Image.fromarray(image, mode='RGB')
result.save('exercise2_color.png')
print("Saved exercise2_color.png")
print(f"Your color: R={image[0, 0, 0]}, G={image[0, 0, 1]}, B={image[0, 0, 2]}")
