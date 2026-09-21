import numpy as np
from PIL import Image

# 1. Load the image file
img = Image.open("blackandwhite.jpg")

# 2. Convert the visual image into a raw math grid (NumPy Array)
pixel_grid = np.array(img)

# 3. Check the internal structure: (Height, Width, Color Channels)
print("Image Grid Shape:", pixel_grid.shape) 
# Output example: (100, 100, 3) -> 100 high, 100 wide, 3 colors deep (RGB)

# 4. Look closely at the very top-left single pixel [Row 0, Column 0]
top_left_pixel = pixel_grid[200, 200]
print("Top-Left Pixel RGB Values:", top_left_pixel)
# Output example: [255, 0, 0] 
