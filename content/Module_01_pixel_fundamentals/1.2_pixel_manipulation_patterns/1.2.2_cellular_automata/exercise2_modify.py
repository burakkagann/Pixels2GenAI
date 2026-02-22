"""
Exercise 2: Modify Blinker Variations

Change the marked values below to achieve each goal.
Run the script after each change to see how the output differs.

Goals (see README for full descriptions):
  Goal 1: Move the blinker to the very top edge (row 0)
  Goal 2: Keep the edge blinker and switch to fixed boundary
  Goal 3: Create multiple blinkers at different positions
"""

import numpy as np
from PIL import Image
from scipy.ndimage import convolve


def grid_to_image(grid, scale=10):
    """Convert binary grid to RGB image."""
    gray = np.repeat(np.repeat(grid * 255, scale, axis=0), scale, axis=1)
    return np.stack([gray, gray, gray], axis=2).astype(np.uint8)


def game_of_life_step(grid):
    """Apply one generation of Conway's Game of Life rules."""
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    # =============================================
    # MODIFY the boundary mode for Goal 2
    # =============================================
    neighbor_count = convolve(grid, kernel, mode='wrap')
    # =============================================
    return ((neighbor_count == 3) | ((grid == 1) & (neighbor_count == 2))).astype(int)


grid = np.zeros((30, 30), dtype=int)

# =============================================
# MODIFY the pattern placement below
# =============================================
grid[15, 14:17] = [1, 1, 1] 
# =============================================

# Evolve for 6 generations
for generation in range(6):
    grid = game_of_life_step(grid)
    print(f"Generation {generation + 1}: {np.sum(grid)} living cells")

# Save result
Image.fromarray(grid_to_image(grid)).save('exercise2_result.png')
print("Saved exercise2_result.png")
