# graph.py
import numpy as np
from detection import generate_crowd_map

# Normalize the grid to [0, 1] so that costs are comparable and one large value doesn't dominate
def normalize_grid(grid):
    """Normalize grid values to [0,1] for fair cost calculation."""
    max_val = np.max(grid)
    if max_val == 0:
        return grid
    return grid / max_val