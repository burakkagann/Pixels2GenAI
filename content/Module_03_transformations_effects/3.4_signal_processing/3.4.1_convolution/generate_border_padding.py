"""
Generate border_padding_explained.gif

Creates an animated diagram showing why padding matters in convolution:
  Frame 0: 5x5 image grid + intro text
  Frame 1: Kernel on center pixel (1,1) -- all neighbors available, OK
  Frame 2: Kernel on corner pixel (0,0) -- missing neighbors, X
  Frame 3: Same as Frame 2 + "Problem" explanation box
  Frame 4: 7x7 padded image (edge-replicate) + legend
  Frame 5: Kernel on corner of padded image -- all available, OK
  Frame 6: Summary with edge-replicate explanation + legend

Pixels2GenAI Project
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io
import os

# =============================================================================
# Configuration
# =============================================================================
FIG_W, FIG_H = 7.0, 4.2  # inches (slightly taller than original to avoid overlap)
DPI = 100

# Colors
COL_ORIGINAL = '#6A89B5'      # steel blue for original pixels
COL_ORIGINAL_LIGHT = '#B8CCE4' # lighter blue for un-highlighted original cells
COL_PADDING = '#C8A2C8'       # lavender for padding cells
COL_KERNEL_OK = '#2E8B57'     # sea green for valid kernel highlight
COL_KERNEL_BAD = '#CC3333'    # red for missing cells
COL_PROBLEM_BG = '#FFF0F0'    # light red for problem box
COL_PROBLEM_BORDER = '#CC3333'
COL_TEXT = '#2C3E50'           # dark slate for text
COL_OK = '#2E8B57'
COL_X = '#CC3333'

# 5x5 source image values
IMAGE_5x5 = np.array([
    [10, 20, 30, 40, 50],
    [15, 25, 35, 45, 55],
    [20, 30, 40, 50, 60],
    [25, 35, 45, 55, 65],
    [30, 40, 50, 60, 70],
])

# 7x7 padded version (edge-replicate)
PADDED_7x7 = np.pad(IMAGE_5x5, 1, mode='edge')

# Frame durations in milliseconds
DURATIONS = [660, 990, 990, 990, 1650, 1320, 1650]

# Grid geometry (in axes coordinates, computed for each frame type)
CELL_SIZE = 0.09   # each cell as fraction of axes width
TEXT_X = 0.65       # x-position where right-side text starts


# =============================================================================
# Drawing helpers
# =============================================================================
def draw_grid(ax, data, x0, y0, cell_w, cell_h, colors=None, show_border=True):
    """Draw a grid of cells with values.

    Args:
        ax: matplotlib axes
        data: 2D numpy array of values
        x0, y0: top-left corner in axes coordinates
        cell_w, cell_h: cell dimensions in axes coordinates
        colors: 2D array of color strings (same shape as data), or None for default
        show_border: draw outer border around entire grid
    """
    rows, cols = data.shape
    for r in range(rows):
        for c in range(cols):
            cx = x0 + c * cell_w
            cy = y0 - (r + 1) * cell_h  # y goes downward
            color = colors[r, c] if colors is not None else COL_ORIGINAL
            rect = patches.FancyBboxPatch(
                (cx, cy), cell_w, cell_h,
                boxstyle="round,pad=0.01",
                facecolor=color, edgecolor='white', linewidth=1.2,
                transform=ax.transAxes, clip_on=False
            )
            ax.add_patch(rect)
            # Cell text
            ax.text(
                cx + cell_w / 2, cy + cell_h / 2,
                str(int(data[r, c])),
                transform=ax.transAxes, ha='center', va='center',
                fontsize=8, color='#1a1a1a', fontweight='medium'
            )

    if show_border:
        border = patches.Rectangle(
            (x0, y0 - rows * cell_h), cols * cell_w, rows * cell_h,
            fill=False, edgecolor='#444444', linewidth=1.5,
            transform=ax.transAxes, clip_on=False
        )
        ax.add_patch(border)


def draw_missing_cells(ax, positions, x0, y0, cell_w, cell_h):
    """Draw red-dashed '?' cells at given (row, col) positions relative to grid."""
    for r, c in positions:
        cx = x0 + c * cell_w
        cy = y0 - (r + 1) * cell_h
        rect = patches.Rectangle(
            (cx, cy), cell_w, cell_h,
            fill=False, edgecolor=COL_KERNEL_BAD, linewidth=1.5,
            linestyle='--', transform=ax.transAxes, clip_on=False
        )
        ax.add_patch(rect)
        ax.text(
            cx + cell_w / 2, cy + cell_h / 2, '?',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=9, color=COL_KERNEL_BAD, fontweight='bold'
        )


def draw_kernel_highlight(ax, grid_x0, grid_y0, cell_w, cell_h,
                          kernel_row, kernel_col, color, linewidth=2.5):
    """Draw a 3x3 kernel outline on the grid."""
    kx = grid_x0 + kernel_col * cell_w
    ky = grid_y0 - (kernel_row + 3) * cell_h
    rect = patches.Rectangle(
        (kx, ky), 3 * cell_w, 3 * cell_h,
        fill=True, facecolor=color, alpha=0.25,
        edgecolor=color, linewidth=linewidth,
        transform=ax.transAxes, clip_on=False
    )
    ax.add_patch(rect)


def draw_legend_square(ax, x, y, color, label, size=0.025):
    """Draw a small colored square with a label."""
    rect = patches.Rectangle(
        (x, y), size, size,
        facecolor=color, edgecolor='#666666', linewidth=0.8,
        transform=ax.transAxes, clip_on=False
    )
    ax.add_patch(rect)
    ax.text(
        x + size + 0.015, y + size / 2, label,
        transform=ax.transAxes, ha='left', va='center',
        fontsize=8, color=COL_TEXT
    )


def fig_to_array(fig):
    """Convert matplotlib figure to numpy array."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI, bbox_inches='tight',
                pad_inches=0.05, facecolor='white', edgecolor='none')
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    return np.array(img)


def new_frame():
    """Create a fresh figure and axes."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig, ax


# =============================================================================
# 5x5 grid layout constants
# =============================================================================
G5_X0 = 0.06        # left edge
G5_Y0 = 0.82        # top edge (grid draws downward from here)
G5_CW = 0.085       # cell width
G5_CH = 0.12        # cell height
G5_SUBTITLE_Y = 0.85  # subtitle y (above grid top)


# =============================================================================
# Frame generators
# =============================================================================
def frame_0_intro():
    """5x5 grid + intro text."""
    fig, ax = new_frame()

    # Title
    ax.text(0.5, 0.97, 'Border Handling: Why Padding Matters',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=13, fontweight='bold', color=COL_TEXT)

    # Subtitle
    ax.text(G5_X0 + 2.5 * G5_CW, G5_SUBTITLE_Y, 'Image (5x5)',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=9.5, color=COL_TEXT, fontstyle='italic')

    # Grid
    draw_grid(ax, IMAGE_5x5, G5_X0, G5_Y0, G5_CW, G5_CH)

    # Right-side text
    ax.text(TEXT_X, 0.52, '3x3 kernel needs to visit\nevery pixel position...',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=9.5, color='#555555', fontstyle='italic')

    arr = fig_to_array(fig)
    plt.close(fig)
    return arr


def frame_1_center_ok():
    """Kernel on center (1,1) -- all neighbors available."""
    fig, ax = new_frame()

    ax.text(0.5, 0.97, 'Border Handling: Why Padding Matters',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=13, fontweight='bold', color=COL_TEXT)

    ax.text(G5_X0 + 2.5 * G5_CW, G5_SUBTITLE_Y, 'Image (5x5)',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=9.5, color=COL_TEXT, fontstyle='italic')

    draw_grid(ax, IMAGE_5x5, G5_X0, G5_Y0, G5_CW, G5_CH)

    # Kernel highlight on center (row=1, col=1) means 3x3 starting at (0,0)
    draw_kernel_highlight(ax, G5_X0, G5_Y0, G5_CW, G5_CH, 0, 0, COL_KERNEL_OK)

    # Right-side text
    ax.text(TEXT_X, 0.62, 'Center pixel (1,1)',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=11, fontweight='bold', color=COL_TEXT)
    ax.text(TEXT_X, 0.53, 'All 9 neighbors\navailable',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=9, color='#555555')
    ax.text(TEXT_X, 0.36, 'OK',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=28, fontweight='bold', color=COL_OK)

    arr = fig_to_array(fig)
    plt.close(fig)
    return arr


def frame_2_corner_problem():
    """Kernel on corner (0,0) -- missing neighbors."""
    fig, ax = new_frame()

    ax.text(0.5, 0.97, 'Border Handling: Why Padding Matters',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=13, fontweight='bold', color=COL_TEXT)

    # For this frame, the grid shifts down to make room for "?" cells above
    shift_down = G5_CH  # one cell height down
    g_y0 = G5_Y0 - shift_down

    # Subtitle must be above the "?" cells, which are 1 row above shifted grid
    # "?" top = g_y0 (the shifted grid top), "?" cells extend to g_y0 + G5_CH
    # So subtitle needs to be above g_y0 + G5_CH
    subtitle_y = g_y0 + G5_CH + 0.03
    ax.text(G5_X0 + 2.5 * G5_CW, subtitle_y, 'Image (5x5)',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=9.5, color=COL_TEXT, fontstyle='italic')

    draw_grid(ax, IMAGE_5x5, G5_X0, g_y0, G5_CW, G5_CH)

    # Kernel highlight on corner (0,0): 3x3 starts at (-1,-1)
    # Only the inner 2x2 portion (rows 0-1, cols 0-1) is on the grid
    draw_kernel_highlight(ax, G5_X0, g_y0, G5_CW, G5_CH, -1, -1, COL_KERNEL_OK)

    # Missing "?" cells: row -1 (col -1,0,1) and row 0,1 (col -1)
    missing_positions = [
        (-1, -1), (-1, 0), (-1, 1),  # top row of kernel (above grid)
        (0, -1),                       # left of row 0
        (1, -1),                       # left of row 1
    ]
    draw_missing_cells(ax, missing_positions, G5_X0, g_y0, G5_CW, G5_CH)

    # Right-side text
    ax.text(TEXT_X, 0.62, 'Corner pixel (0,0)',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=11, fontweight='bold', color=COL_TEXT)
    ax.text(TEXT_X, 0.52, '5 neighbors missing!',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=9, color=COL_X)
    ax.text(TEXT_X, 0.35, 'X',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=32, fontweight='bold', color=COL_X)

    arr = fig_to_array(fig)
    plt.close(fig)
    return arr


def frame_3_problem_box():
    """Same as frame 2 + problem explanation box."""
    fig, ax = new_frame()

    ax.text(0.5, 0.97, 'Border Handling: Why Padding Matters',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=13, fontweight='bold', color=COL_TEXT)

    shift_down = G5_CH
    g_y0 = G5_Y0 - shift_down

    subtitle_y = g_y0 + G5_CH + 0.03
    ax.text(G5_X0 + 2.5 * G5_CW, subtitle_y, 'Image (5x5)',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=9.5, color=COL_TEXT, fontstyle='italic')

    draw_grid(ax, IMAGE_5x5, G5_X0, g_y0, G5_CW, G5_CH)
    draw_kernel_highlight(ax, G5_X0, g_y0, G5_CW, G5_CH, -1, -1, COL_KERNEL_OK)

    missing_positions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),
        (1, -1),
    ]
    draw_missing_cells(ax, missing_positions, G5_X0, g_y0, G5_CW, G5_CH)

    # Problem box on the right
    box_x, box_y, box_w, box_h = 0.54, 0.35, 0.38, 0.30
    problem_box = patches.FancyBboxPatch(
        (box_x, box_y), box_w, box_h,
        boxstyle="round,pad=0.02",
        facecolor=COL_PROBLEM_BG, edgecolor=COL_PROBLEM_BORDER,
        linewidth=1.5, transform=ax.transAxes, clip_on=False
    )
    ax.add_patch(problem_box)

    ax.text(box_x + box_w / 2, box_y + box_h - 0.04, 'Problem:',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=11, fontweight='bold', color=COL_X)
    ax.text(box_x + box_w / 2, box_y + box_h / 2 - 0.02,
            'Cannot compute weighted sum\nwith missing values.',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=8.5, color=COL_TEXT, fontweight='medium')
    ax.text(box_x + box_w / 2, box_y + 0.04,
            'Output shrinks or has artifacts.',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=8, color='#777777', fontstyle='italic')

    arr = fig_to_array(fig)
    plt.close(fig)
    return arr


# =============================================================================
# 7x7 padded grid layout constants
# =============================================================================
G7_X0 = 0.04        # left edge (wider grid needs to start further left)
G7_CW = 0.065       # smaller cells for 7 columns
G7_CH = 0.10        # smaller cells for 7 rows
G7_Y0 = 0.84        # top edge of padded grid (leaving room for subtitle above)
G7_SUBTITLE_Y = 0.87  # subtitle well above grid top


def _build_padded_colors():
    """Create color array: original=blue, padding=lavender."""
    colors = np.full((7, 7), COL_PADDING, dtype=object)
    colors[1:6, 1:6] = COL_ORIGINAL
    return colors


def frame_4_padded_grid():
    """7x7 padded grid with legend."""
    fig, ax = new_frame()

    ax.text(0.5, 0.97, 'Border Handling: Why Padding Matters',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=13, fontweight='bold', color=COL_TEXT)

    ax.text(G7_X0 + 3.5 * G7_CW, G7_SUBTITLE_Y, 'Padded Image (7x7)',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=9.5, color=COL_TEXT, fontstyle='italic')

    colors = _build_padded_colors()
    draw_grid(ax, PADDED_7x7, G7_X0, G7_Y0, G7_CW, G7_CH, colors=colors)

    # Original pixels inner border
    inner_border = patches.Rectangle(
        (G7_X0 + G7_CW, G7_Y0 - 6 * G7_CH),
        5 * G7_CW, 5 * G7_CH,
        fill=False, edgecolor='#444444', linewidth=1.5,
        transform=ax.transAxes, clip_on=False
    )
    ax.add_patch(inner_border)

    # Legend
    draw_legend_square(ax, TEXT_X - 0.05, 0.58, COL_ORIGINAL, 'Original pixels')
    draw_legend_square(ax, TEXT_X - 0.05, 0.52, COL_PADDING, 'Padding (edge-replicate)')

    ax.text(TEXT_X + 0.05, 0.42, 'Border values copied outward',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=9, color=COL_OK, fontstyle='italic')

    ax.text(TEXT_X + 0.05, 0.20, "np.pad(image, 1, mode='edge')",
            transform=ax.transAxes, ha='center', va='center',
            fontsize=8, color='#777777', family='monospace')

    arr = fig_to_array(fig)
    plt.close(fig)
    return arr


def frame_5_padded_corner_ok():
    """Kernel on corner of original (which is (1,1) of padded) -- all available."""
    fig, ax = new_frame()

    ax.text(0.5, 0.97, 'Border Handling: Why Padding Matters',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=13, fontweight='bold', color=COL_TEXT)

    ax.text(G7_X0 + 3.5 * G7_CW, G7_SUBTITLE_Y, 'Padded Image (7x7)',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=9.5, color=COL_TEXT, fontstyle='italic')

    colors = _build_padded_colors()
    draw_grid(ax, PADDED_7x7, G7_X0, G7_Y0, G7_CW, G7_CH, colors=colors)

    # Inner border
    inner_border = patches.Rectangle(
        (G7_X0 + G7_CW, G7_Y0 - 6 * G7_CH),
        5 * G7_CW, 5 * G7_CH,
        fill=False, edgecolor='#444444', linewidth=1.5,
        transform=ax.transAxes, clip_on=False
    )
    ax.add_patch(inner_border)

    # Kernel on padded (0,0) which covers original corner pixel (0,0) = padded (1,1)
    draw_kernel_highlight(ax, G7_X0, G7_Y0, G7_CW, G7_CH, 0, 0, COL_KERNEL_OK)

    # Right-side text
    ax.text(TEXT_X + 0.05, 0.62, 'Corner pixel (0,0)',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=11, fontweight='bold', color=COL_TEXT)
    ax.text(TEXT_X + 0.05, 0.52, 'All 9 values available!',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=9, color=COL_OK)
    ax.text(TEXT_X + 0.05, 0.37, 'OK',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=28, fontweight='bold', color=COL_OK)
    ax.text(TEXT_X + 0.05, 0.22, 'Output size = Input size',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=9, fontweight='bold', color=COL_TEXT)

    arr = fig_to_array(fig)
    plt.close(fig)
    return arr


def frame_6_summary():
    """Summary with explanation + legend."""
    fig, ax = new_frame()

    ax.text(0.5, 0.97, 'Border Handling: Why Padding Matters',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=13, fontweight='bold', color=COL_TEXT)

    ax.text(G7_X0 + 3.5 * G7_CW, G7_SUBTITLE_Y, 'Padded Image (7x7)',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=9.5, color=COL_TEXT, fontstyle='italic')

    colors = _build_padded_colors()
    draw_grid(ax, PADDED_7x7, G7_X0, G7_Y0, G7_CW, G7_CH, colors=colors)

    # Inner border
    inner_border = patches.Rectangle(
        (G7_X0 + G7_CW, G7_Y0 - 6 * G7_CH),
        5 * G7_CW, 5 * G7_CH,
        fill=False, edgecolor='#444444', linewidth=1.5,
        transform=ax.transAxes, clip_on=False
    )
    ax.add_patch(inner_border)

    # Kernel on corner to show it works
    draw_kernel_highlight(ax, G7_X0, G7_Y0, G7_CW, G7_CH, 0, 0, COL_KERNEL_OK)

    # Right-side summary
    ax.text(TEXT_X + 0.05, 0.68, 'Edge-replicate padding',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=10.5, fontweight='bold', color=COL_OK)
    ax.text(TEXT_X + 0.05, 0.55,
            'copies border pixels outward\nso every pixel has a\ncomplete neighborhood.',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=8.5, color=COL_TEXT)

    # Legend
    draw_legend_square(ax, TEXT_X - 0.05, 0.38, COL_ORIGINAL, 'Original')
    draw_legend_square(ax, TEXT_X - 0.05, 0.32, COL_PADDING, 'Padding')

    ax.text(TEXT_X + 0.05, 0.20, "np.pad(image, 1, mode='edge')",
            transform=ax.transAxes, ha='center', va='center',
            fontsize=8, color='#777777', family='monospace')

    arr = fig_to_array(fig)
    plt.close(fig)
    return arr


# =============================================================================
# Main: generate all frames and save GIF
# =============================================================================
if __name__ == '__main__':
    frame_funcs = [
        frame_0_intro,
        frame_1_center_ok,
        frame_2_corner_problem,
        frame_3_problem_box,
        frame_4_padded_grid,
        frame_5_padded_corner_ok,
        frame_6_summary,
    ]

    frames = []
    for i, func in enumerate(frame_funcs):
        print(f'  Generating frame {i}: {func.__name__}...')
        arr = func()
        frames.append(Image.fromarray(arr))

    # Save as animated GIF into visuals/ subdirectory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visuals')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'border_padding_explained.gif')

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=DURATIONS,
        loop=0,
    )

    file_size = os.path.getsize(output_path)
    print(f'\nSaved {len(frames)} frames to {output_path}')
    print(f'File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)')
