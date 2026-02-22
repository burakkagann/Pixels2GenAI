"""
ControlNet Morphing Animation Generator

Creates a smooth morphing animation showing color/pattern evolution within a fixed
circle structure using spherical linear interpolation (SLERP) in latent space.

The animation demonstrates how varying the initial latent while keeping the same
conditioning image produces diverse but structurally consistent outputs. The circle
structure remains fixed while colors smoothly transition between patterns.

Usage:
    python generate_controlnet_morph.py

Requirements:
    - diffusers library
    - Pretrained ControlNet model (downloaded automatically from Hugging Face)
    - GPU recommended (8GB+ VRAM)

Author: NumPy-to-GenAI Project
"""

import os
import sys
from pathlib import Path

# Check for required libraries
try:
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    import torch
except ImportError:
    print("Error: Required libraries not found.")
    print("Please install with: pip install diffusers transformers accelerate torch")
    sys.exit(1)

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio


# =============================================================================
# Configuration
# =============================================================================

# Animation parameters
NUM_KEYFRAMES = 4           # Number of distinct color patterns to morph between
DURATION_SECONDS = 8        # Total animation duration
FPS = 15                    # Frames per second
TOTAL_FRAMES = 120          # 8 seconds * 15 fps

# Generation parameters
IMAGE_SIZE = 512            # Output resolution
NUM_INFERENCE_STEPS = 25    # Diffusion sampling steps
GUIDANCE_SCALE = 7.5        # Text guidance strength
CONTROLNET_SCALE = 0.8      # How strictly to follow the control image

# Prompts
PROMPT = "a beautiful colorful circle filled with smooth gradient colors, abstract art"
NEGATIVE_PROMPT = "blurry, low quality, distorted, text, watermark"

# Random seed for reproducible keyframes
RANDOM_SEED = 42

# Output paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_GIF = SCRIPT_DIR / 'controlnet_circle_morph.gif'
OUTPUT_KEYFRAMES = SCRIPT_DIR / 'controlnet_morph_keyframes.png'
OUTPUT_CONTROL = SCRIPT_DIR / 'controlnet_morph_control.png'

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# SLERP Interpolation (adapted from DDPM module)
# =============================================================================

def slerp(val, low, high):
    """
    Spherical linear interpolation between two tensors.

    SLERP produces smoother interpolations than linear interpolation
    for high-dimensional vectors like latents, as it interpolates along
    the surface of a hypersphere rather than cutting through it.

    Parameters:
        val: Interpolation factor (0.0 = low, 1.0 = high)
        low: Starting tensor
        high: Ending tensor

    Returns:
        Interpolated tensor
    """
    # Flatten tensors for dot product
    low_flat = low.flatten()
    high_flat = high.flatten()

    # Normalize vectors
    low_norm = low_flat / torch.norm(low_flat)
    high_norm = high_flat / torch.norm(high_flat)

    # Compute angle between vectors
    dot = torch.clamp(torch.dot(low_norm, high_norm), -1.0, 1.0)
    omega = torch.acos(dot)

    # Handle near-parallel vectors (fall back to linear interpolation)
    if torch.abs(omega) < 1e-10:
        return low * (1.0 - val) + high * val

    # SLERP formula
    so = torch.sin(omega)
    result = (torch.sin((1.0 - val) * omega) / so) * low_flat + \
             (torch.sin(val * omega) / so) * high_flat

    return result.view(low.shape)


def lerp(val, low, high):
    """Simple linear interpolation (fallback)."""
    return low * (1.0 - val) + high * val


# =============================================================================
# Control Image Generation
# =============================================================================

def create_circle_control(size=512, radius_ratio=0.35, line_width=3):
    """
    Create a circle outline control image for ControlNet.

    This creates a simple circle outline that matches the conditioning
    format expected by ControlNet (white edges on black background).

    Parameters:
        size: Image dimensions (square)
        radius_ratio: Circle radius as fraction of image size
        line_width: Width of the circle outline

    Returns:
        PIL.Image: Circle outline image
    """
    # Create black background
    img = Image.new('RGB', (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Calculate circle parameters
    center = size // 2
    radius = int(size * radius_ratio)

    # Draw white circle outline
    bbox = [
        center - radius,
        center - radius,
        center + radius,
        center + radius
    ]
    draw.ellipse(bbox, outline=(255, 255, 255), width=line_width)

    return img


# =============================================================================
# Pipeline Loading
# =============================================================================

def load_controlnet_pipeline():
    """
    Load the ControlNet pipeline with Stable Diffusion.

    Uses the Canny ControlNet model which follows edge structures.
    The pipeline is configured for optimal memory usage on GPU.

    Returns:
        StableDiffusionControlNetPipeline: Loaded pipeline
    """
    print("Loading ControlNet (Canny) model...")
    print("This may take a few minutes on first run as models are downloaded.")

    # Use appropriate dtype
    dtype = torch.float16 if DEVICE == 'cuda' else torch.float32

    # Load Canny ControlNet
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=dtype
    )

    # Load Stable Diffusion pipeline with ControlNet
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None  # Disable for educational use
    )

    pipeline = pipeline.to(DEVICE)

    # Enable memory optimizations if available
    if DEVICE == 'cuda':
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory optimization")
        except Exception:
            pass

    print("Pipeline loaded successfully!")
    return pipeline


# =============================================================================
# Latent Generation and Interpolation
# =============================================================================

def generate_keyframe_latents(num_keyframes, shape, seed=None):
    """
    Generate random latent vectors for keyframes.

    Parameters:
        num_keyframes: Number of keyframe latents to generate
        shape: Shape of each latent tensor [1, C, H, W]
        seed: Random seed for reproducibility

    Returns:
        List of latent tensors
    """
    if seed is not None:
        torch.manual_seed(seed)

    latents = [torch.randn(shape, dtype=torch.float16 if DEVICE == 'cuda' else torch.float32)
               for _ in range(num_keyframes)]
    return latents


def interpolate_latents(latent1, latent2, num_steps):
    """
    Generate SLERP-interpolated latents between two keyframes.

    Parameters:
        latent1: Starting latent
        latent2: Ending latent
        num_steps: Number of interpolation steps

    Returns:
        List of interpolated latent tensors
    """
    interpolated = []
    for i in range(num_steps):
        t = i / num_steps  # Goes from 0 to just before 1
        interp_latent = slerp(t, latent1, latent2)
        interpolated.append(interp_latent)
    return interpolated


# =============================================================================
# Frame Generation
# =============================================================================

@torch.no_grad()
def generate_frame(pipeline, control_image, latent, prompt, negative_prompt):
    """
    Generate a single frame from a latent and control image.

    Parameters:
        pipeline: ControlNet pipeline
        control_image: Conditioning image (circle outline)
        latent: Starting latent tensor
        prompt: Text prompt
        negative_prompt: Negative prompt

    Returns:
        PIL.Image: Generated image
    """
    # Move latent to device
    latent = latent.to(DEVICE)

    # Generate image with custom latent
    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image,
        latents=latent,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        controlnet_conditioning_scale=CONTROLNET_SCALE,
    )

    return result.images[0]


# =============================================================================
# GIF Creation
# =============================================================================

def create_morphing_gif(frames, output_path, fps):
    """
    Create a smooth GIF from frames.

    Parameters:
        frames: List of PIL Images or numpy arrays
        output_path: Output file path
        fps: Frames per second
    """
    print(f"\nCreating GIF with {len(frames)} frames at {fps} FPS...")

    # Convert PIL images to numpy arrays if needed
    frame_arrays = []
    for frame in frames:
        if isinstance(frame, Image.Image):
            frame_arrays.append(np.array(frame))
        else:
            frame_arrays.append(frame)

    # Save GIF
    imageio.mimsave(
        str(output_path),
        frame_arrays,
        fps=fps,
        loop=0  # Infinite loop
    )

    # Get file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved to: {output_path}")
    print(f"File size: {file_size:.2f} MB")
    print(f"Duration: {len(frames) / fps:.1f} seconds")


def create_keyframe_grid(keyframe_images, output_path):
    """
    Create a grid showing all keyframe images.

    Parameters:
        keyframe_images: List of PIL Images
        output_path: Output file path
    """
    import matplotlib.pyplot as plt

    n_keyframes = len(keyframe_images)
    fig, axes = plt.subplots(1, n_keyframes, figsize=(4 * n_keyframes, 4), dpi=150)

    for i, img in enumerate(keyframe_images):
        axes[i].imshow(img)
        axes[i].set_title(f'Keyframe {i + 1}', fontsize=12, fontweight='bold')
        axes[i].axis('off')

    plt.suptitle('ControlNet Morphing Keyframes', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"Keyframe grid saved to: {output_path}")


# =============================================================================
# Main Generation
# =============================================================================

def generate_morph_animation():
    """Main function to generate the morphing animation."""
    print("=" * 60)
    print("ControlNet Circle Morphing Animation Generator")
    print("=" * 60)
    print()

    # Device info
    print(f"Device: {DEVICE}")
    if DEVICE == 'cpu':
        print("Warning: Generation on CPU will be very slow!")
        print("Consider using a GPU for faster generation.")
    print()

    # Calculate frame distribution
    frames_per_transition = TOTAL_FRAMES // NUM_KEYFRAMES
    actual_total_frames = frames_per_transition * NUM_KEYFRAMES

    print("Animation parameters:")
    print(f"  - Duration: {DURATION_SECONDS} seconds")
    print(f"  - FPS: {FPS}")
    print(f"  - Keyframes: {NUM_KEYFRAMES}")
    print(f"  - Frames per transition: {frames_per_transition}")
    print(f"  - Total frames: {actual_total_frames}")
    print(f"  - Inference steps per frame: {NUM_INFERENCE_STEPS}")
    print()

    # Create control image
    print("Creating circle control image...")
    control_image = create_circle_control(size=IMAGE_SIZE, radius_ratio=0.35)
    control_image.save(OUTPUT_CONTROL)
    print(f"Control image saved to: {OUTPUT_CONTROL}")
    print()

    # Load pipeline
    pipeline = load_controlnet_pipeline()
    print()

    # Get latent shape for SD v1.5 (512x512 image -> 64x64 latent)
    latent_height = IMAGE_SIZE // 8
    latent_width = IMAGE_SIZE // 8
    latent_shape = (1, 4, latent_height, latent_width)  # SD v1.5 has 4 latent channels

    # Generate keyframe latents
    print(f"Generating {NUM_KEYFRAMES} keyframe latents (seed={RANDOM_SEED})...")
    keyframe_latents = generate_keyframe_latents(NUM_KEYFRAMES, latent_shape, seed=RANDOM_SEED)

    # Generate keyframe images first (for preview grid)
    print("Generating keyframe images...")
    keyframe_images = []
    for i, latent in enumerate(keyframe_latents):
        print(f"  Keyframe {i + 1}/{NUM_KEYFRAMES}")
        img = generate_frame(pipeline, control_image, latent, PROMPT, NEGATIVE_PROMPT)
        keyframe_images.append(img)

    # Save keyframe grid
    create_keyframe_grid(keyframe_images, OUTPUT_KEYFRAMES)
    print()

    # Create all interpolated latents
    print("Creating interpolated latent sequence for seamless loop...")
    all_latents = []

    for i in range(NUM_KEYFRAMES):
        # Get current and next keyframe (wrap around for seamless loop)
        current_latent = keyframe_latents[i]
        next_latent = keyframe_latents[(i + 1) % NUM_KEYFRAMES]

        # Interpolate between them
        interpolated = interpolate_latents(current_latent, next_latent, frames_per_transition)
        all_latents.extend(interpolated)

    print(f"Total latent vectors: {len(all_latents)}")

    # Generate all frames
    print(f"\nGenerating {len(all_latents)} frames...")
    frames = []

    for i, latent in enumerate(tqdm(all_latents, desc="Generating frames")):
        # Generate frame
        img = generate_frame(pipeline, control_image, latent, PROMPT, NEGATIVE_PROMPT)
        frames.append(img)

        # Clear GPU cache periodically
        if DEVICE == 'cuda' and i % 20 == 0:
            torch.cuda.empty_cache()

    # Create GIF
    create_morphing_gif(frames, OUTPUT_GIF, FPS)

    print()
    print("=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print()
    print("Output files:")
    print(f"  - {OUTPUT_GIF} (main animation)")
    print(f"  - {OUTPUT_KEYFRAMES} (keyframe grid)")
    print(f"  - {OUTPUT_CONTROL} (conditioning image)")
    print()
    print("The animation shows smooth color morphing within a fixed circle structure,")
    print("demonstrating how ControlNet maintains structural consistency while")
    print("allowing creative variation through latent space interpolation.")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    generate_morph_animation()
