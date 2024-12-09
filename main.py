import argparse
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def high_pass_filter(image_path, output_path, cutoff_distance=50):
    """
    Apply a high-pass filter to an image using Gaussian blur.

    Parameters:
        image_path (str): Path to the input image.
        output_path (str): Path to save the filtered image.
        cutoff_distance (float): Distance for Gaussian blur, controlling the cutoff frequency.
    """
    # Load the image using PIL to retain the color profile
    with Image.open(image_path) as pil_image:
        # Preserve the ICC color profile and data type
        icc_profile = pil_image.info.get("icc_profile")
        image = np.array(pil_image)
        original_dtype = image.dtype  # Preserve original bit depth
        mode = pil_image.mode

    # Convert image to a higher precision format for calculations
    image = image.astype('float32')

    # Gaussian blur for low-pass filtering
    blurred = gaussian_filter(image, sigma=cutoff_distance, axes=(0, 1))

    # Compute brightness to be added back
    mean = image.mean(axis=(0, 1))

    # Subtract blurred image from original and add back brightness
    high_pass = image - blurred + mean

    # Clip values to valid range and convert back to original data type
    high_pass = np.clip(high_pass, 0, np.iinfo(original_dtype).max).astype(original_dtype)

    # Save the high-pass filtered image
    high_pass_image = Image.fromarray(high_pass, mode=mode)
    if icc_profile:
        high_pass_image.save(output_path, icc_profile=icc_profile)
    else:
        high_pass_image.save(output_path)
    print(f"High-pass filtered image ({high_pass.shape}) saved to: {output_path}")


def cli():
    parser = argparse.ArgumentParser(description="Apply a high-pass filter to an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("output_path", type=str, help="Path to save the filtered image.")
    parser.add_argument(
        "--cutoff-distance",
        type=float,
        default=50,
        help="Distance for Gaussian blur, controlling the cutoff frequency (default: 50).",
    )
    args = parser.parse_args()

    try:
        high_pass_filter(args.image_path, args.output_path, args.cutoff_distance)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)