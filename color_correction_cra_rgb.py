import numpy as np
from PIL import Image
from skimage import exposure
import argparse

# ============== Dithering (conditional numba/WASM) ==============

try:
    import numba

    @numba.jit(nopython=True)
    def floyd_steinberg_dither(img):
        """
        Floyd-Steinberg dithering with linear buffer and overflow padding.
        Matches the Rust WASM implementation exactly for bit-perfect output.

        Args:
            img: 2D numpy array of float32, values in range [0, 255]

        Returns:
            2D numpy array of uint8, same shape as input
        """
        h, w = img.shape
        length = h * w

        buf = np.zeros(length + w + 2, dtype=np.float32)
        buf[:length] = img.ravel()

        for i in range(length):
            old = buf[i]
            new = np.round(old)
            buf[i] = new
            err = old - new

            buf[i + 1] += err * (7.0 / 16.0)
            buf[i + w - 1] += err * (3.0 / 16.0)
            buf[i + w] += err * (5.0 / 16.0)
            buf[i + w + 1] += err * (1.0 / 16.0)

        return np.clip(buf[:length], 0, 255).astype(np.uint8).reshape(h, w)

except ImportError:
    # Running in Pyodide — use WASM implementation
    from js import floyd_steinberg_dither_wasm

    def floyd_steinberg_dither(img):
        """
        Floyd-Steinberg dithering via Rust WASM.
        Called when running in Pyodide where Numba is unavailable.
        """
        h, w = img.shape
        # Convert to Python list to avoid proxy issues
        flat = img.astype(np.float32).ravel().tolist()

        # Call Rust WASM function
        result = floyd_steinberg_dither_wasm(flat, w, h)

        # Convert result back to numpy
        return np.asarray(result.to_py(), dtype=np.uint8).reshape(h, w)

# Perceptual luminance weights (Rec.709)
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# Compress less-important channels, match green precisely
PERCEPTUAL_SCALE = LUMA_WEIGHTS / LUMA_WEIGHTS.max()
# ≈ [0.297, 1.0, 0.101]

# ============== sRGB <-> Linear RGB ==============

def srgb_to_linear(srgb):
    """Convert sRGB to linear RGB."""
    return np.where(srgb <= 0.04045,
                    srgb / 12.92,
                    ((srgb + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(linear):
    """Convert linear RGB to sRGB."""
    return np.where(linear <= 0.04045 / 12.92,
                    linear * 12.92,
                    1.055 * (np.clip(linear, 0, None) ** (1.0 / 2.4)) - 0.055)

# ============== Perceptual Scaling ==============

def perceptual_scale_rgb(linear_rgb, scale_factors):
    """Scale RGB by given scale factors."""
    return linear_rgb * scale_factors

def perceptual_unscale_rgb(scaled_rgb, scale_factors):
    """Reverse perceptual scaling."""
    return scaled_rgb / scale_factors

# ============== RGB Rotation around (1,1,1) axis ==============

def rotation_matrix_around_111(theta):
    """Rodrigues' rotation formula for axis (1,1,1)/√3"""
    c = np.cos(theta)
    s = np.sin(theta)
    third = 1.0 / 3.0
    sqrt3_inv = 1.0 / np.sqrt(3)
    
    return np.array([
        [third + (2*third)*c,  third - third*c - s*sqrt3_inv,  third - third*c + s*sqrt3_inv],
        [third - third*c + s*sqrt3_inv,  third + (2*third)*c,  third - third*c - s*sqrt3_inv],
        [third - third*c - s*sqrt3_inv,  third - third*c + s*sqrt3_inv,  third + (2*third)*c]
    ])

def rotate_rgb(rgb, theta):
    """Rotate RGB values around the gray axis by theta radians."""
    R = rotation_matrix_around_111(theta)
    shape = rgb.shape
    flat = rgb.reshape(-1, 3)
    rotated = flat @ R.T
    return rotated.reshape(shape)

# ============== Channel Range Computation ==============

def compute_channel_ranges(theta_deg, perceptual_scale=None):
    """
    Compute the min/max range for each channel after rotation.
    
    Args:
        theta_deg: Rotation angle in degrees
        perceptual_scale: Optional perceptual scale factors [r, g, b]
    
    Returns:
        Array of shape (3, 2) with [min, max] for each channel
    """
    if perceptual_scale is None:
        perceptual_scale = np.array([1.0, 1.0, 1.0])
    
    # 8 corners of the unit cube, scaled by perceptual factors
    corners = np.array([
        [r * perceptual_scale[0], g * perceptual_scale[1], b * perceptual_scale[2]]
        for r in [0, 1] for g in [0, 1] for b in [0, 1]
    ])
    
    # Rotate all corners
    theta_rad = np.radians(theta_deg)
    R = rotation_matrix_around_111(theta_rad)
    rotated = corners @ R.T
    
    # Find min/max per channel
    ranges = np.array([
        [rotated[:, c].min(), rotated[:, c].max()]
        for c in range(3)
    ])
    
    return ranges

def precompute_all_ranges(rotation_angles, perceptual_scale=None, log=None):
    """
    Precompute channel ranges for all rotation angles.
    
    Returns:
        Dict mapping theta_deg -> channel_ranges array (3, 2)
    """
    if log is None:
        log = lambda *args, **kwargs: None
    
    ranges = {}
    for theta_deg in rotation_angles:
        ranges[theta_deg] = compute_channel_ranges(theta_deg, perceptual_scale)
        log(f"  {theta_deg}°: R=[{ranges[theta_deg][0,0]:.4f}, {ranges[theta_deg][0,1]:.4f}], "
            f"G=[{ranges[theta_deg][1,0]:.4f}, {ranges[theta_deg][1,1]:.4f}], "
            f"B=[{ranges[theta_deg][2,0]:.4f}, {ranges[theta_deg][2,1]:.4f}]")
    
    return ranges

# ============== RGB Scaling Functions ==============

def scale_rgb_to_uint8(rgb, channel_ranges):
    """Scale RGB to uint8 range based on precomputed channel ranges."""
    result = np.empty_like(rgb)
    for c in range(3):
        min_val, max_val = channel_ranges[c]
        result[..., c] = (rgb[..., c] - min_val) / (max_val - min_val) * 255.0
    return result

def scale_uint8_to_rgb(rgb_uint8, channel_ranges):
    """Reverse the uint8 scaling."""
    result = np.empty(rgb_uint8.shape, dtype=np.float32)
    for c in range(3):
        min_val, max_val = channel_ranges[c]
        result[..., c] = rgb_uint8[..., c].astype(np.float32) / 255.0 * (max_val - min_val) + min_val
    return result

def dither_channel_stack(channels):
    """Dither a list of scaled float channels and stack them."""
    return np.stack([floyd_steinberg_dither(ch) for ch in channels], axis=2)

def dither_rgb(rgb_float):
    """Dither RGB float image (0-1) to uint8."""
    channels = [np.clip(rgb_float[..., i] * 255.0, 0, 255) for i in range(3)]
    return dither_channel_stack(channels)

# ============== RGB Processing ==============

def process_rgb_iteration(current_rgb, ref_rgb, rotation_angles, channel_ranges_dict, log):
    """Process one iteration of RGB-space histogram matching."""
    all_corrected = []
    
    for theta_deg in rotation_angles:
        log(f"    Processing angle {theta_deg}°")
        
        theta_rad = np.radians(theta_deg)
        channel_ranges = channel_ranges_dict[theta_deg]
        
        # Rotate
        current_rot = rotate_rgb(current_rgb, theta_rad)
        ref_rot = rotate_rgb(ref_rgb, theta_rad)
        
        # Scale and dither
        log(f"      Dithering...")
        current_scaled = scale_rgb_to_uint8(current_rot, channel_ranges)
        ref_scaled = scale_rgb_to_uint8(ref_rot, channel_ranges)
        
        current_uint8 = dither_channel_stack([current_scaled[..., i] for i in range(3)])
        ref_uint8 = dither_channel_stack([ref_scaled[..., i] for i in range(3)])
        
        # Match histograms
        log(f"      Matching histograms...")
        matched_uint8 = exposure.match_histograms(current_uint8, ref_uint8, channel_axis=2)
        
        # Scale back and rotate back
        matched_rot = scale_uint8_to_rgb(matched_uint8, channel_ranges)
        matched = rotate_rgb(matched_rot, -theta_rad)
        
        all_corrected.append(matched)
    
    return np.mean(all_corrected, axis=0)

# ============== Main Processing ==============

def main(input_path, ref_path, output_path, verbose=False, use_perceptual=False):
    # Set up logging
    log = print if verbose else lambda *args, **kwargs: None
    
    # Determine scale factors
    if use_perceptual:
        perceptual_scale = PERCEPTUAL_SCALE
        log("Using perceptual scaling (compress red/blue, match green precisely)")
        log(f"  Scale factors: R={perceptual_scale[0]:.3f}, G={perceptual_scale[1]:.3f}, B={perceptual_scale[2]:.3f}")
    else:
        perceptual_scale = None
        log("No perceptual scaling")
    
    # Configuration
    blend_factors = [0.25, 0.5, 1.0]
    rotation_angles = [0, 40, 80]  # Evenly spaced across 120° period
    
    # Precompute all channel ranges
    log("Precomputing channel ranges for rotations:")
    channel_ranges_dict = precompute_all_ranges(rotation_angles, perceptual_scale, log)
    
    # Also need ranges for 0° (final histogram match uses unrotated data)
    if 0 not in channel_ranges_dict:
        channel_ranges_dict[0] = compute_channel_ranges(0, perceptual_scale)
    
    # Load images
    input_image = Image.open(input_path).convert("RGB")
    ref_image = Image.open(ref_path).convert("RGB")
    
    input_np = np.array(input_image, dtype=np.float32) / 255.0
    ref_np = np.array(ref_image, dtype=np.float32) / 255.0
    
    # Convert to linear RGB
    log("Converting to linear RGB...")
    input_linear = srgb_to_linear(input_np).astype(np.float32)
    ref_linear = srgb_to_linear(ref_np).astype(np.float32)
    
    # Apply perceptual scaling if enabled
    if perceptual_scale is not None:
        log("Applying perceptual scaling...")
        current = perceptual_scale_rgb(input_linear, perceptual_scale)
        ref = perceptual_scale_rgb(ref_linear, perceptual_scale)
    else:
        current = input_linear.copy()
        ref = ref_linear
    
    for iteration, blend_factor in enumerate(blend_factors):
        log(f"Iteration {iteration + 1}, blend factor {blend_factor * 100:.0f}%")
        
        # Process in RGB space
        log("  Processing RGB space...")
        rgb_result = process_rgb_iteration(current, ref, rotation_angles, channel_ranges_dict, log)
        
        # Blend with current
        current = current * (1 - blend_factor) + rgb_result * blend_factor

    # Final histogram match
    log("Performing final histogram match...")
    
    final_ranges = channel_ranges_dict[0]
    current_scaled = scale_rgb_to_uint8(current, final_ranges)
    ref_scaled = scale_rgb_to_uint8(ref, final_ranges)
    
    current_uint8 = dither_channel_stack([current_scaled[..., i] for i in range(3)])
    ref_uint8 = dither_channel_stack([ref_scaled[..., i] for i in range(3)])
    
    matched_uint8 = exposure.match_histograms(current_uint8, ref_uint8, channel_axis=2)
    
    final_scaled = scale_uint8_to_rgb(matched_uint8, final_ranges)
    
    if perceptual_scale is not None:
        log("Removing perceptual scaling...")
        final_linear = perceptual_unscale_rgb(final_scaled, perceptual_scale)
    else:
        final_linear = final_scaled
    
    # Convert back to sRGB
    log("Converting back to sRGB...")
    final_srgb = linear_to_srgb(final_linear)
    
    # Final dither to uint8
    final_uint8 = dither_rgb(final_srgb)

    result = Image.fromarray(final_uint8)
    result.save(output_path)
    log(f"Saved result to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Color correction with RGB histogram matching.")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--ref", required=True, help="Reference image path")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed progress logging")
    parser.add_argument("--perceptual", "-p", action="store_true",
                        help="Rotate in perceptually-weighted color space (preserves green, mixes red/blue)")
    
    args = parser.parse_args()
    
    main(args.input, args.ref, args.output, args.verbose, args.perceptual)
