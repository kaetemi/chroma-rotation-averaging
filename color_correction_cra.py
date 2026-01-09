import cv2
import numpy as np
from PIL import Image
from skimage import exposure
import argparse
import numba

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

# ============== AB Range Computation ==============

def compute_ab_ranges(theta_deg):
    """
    Compute the min/max range for A and B channels after rotation.
    
    The AB plane forms a square from -127 to +127 on each axis.
    After rotation, we need the bounding box of the rotated square.
    
    Args:
        theta_deg: Rotation angle in degrees
    
    Returns:
        Array of shape (2, 2) with [min, max] for A and B channels
    """
    # Corners of the AB square
    corners = np.array([
        [-127, -127],
        [-127, +127],
        [+127, -127],
        [+127, +127],
    ], dtype=np.float32)
    
    # Rotation matrix
    theta_rad = np.radians(theta_deg)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    R = np.array([
        [cos_t, -sin_t],
        [sin_t,  cos_t]
    ])
    
    # Rotate all corners
    rotated = corners @ R.T
    
    # Find min/max per channel
    ranges = np.array([
        [rotated[:, 0].min(), rotated[:, 0].max()],  # A range
        [rotated[:, 1].min(), rotated[:, 1].max()],  # B range
    ])
    
    return ranges

def precompute_all_ab_ranges(rotation_angles, log=None):
    """
    Precompute AB channel ranges for all rotation angles.
    
    Returns:
        Dict mapping theta_deg -> ab_ranges array (2, 2)
    """
    if log is None:
        log = lambda *args, **kwargs: None
    
    ranges = {}
    for theta_deg in rotation_angles:
        ranges[theta_deg] = compute_ab_ranges(theta_deg)
        log(f"  {theta_deg}°: A=[{ranges[theta_deg][0,0]:.2f}, {ranges[theta_deg][0,1]:.2f}], "
            f"B=[{ranges[theta_deg][1,0]:.2f}, {ranges[theta_deg][1,1]:.2f}]")
    
    return ranges

# ============== LAB Scaling Functions ==============

def scale_l_to_uint8(L):
    """L: 0 to 100 -> 0 to 255"""
    return L * 255.0 / 100.0

def scale_uint8_to_l(L):
    """Reverse the L scaling."""
    return L.astype(np.float32) * 100.0 / 255.0

def scale_ab_to_uint8(a, b, ab_ranges):
    """Scale AB values to uint8 range based on precomputed ranges."""
    a_min, a_max = ab_ranges[0]
    b_min, b_max = ab_ranges[1]
    a_scaled = (a - a_min) / (a_max - a_min) * 255.0
    b_scaled = (b - b_min) / (b_max - b_min) * 255.0
    return a_scaled, b_scaled

def scale_uint8_to_ab(a, b, ab_ranges):
    """Reverse the uint8 scaling."""
    a_min, a_max = ab_ranges[0]
    b_min, b_max = ab_ranges[1]
    a_lab = a.astype(np.float32) / 255.0 * (a_max - a_min) + a_min
    b_lab = b.astype(np.float32) / 255.0 * (b_max - b_min) + b_min
    return a_lab, b_lab

# ============== Dithering ==============

@numba.jit(nopython=True)
def floyd_steinberg_dither(img):
    """Apply Floyd-Steinberg dithering to a single channel."""
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

def dither_channel_stack(channels):
    """Dither a list of scaled float channels and stack them."""
    return np.stack([floyd_steinberg_dither(ch) for ch in channels], axis=2)

def dither_rgb(rgb_float):
    """Dither RGB float image (0-1) to uint8."""
    channels = [np.clip(rgb_float[..., i] * 255.0, 0, 255) for i in range(3)]
    return dither_channel_stack(channels)

# ============== LAB Processing ==============

def process_lab_iteration(current_lab, ref_lab, rotation_angles, ab_ranges_dict, log):
    """Process one iteration of LAB-space histogram matching."""
    current_L, current_A, current_B = cv2.split(current_lab)
    ref_L, ref_A, ref_B = cv2.split(ref_lab)
    
    all_corrected_A = []
    all_corrected_B = []
    
    for theta in rotation_angles:
        log(f"    Processing angle {theta}°")
        
        theta_rad = np.radians(theta)
        cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
        ab_ranges = ab_ranges_dict[theta]
        
        # Rotate AB
        a_rot = current_A * cos_t - current_B * sin_t
        b_rot = current_A * sin_t + current_B * cos_t
        ref_a_rot = ref_A * cos_t - ref_B * sin_t
        ref_b_rot = ref_A * sin_t + ref_B * cos_t
        
        # Scale and dither
        log(f"      Dithering...")
        a_scaled, b_scaled = scale_ab_to_uint8(a_rot, b_rot, ab_ranges)
        ref_a_scaled, ref_b_scaled = scale_ab_to_uint8(ref_a_rot, ref_b_rot, ab_ranges)
        
        input_uint8 = dither_channel_stack([a_scaled, b_scaled])
        ref_uint8 = dither_channel_stack([ref_a_scaled, ref_b_scaled])
        
        # Match histograms
        log(f"      Matching histograms...")
        matched = exposure.match_histograms(input_uint8, ref_uint8, channel_axis=2)
        
        # Scale back
        a_matched, b_matched = scale_uint8_to_ab(matched[..., 0], matched[..., 1], ab_ranges)
        
        # Rotate back
        cos_b, sin_b = np.cos(-theta_rad), np.sin(-theta_rad)
        a_back = a_matched * cos_b - b_matched * sin_b
        b_back = a_matched * sin_b + b_matched * cos_b
        
        all_corrected_A.append(a_back)
        all_corrected_B.append(b_back)
    
    avg_A = np.mean(all_corrected_A, axis=0)
    avg_B = np.mean(all_corrected_B, axis=0)
    
    return cv2.merge([current_L, avg_A.astype(np.float32), avg_B.astype(np.float32)])

# ============== Main Processing ==============

def main(input_path, ref_path, output_path, keep_luminosity, verbose=False):
    # Set up logging
    log = print if verbose else lambda *args, **kwargs: None
    
    # Configuration
    blend_factors = [0.25, 0.5, 1.0]
    rotation_angles = [0, 30, 60]
    
    # Precompute all AB ranges
    log("Precomputing AB ranges for rotations:")
    ab_ranges_dict = precompute_all_ab_ranges(rotation_angles, log)
    
    # Also need ranges for 0° for final histogram match (if not already included)
    if 0 not in ab_ranges_dict:
        ab_ranges_dict[0] = compute_ab_ranges(0)
    
    # Load images
    input_image = Image.open(input_path).convert("RGB")
    ref_image = Image.open(ref_path).convert("RGB")
    
    input_np = np.array(input_image, dtype=np.float32) / 255.0
    ref_np = np.array(ref_image, dtype=np.float32) / 255.0
    
    # Convert to linear RGB
    log("Converting to linear RGB...")
    input_linear = srgb_to_linear(input_np).astype(np.float32)
    ref_linear = srgb_to_linear(ref_np).astype(np.float32)
    
    # Convert to LAB (using linear RGB)
    input_lab = cv2.cvtColor(input_linear, cv2.COLOR_LRGB2Lab)
    ref_lab = cv2.cvtColor(ref_linear, cv2.COLOR_LRGB2Lab)
    
    # Store original L channel if preserving luminosity
    original_L = input_lab[..., 0].copy()
    
    # Current state
    current_lab = input_lab.copy()
    
    for iteration, blend_factor in enumerate(blend_factors):
        log(f"Iteration {iteration + 1}, blend factor {blend_factor * 100:.0f}%")
        
        # Process in LAB space
        log("  Processing LAB space...")
        lab_result = process_lab_iteration(current_lab, ref_lab, rotation_angles, ab_ranges_dict, log)
        
        # Blend with current
        current_lab = current_lab * (1 - blend_factor) + lab_result * blend_factor

    # Final LAB histogram match
    log("Performing final LAB histogram match...")
    current_L, current_A, current_B = cv2.split(current_lab)
    ref_L, ref_A, ref_B = cv2.split(ref_lab)
    
    final_ab_ranges = ab_ranges_dict[0]  # Unrotated ranges
    
    L_scaled = scale_l_to_uint8(current_L)
    a_scaled, b_scaled = scale_ab_to_uint8(current_A, current_B, final_ab_ranges)
    current_uint8 = dither_channel_stack([L_scaled, a_scaled, b_scaled])
    
    ref_L_scaled = scale_l_to_uint8(ref_L)
    ref_a_scaled, ref_b_scaled = scale_ab_to_uint8(ref_A, ref_B, final_ab_ranges)
    ref_uint8 = dither_channel_stack([ref_L_scaled, ref_a_scaled, ref_b_scaled])

    if keep_luminosity:
        matched_ab = exposure.match_histograms(current_uint8[..., 1:], ref_uint8[..., 1:], channel_axis=2)
        final_L = original_L
        final_a, final_b = scale_uint8_to_ab(matched_ab[..., 0], matched_ab[..., 1], final_ab_ranges)
    else:
        matched_lab = exposure.match_histograms(current_uint8, ref_uint8, channel_axis=2)
        final_L = scale_uint8_to_l(matched_lab[..., 0])
        final_a, final_b = scale_uint8_to_ab(matched_lab[..., 1], matched_lab[..., 2], final_ab_ranges)
    
    final_lab = cv2.merge([final_L, final_a.astype(np.float32), final_b.astype(np.float32)])
    
    # Convert to linear RGB, then sRGB
    log("Converting back to sRGB...")
    final_linear = cv2.cvtColor(final_lab, cv2.COLOR_Lab2LRGB)
    final_srgb = linear_to_srgb(final_linear)
    
    # Final dither to uint8
    final_uint8 = dither_rgb(final_srgb)

    result = Image.fromarray(final_uint8)
    result.save(output_path)
    log(f"Saved result to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Color correction with LAB histogram matching.")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--ref", required=True, help="Reference image path")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--keep-luminosity", action="store_true", 
                        help="Preserve original luminosity (L channel)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed progress logging")
    
    args = parser.parse_args()
    
    main(args.input, args.ref, args.output, args.keep_luminosity, args.verbose)
