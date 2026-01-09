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

# ============== Tiling ==============

def generate_tile_blocks(H, W, H_ref, W_ref):
    """
    Generate overlapping blocks for tiled processing.
    
    Creates a 9x9 grid of tiles, then generates 8x8 blocks where each
    block spans 2x2 tiles, resulting in 50% overlap between adjacent blocks.
    Edge blocks are extended to cover the full image dimensions.
    
    Returns:
        List of tuples: (input_y_slice, input_x_slice, ref_y_slice, ref_x_slice)
    """
    tile_h, tile_w = H // 9, W // 9
    tile_h_ref, tile_w_ref = H_ref // 9, W_ref // 9
    
    blocks = []
    for i in range(8):
        for j in range(8):
            y_start = i * tile_h
            x_start = j * tile_w
            y_start_ref = i * tile_h_ref
            x_start_ref = j * tile_w_ref
            
            # Extend edge blocks to cover full image
            y_end = H if i == 7 else y_start + 2 * tile_h
            x_end = W if j == 7 else x_start + 2 * tile_w
            y_end_ref = H_ref if i == 7 else y_start_ref + 2 * tile_h_ref
            x_end_ref = W_ref if j == 7 else x_start_ref + 2 * tile_w_ref
            
            blocks.append((
                slice(y_start, y_end), slice(x_start, x_end),
                slice(y_start_ref, y_end_ref), slice(x_start_ref, x_end_ref)
            ))
    
    return blocks

def create_hamming_weights(height, width):
    """Create 2D Hamming window for smooth block blending."""
    y_win = np.hamming(height)
    x_win = np.hamming(width)
    return np.outer(y_win, x_win)

# ============== Block Processing ==============

def process_block_iteration(current_A, current_B, ref_A, ref_B, 
                            rotation_angles, ab_ranges_dict):
    """
    Process one iteration of AB-channel histogram matching for a single block.
    
    Rotates AB channels at multiple angles, performs histogram matching at each,
    then averages the results to reduce axis-aligned artifacts.
    
    Returns:
        Tuple of (corrected_A, corrected_B) as float32 arrays
    """
    all_corrected_A = []
    all_corrected_B = []
    
    for theta in rotation_angles:
        theta_rad = np.radians(theta)
        cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
        ab_ranges = ab_ranges_dict[theta]
        
        # Rotate AB
        a_rot = current_A * cos_t - current_B * sin_t
        b_rot = current_A * sin_t + current_B * cos_t
        ref_a_rot = ref_A * cos_t - ref_B * sin_t
        ref_b_rot = ref_A * sin_t + ref_B * cos_t
        
        # Scale and dither
        a_scaled, b_scaled = scale_ab_to_uint8(a_rot, b_rot, ab_ranges)
        ref_a_scaled, ref_b_scaled = scale_ab_to_uint8(ref_a_rot, ref_b_rot, ab_ranges)
        
        input_uint8 = dither_channel_stack([a_scaled, b_scaled])
        ref_uint8 = dither_channel_stack([ref_a_scaled, ref_b_scaled])
        
        # Match histograms
        matched = exposure.match_histograms(input_uint8, ref_uint8, channel_axis=2)
        
        # Scale back
        a_matched, b_matched = scale_uint8_to_ab(matched[..., 0], matched[..., 1], ab_ranges)
        
        # Rotate back
        cos_b, sin_b = np.cos(-theta_rad), np.sin(-theta_rad)
        a_back = a_matched * cos_b - b_matched * sin_b
        b_back = a_matched * sin_b + b_matched * cos_b
        
        all_corrected_A.append(a_back)
        all_corrected_B.append(b_back)
    
    return np.mean(all_corrected_A, axis=0), np.mean(all_corrected_B, axis=0)

def process_block_iteration_with_l(current_L, current_A, current_B, 
                                   ref_L, ref_A, ref_B,
                                   rotation_angles, ab_ranges_dict):
    """
    Process one iteration including L channel for a single block.
    
    This performs per-block luminosity matching in addition to AB matching.
    Note: The result feeds into a subsequent global histogram match, so this
    provides localized pre-correction rather than final luminosity values.
    
    Returns:
        Tuple of (corrected_L, corrected_A, corrected_B) as float32 arrays
    """
    # Process AB channels
    avg_A, avg_B = process_block_iteration(
        current_A, current_B, ref_A, ref_B, 
        rotation_angles, ab_ranges_dict
    )
    
    # Process L channel (no rotation needed)
    L_scaled = scale_l_to_uint8(current_L)
    ref_L_scaled = scale_l_to_uint8(ref_L)
    
    L_uint8 = floyd_steinberg_dither(L_scaled)
    ref_L_uint8 = floyd_steinberg_dither(ref_L_scaled)
    
    matched_L = exposure.match_histograms(L_uint8, ref_L_uint8)
    avg_L = scale_uint8_to_l(matched_L)
    
    return avg_L, avg_A, avg_B

# ============== Main Processing ==============

def main(input_path, ref_path, output_path, tiled_luminosity=False, verbose=False):
    """
    Main processing pipeline for localized color correction.
    
    Processing stages:
    1. Per-block iterative histogram matching (AB always, L if tiled_luminosity)
    2. Hamming-weighted accumulation of block results
    3. Global histogram match on all LAB channels (always applied)
    
    The --tiled-luminosity flag controls whether L gets per-block correction
    before the global pass. Without it, the original L passes through to the
    global histogram match unchanged. The global LAB histogram match is always
    performed as the final step.
    """
    # Set up logging
    log = print if verbose else lambda *args, **kwargs: None
    
    # Configuration
    blend_factors = [0.25, 0.5, 1.0]
    rotation_angles = [0, 30, 60]
    
    # Precompute all AB ranges
    log("Precomputing AB ranges for rotations:")
    ab_ranges_dict = precompute_all_ab_ranges(rotation_angles, log)
    
    # Ensure 0° is available for final histogram match
    if 0 not in ab_ranges_dict:
        ab_ranges_dict[0] = compute_ab_ranges(0)
    
    # Load images
    log("Loading images...")
    input_image = Image.open(input_path).convert("RGB")
    ref_image = Image.open(ref_path).convert("RGB")
    
    input_np = np.array(input_image, dtype=np.float32) / 255.0
    ref_np = np.array(ref_image, dtype=np.float32) / 255.0
    
    # Convert to linear RGB
    log("Converting to linear RGB...")
    input_linear = srgb_to_linear(input_np).astype(np.float32)
    ref_linear = srgb_to_linear(ref_np).astype(np.float32)
    
    # Convert to LAB
    log("Converting to LAB...")
    input_lab = cv2.cvtColor(input_linear, cv2.COLOR_LRGB2Lab)
    ref_lab = cv2.cvtColor(ref_linear, cv2.COLOR_LRGB2Lab)
    
    input_L, input_A, input_B = cv2.split(input_lab)
    ref_L, ref_A, ref_B = cv2.split(ref_lab)
    
    # Store original L for use when tiled_luminosity is disabled
    original_L = input_L.copy()
    
    # Generate tile blocks
    H, W = input_A.shape
    H_ref, W_ref = ref_A.shape
    blocks = generate_tile_blocks(H, W, H_ref, W_ref)
    log(f"Generated {len(blocks)} overlapping blocks for {H}x{W} image")
    
    # Initialize accumulators for weighted block blending
    A_acc = np.zeros_like(input_A, dtype=np.float32)
    B_acc = np.zeros_like(input_B, dtype=np.float32)
    weight_acc = np.zeros_like(input_A, dtype=np.float32)
    
    if tiled_luminosity:
        L_acc = np.zeros_like(input_L, dtype=np.float32)
        log("Tiled luminosity enabled - L channel will be processed per-block before global match")
    
    # Process each block
    for block_idx, block in enumerate(blocks):
        y_slice, x_slice, y_slice_ref, x_slice_ref = block
        
        if verbose and block_idx % 16 == 0:
            log(f"  Processing block {block_idx + 1}/{len(blocks)}...")
        
        # Extract block data
        current_A = input_A[y_slice, x_slice].copy()
        current_B = input_B[y_slice, x_slice].copy()
        ref_A_block = ref_A[y_slice_ref, x_slice_ref].copy()
        ref_B_block = ref_B[y_slice_ref, x_slice_ref].copy()
        
        if tiled_luminosity:
            current_L = input_L[y_slice, x_slice].copy()
            ref_L_block = ref_L[y_slice_ref, x_slice_ref].copy()
        
        # Iterative correction with increasing blend factors
        for iteration, blend_factor in enumerate(blend_factors):
            if tiled_luminosity:
                avg_L, avg_A, avg_B = process_block_iteration_with_l(
                    current_L, current_A, current_B,
                    ref_L_block, ref_A_block, ref_B_block,
                    rotation_angles, ab_ranges_dict
                )
                current_L = current_L * (1 - blend_factor) + avg_L * blend_factor
            else:
                avg_A, avg_B = process_block_iteration(
                    current_A, current_B, ref_A_block, ref_B_block,
                    rotation_angles, ab_ranges_dict
                )
            
            current_A = current_A * (1 - blend_factor) + avg_A * blend_factor
            current_B = current_B * (1 - blend_factor) + avg_B * blend_factor
        
        # Create Hamming weights for smooth blending between overlapping blocks
        bh, bw = current_A.shape
        weights = create_hamming_weights(bh, bw)
        
        # Accumulate weighted results
        A_acc[y_slice, x_slice] += current_A * weights
        B_acc[y_slice, x_slice] += current_B * weights
        weight_acc[y_slice, x_slice] += weights
        
        if tiled_luminosity:
            L_acc[y_slice, x_slice] += current_L * weights
    
    # Normalize accumulated results by total weights
    log("Normalizing accumulated results...")
    final_A = A_acc / np.maximum(weight_acc, 1e-7)
    final_B = B_acc / np.maximum(weight_acc, 1e-7)
    
    # Set L channel input for global histogram match:
    # - With tiled_luminosity: use per-block corrected L as input
    # - Without: use original L as input (no per-block L correction)
    if tiled_luminosity:
        final_L = L_acc / np.maximum(weight_acc, 1e-7)
    else:
        final_L = original_L
    
    # Final global histogram match (always applied to all LAB channels)
    # This ensures the output histogram matches the reference globally,
    # using the per-block results as the input to be matched.
    log("Performing final global histogram match on all LAB channels...")
    final_ab_ranges = ab_ranges_dict[0]
    
    L_scaled = scale_l_to_uint8(final_L)
    a_scaled, b_scaled = scale_ab_to_uint8(final_A, final_B, final_ab_ranges)
    current_uint8 = dither_channel_stack([L_scaled, a_scaled, b_scaled])
    
    ref_L_scaled = scale_l_to_uint8(ref_L)
    ref_a_scaled, ref_b_scaled = scale_ab_to_uint8(ref_A, ref_B, final_ab_ranges)
    ref_uint8 = dither_channel_stack([ref_L_scaled, ref_a_scaled, ref_b_scaled])
    
    matched_lab = exposure.match_histograms(current_uint8, ref_uint8, channel_axis=2)
    final_L = scale_uint8_to_l(matched_lab[..., 0])
    final_A, final_B = scale_uint8_to_ab(matched_lab[..., 1], matched_lab[..., 2], final_ab_ranges)
    
    # Log L channel stats
    log(f"L channel stats:")
    log(f"  Original  - min: {original_L.min():.2f}, max: {original_L.max():.2f}, mean: {original_L.mean():.2f}")
    log(f"  Final     - min: {final_L.min():.2f}, max: {final_L.max():.2f}, mean: {final_L.mean():.2f}")
    log(f"  Reference - min: {ref_L.min():.2f}, max: {ref_L.max():.2f}, mean: {ref_L.mean():.2f}")
    
    final_lab = cv2.merge([final_L, final_A.astype(np.float32), final_B.astype(np.float32)])
    
    # Convert back to linear RGB, then sRGB
    log("Converting back to sRGB...")
    final_linear = cv2.cvtColor(final_lab, cv2.COLOR_Lab2LRGB)
    final_srgb = linear_to_srgb(final_linear)
    
    # Final dithering
    log("Final dithering...")
    final_uint8 = dither_rgb(final_srgb)
    
    # Save result
    result = Image.fromarray(final_uint8)
    result.save(output_path)
    log(f"Saved result to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Localized color correction using overlapping blocks with LAB histogram matching.")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--ref", required=True, help="Reference image path")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--tiled-luminosity", action="store_true",
                        help="Process luminosity (L channel) per-tile before global match. "
                             "Without this flag, original L passes directly to global match.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed progress logging")
    
    args = parser.parse_args()
    
    main(args.input, args.ref, args.output, args.tiled_luminosity, args.verbose)
