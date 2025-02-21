import cv2
import numpy as np
from PIL import Image
from skimage import exposure
import argparse
import numba

# Scale to uint8 range without applying clipping
def scale_l_to_uint8(L):
    # L: 0 to 100 -> 0 to 255
    L_scaled = L * 255.0 / 100.0
    return L_scaled

def scale_uint8_to_l(L):
    # Reverse the scaling
    L_lab = L.astype(np.float32) * 100.0 / 255.0
    return L_lab

def scale_ab_to_uint8(a, b):
    # a/b: -127 to 127 -> 0 to 255
    a_scaled = (a + 127.0) * 255.0 / 254.0
    b_scaled = (b + 127.0) * 255.0 / 254.0
    return a_scaled, b_scaled

def scale_uint8_to_ab(a, b):
    # Reverse the scaling
    a_lab = (a.astype(np.float32) * 254.0 / 255.0) - 127.0
    b_lab = (b.astype(np.float32) * 254.0 / 255.0) - 127.0
    return a_lab, b_lab

scale_safer_45 = np.sin(np.radians(45)) + np.cos(np.radians(45))
scale_safer_45_inv = 1.0 / scale_safer_45
scale_safer_30 = np.sin(np.radians(30)) + np.cos(np.radians(30))
scale_safer_30_inv = 1.0 / scale_safer_30
scale_safer_15 = np.sin(np.radians(15)) + np.cos(np.radians(15))
scale_safer_15_inv = 1.0 / scale_safer_15

def scale_ab_to_uint8_safer(a, b, scale_inv):
    # a/b: -127 to 127 -> 0 to 255
    a_scaled = (a + 127.0) * 255.0 * scale_inv / 254.0
    b_scaled = (b + 127.0) * 255.0 * scale_inv / 254.0
    return a_scaled, b_scaled

def scale_uint8_to_ab_safer(a, b, scale):
    # Reverse the scaling
    a_lab = (a.astype(np.float32) * 254.0 * scale / 255.0) - 127.0
    b_lab = (b.astype(np.float32) * 254.0 * scale / 255.0) - 127.0
    return a_lab, b_lab
    
def scale_ab_to_uint8_checked(a, b, theta):
    if theta == 0:
        return scale_ab_to_uint8(a, b)
    elif theta == 15 or theta == 75:
        return scale_ab_to_uint8_safer(a, b, scale_safer_15_inv)
    elif theta == 45:
        return scale_ab_to_uint8_safer(a, b, scale_safer_45_inv)
    else:
        return scale_ab_to_uint8_safer(a, b, scale_safer_30_inv)

def scale_uint8_to_ab_checked(a, b, theta):
    if theta == 0:
        return scale_uint8_to_ab(a, b)
    elif theta == 15 or theta == 75:
        return scale_uint8_to_ab_safer(a, b, scale_safer_15)
    elif theta == 45:
        return scale_uint8_to_ab_safer(a, b, scale_safer_45)
    else:
        return scale_uint8_to_ab_safer(a, b, scale_safer_30)

@numba.jit(nopython=True)
def floyd_steinberg_dither(img):
    h, w = img.shape
    for y in range(h-1):
        for x in range(w-1):
            old_pixel = img[y, x]
            new_pixel = np.round(old_pixel)
            img[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            
            img[y, x+1] += quant_error * 7/16
            img[y+1, x-1] += quant_error * 3/16
            img[y+1, x] += quant_error * 5/16
            img[y+1, x+1] += quant_error * 1/16
            
    return np.clip(img, 0, 255).astype(np.uint8)

def scale_ab_to_uint8_dither(a, b):
    a_scaled, b_scaled = scale_ab_to_uint8(a, b)
    return floyd_steinberg_dither(a_scaled), floyd_steinberg_dither(b_scaled)

def scale_ab_to_uint8_dither_checked(a, b, theta):
    a_scaled, b_scaled = scale_ab_to_uint8_checked(a, b, theta)
    return floyd_steinberg_dither(a_scaled), floyd_steinberg_dither(b_scaled)

# Scale to uint8 range and dither
def scale_lab_to_uint8_dither(L, a, b):
    L_scaled = scale_l_to_uint8(L)
    a_scaled, b_scaled = scale_ab_to_uint8(a, b)
    return floyd_steinberg_dither(L_scaled), floyd_steinberg_dither(a_scaled), floyd_steinberg_dither(b_scaled)

def main(input_path, ref_path, output_path):
    # Load images and convert to float32 (0-1 range)
    input_image = Image.open(input_path).convert("RGB")
    ref_image = Image.open(ref_path).convert("RGB")
    
    input_np = np.array(input_image, dtype=np.float32) / 255.0
    ref_np = np.array(ref_image, dtype=np.float32) / 255.0
    
    # Convert to LAB using OpenCV
    input_lab = cv2.cvtColor(input_np, cv2.COLOR_RGB2LAB)
    input_L, input_A, input_B = cv2.split(input_lab)
    
    ref_lab = cv2.cvtColor(ref_np, cv2.COLOR_RGB2LAB)
    ref_L, ref_A, ref_B = cv2.split(ref_lab)
    
    H, W = input_A.shape
    tile_h, tile_w = H // 9, W // 9  # 9x9 tiles
    H_ref, W_ref = ref_A.shape
    tile_h_ref, tile_w_ref = H_ref // 9, W_ref // 9
    
    # Generate overlapping blocks (2x2 tiles each)
    blocks = []
    for i in range(8):
        for j in range(8):
            y_start = i * tile_h
            y_end = y_start + 2 * tile_h
            x_start = j * tile_w
            x_end = x_start + 2 * tile_w
            y_start_ref = i * tile_h_ref
            y_end_ref = y_start_ref + 2 * tile_h_ref
            x_start_ref = j * tile_w_ref
            x_end_ref = x_start_ref + 2 * tile_w_ref
            blocks.append( (slice(y_start, y_end), slice(x_start, x_end), slice(y_start_ref, y_end_ref), slice(x_start_ref, x_end_ref)) )
    
    # Initialize accumulators
    A_acc = np.zeros_like(input_A, dtype=np.float32)
    B_acc = np.zeros_like(input_B, dtype=np.float32)
    weight_acc = np.zeros_like(input_A, dtype=np.float32)
    
    for block in blocks:
        y_slice, x_slice, y_slice_ref, x_slice_ref = block
        
        # Extract current block data
        current_A_block = input_A[y_slice, x_slice].copy()
        current_B_block = input_B[y_slice, x_slice].copy()
        ref_A_block = ref_A[y_slice_ref, x_slice_ref].copy()
        ref_B_block = ref_B[y_slice_ref, x_slice_ref].copy()
        
        blend_factors = [0.25, 0.5, 1.0]
        current_A = current_A_block.copy()
        current_B = current_B_block.copy()
        
        for blend_factor in blend_factors:
            all_corrected_A = []
            all_corrected_B = []
            
            for theta in [0, 30, 60]:
                # Rotate AB channels
                theta_rad = np.radians(theta)
                cos_theta = np.cos(theta_rad)
                sin_theta = np.sin(theta_rad)
                a_rot = current_A * cos_theta - current_B * sin_theta
                b_rot = current_A * sin_theta + current_B * cos_theta
                
                # Scale and dither
                a_rot_scaled, b_rot_scaled = scale_ab_to_uint8_dither_checked(a_rot, b_rot, theta)
                
                # Process reference block
                ref_a_rot = ref_A_block * cos_theta - ref_B_block * sin_theta
                ref_b_rot = ref_A_block * sin_theta + ref_B_block * cos_theta
                ref_a_rot_scaled, ref_b_rot_scaled = scale_ab_to_uint8_dither_checked(ref_a_rot, ref_b_rot, theta)
                
                # Match histograms
                input_rot_ab = np.stack([a_rot_scaled, b_rot_scaled], axis=2)
                ref_rot_ab = np.stack([ref_a_rot_scaled, ref_b_rot_scaled], axis=2)
                matched_rot_ab = exposure.match_histograms(input_rot_ab, ref_rot_ab, channel_axis=2)
                
                # Rotate back
                a_rot_matched, b_rot_matched = scale_uint8_to_ab_checked(matched_rot_ab[...,0], matched_rot_ab[...,1], theta)
                theta_back_rad = np.radians(-theta)
                cos_back = np.cos(theta_back_rad)
                sin_back = np.sin(theta_back_rad)
                a_back = a_rot_matched * cos_back - b_rot_matched * sin_back
                b_back = a_rot_matched * sin_back + b_rot_matched * cos_back
                
                all_corrected_A.append(a_back)
                all_corrected_B.append(b_back)
            
            # Average corrections and blend
            avg_A = np.mean(all_corrected_A, axis=0)
            avg_B = np.mean(all_corrected_B, axis=0)
            current_A = current_A * (1 - blend_factor) + avg_A * blend_factor
            current_B = current_B * (1 - blend_factor) + avg_B * blend_factor
        
        # Create weighting matrix (Hamming window)
        bh, bw = current_A.shape
        y_win = np.hamming(bh)
        x_win = np.hamming(bw)
        weights = np.outer(y_win, x_win)
        
        # Accumulate results
        A_acc[y_slice, x_slice] += current_A * weights
        B_acc[y_slice, x_slice] += current_B * weights
        weight_acc[y_slice, x_slice] += weights
    
    # Normalize accumulated results
    final_A = A_acc / np.maximum(weight_acc, 1e-7)
    final_B = B_acc / np.maximum(weight_acc, 1e-7)
    
    # Final LAB processing
    L_scaled = scale_l_to_uint8(input_L)
    L_dithered = floyd_steinberg_dither(L_scaled)
    a_scaled, b_scaled = scale_ab_to_uint8(final_A, final_B)
    a_dithered = floyd_steinberg_dither(a_scaled)
    b_dithered = floyd_steinberg_dither(b_scaled)
    
    current_lab_scaled = cv2.merge([L_dithered, a_dithered, b_dithered])
    ref_L_scaled = scale_l_to_uint8(ref_L)
    ref_a_scaled, ref_b_scaled = scale_ab_to_uint8(ref_A, ref_B)
    ref_L_dithered = floyd_steinberg_dither(ref_L_scaled)
    ref_a_dithered = floyd_steinberg_dither(ref_a_scaled)
    ref_b_dithered = floyd_steinberg_dither(ref_b_scaled)
    ref_lab_scaled = cv2.merge([ref_L_dithered, ref_a_dithered, ref_b_dithered])
    
    # Final histogram matching
    matched_lab = exposure.match_histograms(current_lab_scaled, ref_lab_scaled, channel_axis=2)
    
    # Convert back to RGB
    final_rgb = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)
    result = Image.fromarray(final_rgb)
    result.save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Localized color correction using overlapping blocks.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.input, args.ref, args.output)
