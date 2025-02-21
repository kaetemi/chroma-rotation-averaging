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
    
    # Convert to LAB using OpenCV (will use proper LAB ranges)
    input_lab = cv2.cvtColor(input_np, cv2.COLOR_RGB2LAB)
    input_L, input_A, input_B = cv2.split(input_lab)
    
    ref_lab = cv2.cvtColor(ref_np, cv2.COLOR_RGB2LAB)
    ref_L, ref_A, ref_B = cv2.split(ref_lab)
    
    # Initialize current A and B
    current_A = input_A.copy()
    current_B = input_B.copy()
    
    # Blend factors for each iteration: 25%, 50%, 100%
    blend_factors = [0.25, 0.5, 1.0]
    # blend_factors = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    
    for i, blend_factor in enumerate(blend_factors):
        print(f"Iteration {i+1}, blend factor {blend_factor*100}%")
        all_corrected_A = []
        all_corrected_B = []
        
        # for theta in [0, 15, 30, 45, 60, 75]:
        for theta in [0, 30, 60]:
            print(f"  Processing angle {theta}°")
            # Center around origin
            a = current_A  # already centered around 0
            b = current_B  # already centered around 0
            
            # Rotate AB by theta around center
            theta_rad = np.radians(theta)
            cos_theta = np.cos(theta_rad)
            sin_theta = np.sin(theta_rad)
            a_rot = a * cos_theta - b * sin_theta
            b_rot = a * sin_theta + b * cos_theta
            
            # Scale and dither convert to uint8
            print(f"  Dithering")
            a_rot_scaled, b_rot_scaled = scale_ab_to_uint8_dither_checked(a_rot, b_rot, theta)
            print(f"  Dithering done")
            
            # Center reference AB around origin
            ref_a = ref_A  # already centered around 0
            ref_b = ref_B  # already centered around 0
            
            # Rotate reference AB by theta around center
            ref_a_rot = ref_a * cos_theta - ref_b * sin_theta
            ref_b_rot = ref_a * sin_theta + ref_b * cos_theta

            # Scale and dither convert to uint8
            ref_a_rot_scaled, ref_b_rot_scaled = scale_ab_to_uint8_dither_checked(ref_a_rot, ref_b_rot, theta)
            
            # Stack for histogram matching
            input_rot_ab = np.stack([a_rot_scaled, b_rot_scaled], axis=2)
            ref_rot_ab = np.stack([ref_a_rot_scaled, ref_b_rot_scaled], axis=2)
            
            # Match histograms
            print(f"  Matching to histogram...")
            matched_rot_ab = exposure.match_histograms(input_rot_ab, ref_rot_ab, channel_axis=2)
            print(f"  Matching done")
            
            # Extract matched rotated A and B
            a_rot_matched, b_rot_matched = scale_uint8_to_ab_checked(matched_rot_ab[..., 0], matched_rot_ab[..., 1], theta)
            
            # Rotate back by -theta
            theta_back_rad = np.radians(-theta)
            cos_back = np.cos(theta_back_rad)
            sin_back = np.sin(theta_back_rad)
            a_back = a_rot_matched * cos_back - b_rot_matched * sin_back
            b_back = a_rot_matched * sin_back + b_rot_matched * cos_back
            
            all_corrected_A.append(a_back)
            all_corrected_B.append(b_back)
        
        # Average the corrected A and B channels from all angles
        avg_A = np.mean(all_corrected_A, axis=0)
        avg_B = np.mean(all_corrected_B, axis=0)
        
        # Blend with current A and B
        current_A = current_A * (1 - blend_factor) + avg_A * blend_factor
        current_B = current_B * (1 - blend_factor) + avg_B * blend_factor

    # After iterations, perform final LAB histogram match
    print("Performing final LAB histogram match")
    
    # Merge channels
    input_L_scaled, input_a_scaled, input_b_scaled = scale_lab_to_uint8_dither(input_L, current_A, current_B)
    current_lab_scaled = cv2.merge([input_L_scaled, input_a_scaled, input_b_scaled])
    ref_L_scaled, ref_a_scaled, ref_b_scaled = scale_lab_to_uint8_dither(ref_L, ref_a, ref_b)
    ref_lab_scaled = cv2.merge([ref_L_scaled, ref_a_scaled, ref_b_scaled])
    matched_lab = exposure.match_histograms(current_lab_scaled, ref_lab_scaled, channel_axis=2)
    
    # Convert matched LAB back to RGB (still in float32 0-1 range)
    # rgb_corrected = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)
    
    # Scale to 0-255 range for dithering
    # rgb_corrected = rgb_corrected * 255.0
    
    # Apply dithering to each channel
    # dithered_R = floyd_steinberg_dither(rgb_corrected[:,:,0])
    # dithered_G = floyd_steinberg_dither(rgb_corrected[:,:,1])
    # dithered_B = floyd_steinberg_dither(rgb_corrected[:,:,2])
    
    # Merge dithered channels
    # final_rgb = cv2.merge([dithered_R, dithered_G, dithered_B])

    # Convert matched LAB back to RGB (already in uint8 range)
    final_rgb = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)
    
    # Convert to PIL Image and save
    result = Image.fromarray(final_rgb)
    result.save(output_path)
    print(f"Saved result to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply advanced color correction between images using rotated AB planes.")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--ref", required=True, help="Path to reference image for color correction")
    parser.add_argument("--output", required=True, help="Path to save output image")
    
    args = parser.parse_args()
    
    main(args.input, args.ref, args.output)
