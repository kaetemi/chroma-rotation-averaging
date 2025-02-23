import cv2
import numpy as np
from PIL import Image
from skimage import exposure
import argparse

# pip install opencv-python Pillow scikit-image

def apply_color_correction(reference_image, original_image):
    print("Applying color correction...")
    # Convert reference image to LAB color space
    ref_lab = cv2.cvtColor(np.asarray(reference_image), cv2.COLOR_RGB2LAB)
    
    # Convert and match histograms
    lab_image = cv2.cvtColor(np.asarray(original_image), cv2.COLOR_RGB2LAB)
    corrected = cv2.cvtColor(
        exposure.match_histograms(lab_image, ref_lab, channel_axis=2),
        cv2.COLOR_LAB2RGB
    ).astype("uint8")
    
    return Image.fromarray(corrected).convert("RGB")

def main(input_path, ref_path, output_path):
    # Load images
    ref_image = Image.open(ref_path).convert("RGB")
    input_image = Image.open(input_path).convert("RGB")
    
    # Process images
    result = apply_color_correction(ref_image, input_image)
    
    # Save output
    result.save(output_path)
    print(f"Saved result to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply color correction between images")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--ref", required=True, help="Path to reference image for color correction")
    parser.add_argument("--output", required=True, help="Path to save output image")
    
    args = parser.parse_args()
    main(args.input, args.ref, args.output)
