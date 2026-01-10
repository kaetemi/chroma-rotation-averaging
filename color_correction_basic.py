import cv2
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

# ============== LAB Scaling Functions ==============

def scale_lab_to_uint8(lab):
    """
    Scale LAB values to uint8 range.
    L: 0-100 -> 0-255
    A, B: -127 to 127 -> 0-255
    """
    L, A, B = cv2.split(lab)
    L_scaled = L * 255.0 / 100.0
    A_scaled = (A + 127.0) * 255.0 / 254.0
    B_scaled = (B + 127.0) * 255.0 / 254.0
    return L_scaled, A_scaled, B_scaled

def scale_uint8_to_lab(L, A, B):
    """
    Reverse the uint8 scaling back to LAB range.
    """
    L_lab = L.astype(np.float32) * 100.0 / 255.0
    A_lab = A.astype(np.float32) * 254.0 / 255.0 - 127.0
    B_lab = B.astype(np.float32) * 254.0 / 255.0 - 127.0
    return L_lab, A_lab, B_lab

def dither_channel_stack(channels):
    """Dither a list of scaled float channels and stack them."""
    return np.stack([floyd_steinberg_dither(ch) for ch in channels], axis=2)

def dither_rgb(rgb_float):
    """Dither RGB float image (0-1) to uint8."""
    channels = [np.clip(rgb_float[..., i] * 255.0, 0, 255) for i in range(3)]
    return dither_channel_stack(channels)

# ============== Main Processing ==============

def main(input_path, ref_path, output_path, keep_luminosity=False, verbose=False):
    # Set up logging
    log = print if verbose else lambda *args, **kwargs: None
    
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
    
    # Store original L channel if preserving luminosity
    original_L = input_lab[..., 0].copy()
    
    # Scale to uint8 range with dithering
    log("Scaling and dithering for histogram matching...")
    input_L, input_A, input_B = scale_lab_to_uint8(input_lab)
    ref_L, ref_A, ref_B = scale_lab_to_uint8(ref_lab)
    
    input_uint8 = dither_channel_stack([input_L, input_A, input_B])
    ref_uint8 = dither_channel_stack([ref_L, ref_A, ref_B])
    
    # Match histograms
    log("Matching histograms...")
    if keep_luminosity:
        matched_ab = exposure.match_histograms(input_uint8[..., 1:], ref_uint8[..., 1:], channel_axis=2)
        final_L = original_L
        _, final_A, final_B = scale_uint8_to_lab(
            np.zeros_like(matched_ab[..., 0]),  # Dummy L
            matched_ab[..., 0],
            matched_ab[..., 1]
        )
    else:
        matched_lab = exposure.match_histograms(input_uint8, ref_uint8, channel_axis=2)
        final_L, final_A, final_B = scale_uint8_to_lab(
            matched_lab[..., 0],
            matched_lab[..., 1],
            matched_lab[..., 2]
        )
    
    final_lab = cv2.merge([final_L, final_A.astype(np.float32), final_B.astype(np.float32)])
    
    # Convert back to linear RGB, then sRGB
    log("Converting back to sRGB...")
    final_linear = cv2.cvtColor(final_lab, cv2.COLOR_Lab2LRGB)
    final_srgb = linear_to_srgb(final_linear)
    
    # Final dither to uint8
    log("Final dithering...")
    final_uint8 = dither_rgb(final_srgb)
    
    # Save result
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
