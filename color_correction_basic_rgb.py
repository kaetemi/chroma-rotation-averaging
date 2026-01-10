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

def dither_channel_stack(channels):
    """Dither a list of scaled float channels and stack them."""
    return np.stack([floyd_steinberg_dither(ch) for ch in channels], axis=2)

def dither_rgb(rgb_float):
    """Dither RGB float image (0-1) to uint8."""
    channels = [np.clip(rgb_float[..., i] * 255.0, 0, 255) for i in range(3)]
    return dither_channel_stack(channels)

# ============== Main Processing ==============

def main(input_path, ref_path, output_path, verbose=False):
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
    
    # Scale to uint8 range with dithering
    log("Scaling and dithering for histogram matching...")
    input_255 = np.clip(input_linear * 255.0, 0, 255)
    ref_255 = np.clip(ref_linear * 255.0, 0, 255)
    
    input_uint8 = dither_channel_stack([input_255[..., i] for i in range(3)])
    ref_uint8 = dither_channel_stack([ref_255[..., i] for i in range(3)])
    
    # Match histograms
    log("Matching histograms...")
    matched_uint8 = exposure.match_histograms(input_uint8, ref_uint8, channel_axis=2)
    
    # Scale back to 0-1 range
    matched_linear = matched_uint8.astype(np.float32) / 255.0
    
    # Convert back to sRGB
    log("Converting back to sRGB...")
    final_srgb = linear_to_srgb(matched_linear)
    
    # Final dither to uint8
    log("Final dithering...")
    final_uint8 = dither_rgb(final_srgb)
    
    # Save result
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
    
    args = parser.parse_args()
    
    main(args.input, args.ref, args.output, args.verbose)
