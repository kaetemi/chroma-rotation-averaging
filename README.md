# Color Depth Restoration: Chroma Rotation Averaging (CRA) Technique

Color Depth Restoration is a collection of Python scripts designed to restore the original color vibrance and shading depth contrast in images that have undergone upscaling processes. These scripts are particularly useful in an upscaler pipeline where some steps may cause color shifts, resulting in duller or less vibrant images.

## Overview

The repository includes three main scripts:

1. `color_correction_basic.py`: Performs basic histogram matching to restore colors.
2. `color_correction_cra.py`: Applies the Chroma Rotation Averaging (CRA) technique to mitigate color flips and improve color accuracy.
3. `color_correction_tiled.py`: Builds upon the CRA technique by dividing the image into smaller tiles and blending them together to fix color issues with whole image gradients.

## Chroma Rotation Averaging (CRA) Technique

The Chroma Rotation Averaging (CRA) technique is an extension of the basic color correction method that aims to mitigate color flips and improve overall color accuracy. The rationale behind this technique is as follows:

1. Basic histogram matching considers the histograms of each color channel separately, which can sometimes lead to colors aligned with the Lab color space's a-b axis flipping 180° in hue.

2. To address this issue, the CRA technique performs histogram matching on additional rotations of the chroma plane (e.g., 30° and 60° rotations) in addition to the original orientation.

3. The results from each rotation are then rotated back to the original orientation and averaged together.

4. This process is applied iteratively, with the blending factor increasing from 25% to 50% and finally 100% across iterations.

By considering the rotated chroma planes during histogram matching, the CRA technique takes into account the actual color information more than just the individual channels. This helps to guide the colors in the right direction and avoid most color flips.

The CRA technique is particularly effective in restoring the vividness of skin colors and the depth of the brightest and darkest blues in underwater images.

## Installation

To set up the required dependencies, follow these steps:

1. Create a virtual environment using Python 3.10 on Ubuntu:
   ```
   python3.10 -m venv venv
   ```

2. Install the necessary packages using pip:
   ```
   venv/bin/python -m pip install opencv-python Pillow scikit-image blendmodes numba
   ```

## Usage

To use the color depth restoration scripts, run the desired script with the following command-line arguments:

```
venv/bin/python color_correction_<script>.py --input <input_image> --ref <reference_image> --output <output_image>
```

- `<script>`: The name of the script you want to run (`basic`, `cra`, or `tiled`).
- `<input_image>`: The path to the input image that needs color restoration.
- `<reference_image>`: The path to the reference image with the desired color characteristics.
- `<output_image>`: The path where the color-restored output image will be saved.

## Examples

Here are some sample images demonstrating the effectiveness of the color depth restoration scripts:

| Original | Upscaled (No Color Matching) | Basic Color Matching | CRA Color Matching | Tiled Color Matching |
|----------|------------------------------|----------------------|--------------------|----------------------|
| ![Original](assets/original_closeup.jpg) | ![Upscaled](assets/upscaled_closeup.jpg) | ![Basic](assets/basic_closeup.jpg) | ![CRA](assets/cra_closeup.jpg) | ![Tiled](assets/tiled_closeup.jpg) |
| ![Original](assets/original_full.jpg) | ![Upscaled](assets/upscaled_full.jpg) | ![Basic](assets/basic_full.jpg) | ![CRA](assets/cra_full.jpg) | ![Tiled](assets/tiled_full.jpg) |

The table above shows close-up views of the color-adjusted regions (first row) and the full final images (second row) for each technique. Notice how the CRA technique effectively restores the vivid and complex skin colors and the depth of the brightest and darkest blues. The tiled approach further improves the chroma accuracy of the entire gradient, particularly between the darkest and mid blues.

Here are a few more stunning examples showcasing the beautiful color depth achieved using the tiled script:

![Example 1](assets/example1.jpg)

![Example 2](assets/example2.jpg)

![Example 3](assets/example3.jpg)

## Acknowledgements

The code in this repository was primarily written by DeepSeek R1 and Claude Sonnet, with some manual editing. The README was written by Claude Opus.

## License

This project is licensed under the [BSD 3-Clause License](LICENSE.md).
