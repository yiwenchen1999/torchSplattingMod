# Ship Latents Preprocessing for torchSplattingMod

This directory contains scripts to preprocess the `ship_latents` dataset for use with torchSplattingMod.

## Overview

The `ship_latents` dataset contains RGB images in the `test/` folder with corresponding depth data, but lacks alpha masks that torchSplattingMod requires. This preprocessing script:

1. **Uses only test set data** (ignoring train set entirely)
2. **Generates alpha masks** from RGB images using thresholding
3. **Uses depth data** from the same test folder
4. **Creates the required `info.json`** file in the format expected by torchSplattingMod
5. **Copies all necessary files** to a processed output directory

## Files

- `preprocess_ship_latents.py` - Main Python preprocessing script
- `preprocess_ship_latents.sh` - Shell script wrapper for easy execution
- `README_preprocessing.md` - This documentation file

## Usage

### Option 1: Using the shell script (Recommended)

```bash
# Navigate to torchSplattingMod directory
cd torchSplattingMod

# Make the script executable
chmod +x scripts/preprocess_ship_latents.sh

# Run the preprocessing
./scripts/preprocess_ship_latents.sh
```

### Option 2: Using Python directly

```bash
# Navigate to torchSplattingMod directory
cd torchSplattingMod

python scripts/preprocess_ship_latents.py \
    --ship_latents_dir ../ship_latents \
    --output_dir ../ship_latents_processed \
    --transforms_file transforms_test.json
```

### Command Line Arguments

- `--ship_latents_dir`: Path to the ship_latents directory (default: `../ship_latents`)
- `--output_dir`: Output directory for processed data (default: `../ship_latents_processed`)
- `--transforms_file`: Transforms file to process (default: `transforms_test.json`)

## Output

The script creates a new directory (`ship_latents_processed` by default) containing:

- `info.json` - The main configuration file for torchSplattingMod
- RGB images (copied from test folder)
- Alpha masks (generated from RGB images)
- Depth images (copied from test folder, where available)

## Using with torchSplattingMod

After preprocessing, you can use the data with torchSplattingMod by:

1. **The processed data** will be created in the parent directory as `ship_latents_processed`

2. **Update the folder path** in `train.py`:
   ```python
   folder = '../ship_latents_processed'  # Change this line
   ```

3. **Run the training**:
   ```bash
   cd torchSplattingMod
   python train.py
   ```

## How It Works

### Alpha Mask Generation
The script creates alpha masks by:
1. Converting RGB images to grayscale
2. Applying a threshold (value 10) to separate foreground from background
3. Assuming dark/black backgrounds

### Depth Data Mapping
The script uses depth data from the test folder:
1. Looks for corresponding depth files in the same test directory
2. Uses patterns like `r_0_depth_0002.png`, `r_0_depth.png`, etc.
3. Copies matching depth files to the output directory

### Camera Parameters
The script:
1. Reads camera parameters from the transforms file
2. Calculates focal length from the camera angle
3. Creates proper intrinsic matrices
4. Converts transform matrices to the expected format

## Troubleshooting

### Missing Dependencies
Install required packages:
```bash
pip install numpy imageio pillow opencv-python
```

### No Depth Files Found
If no depth files are found in the test folder, the script will still work but without depth supervision. The training will proceed with `lambda_depth = 0.0`.

### Image Size Issues
The script assumes 512x512 images. If your images are different sizes, modify the `image_width` variable in the script.

## Notes

- The depth loss weight (`lambda_depth`) is set to 0.0 by default in torchSplattingMod, so depth data is not used for training even if available
- Alpha masks are generated using simple thresholding and may need manual adjustment for optimal results
- The script processes the test set by default, which contains both RGB and depth data
