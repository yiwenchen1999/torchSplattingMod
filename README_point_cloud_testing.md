# Point Cloud Testing Scripts

This directory contains scripts to test the `get_point_clouds` function from the Gaussian Splatting implementation.

## Overview

The `get_point_clouds` function converts depth maps and RGB images into 3D point clouds using ray casting. These scripts help you:

1. Test the function with your data
2. Visualize the generated point clouds
3. Analyze point cloud properties
4. Verify the conversion process

## Files

- `test_point_clouds.py` - Comprehensive testing script with detailed analysis and visualizations
- `test_point_clouds_simple.py` - Simplified testing script for basic functionality
- `README_point_cloud_testing.md` - This documentation file

## Prerequisites

Make sure you have the following dependencies installed:

```bash
pip install torch numpy matplotlib scipy
```

## Usage

### Simple Testing

For basic testing of the `get_point_clouds` function:

```bash
cd torchSplattingMod
python test_point_clouds_simple.py
```

This script will:
- Automatically find available data folders
- Load the data using the same method as `train.py`
- Generate point clouds using `get_point_clouds`
- Create 2D visualizations
- Test basic point cloud operations

### Comprehensive Testing

For detailed analysis and testing:

```bash
cd torchSplattingMod
python test_point_clouds.py
```

This script provides:
- Detailed data statistics
- Point cloud analysis
- 2D and 3D visualizations
- Point density analysis
- Comparison with original data
- Testing of all point cloud operations

## Expected Output

The scripts will generate:

1. **Console output** with detailed statistics and analysis
2. **Visualization files**:
   - `point_cloud_simple.png` (simple script)
   - `point_cloud_2d.png` and `point_cloud_3d.png` (comprehensive script)

## Data Requirements

The scripts expect data in the same format as used in `train.py`:

- RGB images
- Depth maps
- Alpha masks
- Camera parameters
- VAE latents (optional)

The data should be organized in a folder structure compatible with the `read_all()` function from `data_utils.py`.

## Understanding the Results

### Point Cloud Generation

The `get_point_clouds` function:
1. Takes camera parameters, depth maps, alpha masks, and RGB images
2. Uses ray casting to convert 2D depth pixels to 3D points
3. Returns a `PointCloud` object with coordinates and color channels

### Key Metrics to Check

1. **Number of points**: Should match the number of valid depth pixels
2. **Coordinate ranges**: Should be reasonable for your scene
3. **Color values**: Should match the original RGB images
4. **Alpha values**: Should indicate valid/invalid points

### Common Issues

1. **No points generated**: Check if alpha masks are all zero
2. **Incorrect coordinate ranges**: Verify depth values and camera parameters
3. **Missing color channels**: Ensure RGB data is properly loaded
4. **Memory issues**: Use the simple script for large datasets

## Customization

You can modify the scripts to:

- Test with different data folders
- Adjust visualization parameters
- Add custom analysis functions
- Test specific point cloud operations

## Troubleshooting

### Data Loading Issues

If the script can't find your data:

1. Check the data folder paths in `find_data_folder()`
2. Ensure the data format matches the expected structure
3. Verify that all required files (RGB, depth, alpha) are present

### Import Errors

If you get import errors:

1. Make sure you're running from the `torchSplattingMod` directory
2. Check that all required modules are available
3. Verify the Python path includes the current directory

### Visualization Issues

If visualizations fail:

1. Check if matplotlib is properly installed
2. Ensure you have write permissions in the current directory
3. Try reducing the number of points for visualization

## Integration with Training

These test scripts use the same data loading and processing pipeline as the main training script (`train.py`). This ensures that:

- The point cloud generation is consistent with training
- Data preprocessing is identical
- Camera parameters are handled the same way

You can use the insights from these tests to:
- Debug training issues
- Optimize data preprocessing
- Validate point cloud quality before training
- Understand the relationship between 2D images and 3D geometry
