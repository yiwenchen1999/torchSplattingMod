#!/usr/bin/env python3
"""
Demonstration script to show how the cleanup functionality works
"""

import os
import sys
from pathlib import Path

def demonstrate_cleanup_logic():
    """Demonstrate the cleanup logic with examples"""
    
    print("=== Cleanup Logic Demonstration ===")
    
    # Example directory structure
    print("Example directory structure:")
    print("datasamples/objaverse_synthetic/shop/train/")
    print("├── depth/                    # ✅ KEPT (depth maps)")
    print("│   ├── depth_0.exr")
    print("│   ├── depth_1.exr")
    print("│   └── ...")
    print("├── white_env_0/              # ✅ KEPT (RGB images)")
    print("│   ├── gt_0.png")
    print("│   ├── gt_1.png")
    print("│   └── ...")
    print("├── black_env_0/              # ❌ REMOVED (other environment)")
    print("│   ├── gt_0.png")
    print("│   ├── gt_1.png")
    print("│   └── ...")
    print("├── colored_env_0/            # ❌ REMOVED (other environment)")
    print("│   ├── gt_0.png")
    print("│   ├── gt_1.png")
    print("│   └── ...")
    print("├── rgb_for_depth_86.png      # ❌ REMOVED (root level)")
    print("├── depth_310001.exr          # ❌ REMOVED (root level)")
    print("└── other_file.txt            # ✅ KEPT (not image file)")
    
    print("\n=== What Gets Cleaned Up ===")
    print("The script will remove:")
    print("1. Root-level .png and .exr files (e.g., rgb_for_depth_86.png)")
    print("2. Files in other environment folders (e.g., black_env_0/, colored_env_0/)")
    print("3. Any other .png or .exr files not in depth/ or white_env_0/")
    
    print("\n=== What Gets Kept ===")
    print("The script will keep:")
    print("1. All files in depth/ folder (depth maps)")
    print("2. All files in white_env_0/ folder (RGB images)")
    print("3. Non-image files (e.g., .txt, .json, etc.)")
    
    print("\n=== Usage Examples ===")
    print("# Preview what would be cleaned up:")
    print("python scripts/generate_bus_info.py --split train --cleanup_input --dry_run")
    
    print("\n# Actually clean up:")
    print("python scripts/generate_bus_info.py --split train --cleanup_input")
    
    print("\n# Process dataset and clean up in one command:")
    print("python scripts/generate_bus_info.py --split train --data_dir datasamples/objaverse_synthetic/shop --output_dir datasamples/shop_processed --cleanup_input")

if __name__ == "__main__":
    demonstrate_cleanup_logic()
