#!/usr/bin/env python3
"""
Test script to demonstrate the new cleanup functionality in generate_bus_info.py
"""

import os
import sys
import subprocess
from pathlib import Path

def test_cleanup_functionality():
    """Test the new cleanup functionality"""
    
    print("=== Testing Cleanup Functionality ===")
    
    # Check if the script exists
    script_path = Path("scripts/generate_bus_info.py")
    if not script_path.exists():
        print(f"Error: Script not found at {script_path}")
        return False
    
    # Check if we have a bus dataset to test with
    bus_data_dir = Path("../datasamples/bus")
    if not bus_data_dir.exists():
        print(f"Warning: Bus dataset not found at {bus_data_dir}")
        print("Skipping actual execution test")
        return True
    
    print(f"Found bus dataset at: {bus_data_dir}")
    
    # Test dry-run mode first (safe)
    print("\n--- Testing Dry-Run Mode (Safe Preview) ---")
    try:
        result = subprocess.run([
            sys.executable, "scripts/generate_bus_info.py",
            "--data_dir", str(bus_data_dir),
            "--output_dir", "test_cleanup_output",
            "--split", "train",
            "--cleanup_input",
            "--dry_run"
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("✓ Dry-run cleanup successful!")
            print("Output:")
            print(result.stdout)
        else:
            print("✗ Dry-run cleanup failed!")
            print("Error:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Error running dry-run cleanup: {e}")
        return False
    
    # Show usage examples
    print("\n=== Usage Examples ===")
    print("1. Preview what would be cleaned up (safe):")
    print("   python scripts/generate_bus_info.py --split train --cleanup_input --dry_run")
    
    print("\n2. Actually clean up the input directory:")
    print("   python scripts/generate_bus_info.py --split train --cleanup_input")
    
    print("\n3. Process dataset and clean up input in one command:")
    print("   python scripts/generate_bus_info.py --split train --data_dir datasamples/bus --output_dir datasamples/bus_processed --cleanup_input")
    
    print("\n=== What Gets Cleaned Up ===")
    print("The script will remove .png and .exr files that are NOT in:")
    print("  - depth/ folder (depth maps)")
    print("  - white_env_0/ folder (RGB images)")
    print("\nFiles that WILL be removed:")
    print("  - Files in other environment folders (e.g., black_env_0/, colored_env_0/)")
    print("  - Root-level files (e.g., rgb_for_depth_86.png, depth_310001.exr)")
    print("  - Any other .png or .exr files not in the protected folders")
    
    return True

if __name__ == "__main__":
    success = test_cleanup_functionality()
    sys.exit(0 if success else 1)
