#!/usr/bin/env python3
"""
Test script to verify the modified generate_bus_info.py script
"""

import os
import sys
import subprocess
from pathlib import Path

def test_bus_script():
    """Test the modified bus script"""
    
    print("=== Testing Modified Bus Script ===")
    
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
    
    # Test with train split
    print("\n--- Testing Train Split ---")
    try:
        result = subprocess.run([
            sys.executable, "scripts/generate_bus_info.py",
            "--data_dir", str(bus_data_dir),
            "--output_dir", "test_bus_train",
            "--split", "train"
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("✓ Train split processing successful!")
            print("Output:")
            print(result.stdout)
        else:
            print("✗ Train split processing failed!")
            print("Error:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Error running train split: {e}")
        return False
    
    # Test with test split
    print("\n--- Testing Test Split ---")
    try:
        result = subprocess.run([
            sys.executable, "scripts/generate_bus_info.py",
            "--data_dir", str(bus_data_dir),
            "--output_dir", "test_bus_test",
            "--split", "test"
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("✓ Test split processing successful!")
            print("Output:")
            print(result.stdout)
        else:
            print("✗ Test split processing failed!")
            print("Error:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Error running test split: {e}")
        return False
    
    # Check output files
    print("\n--- Checking Output Files ---")
    for output_dir in ["test_bus_train", "test_bus_test"]:
        if Path(output_dir).exists():
            print(f"\n{output_dir}:")
            files = list(Path(output_dir).glob("*"))
            for file_path in sorted(files):
                if file_path.is_file():
                    size = file_path.stat().st_size
                    print(f"  {file_path.name} ({size} bytes)")
                else:
                    print(f"  {file_path.name}/ (directory)")
    
    print("\n=== Test Complete ===")
    return True

if __name__ == "__main__":
    success = test_bus_script()
    sys.exit(0 if success else 1)
