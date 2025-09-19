#!/bin/bash

# Example script to run Objaverse processing
# Modify parameters as needed

echo "Starting Objaverse data processing..."

# Option 1: Dry run to see what would be processed
echo "=== DRY RUN ==="
python process_all_objaverse.py --dry_run --max_objects 5

echo ""
echo "=== ACTUAL PROCESSING ==="

# Option 2: Process first 10 objects (for testing)
python process_all_objaverse.py --max_objects 10

# Option 3: Process all objects (uncomment when ready)
# python scripts/process_all_objaverse.py

# Option 4: Resume from a specific object (if processing was interrupted)
# python scripts/process_all_objaverse.py --start_from "74ec82f3e6b34b2ca7886d2ee789f8f9"

echo "Processing complete!"
