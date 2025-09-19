#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --job-name=process_objaverse
#SBATCH --mem=32
#SBATCH --ntasks=8
#SBATCH --output=myjob.process_objaverse.out
#SBATCH --error=myjob.process_objaverse.err


# Option 2: Process first 10 objects (for testing)
python process_all_objaverse.py

# Option 3: Process all objects (uncomment when ready)
# python scripts/process_all_objaverse.py

# Option 4: Resume from a specific object (if processing was interrupted)
# python scripts/process_all_objaverse.py --start_from "74ec82f3e6b34b2ca7886d2ee789f8f9"

echo "Processing complete!"
