srun --partition=jiang --nodes=1 --pty --gres=gpu:a5000:1 --ntasks=8 --mem=32 --time=12:00:00 /bin/bash
cd /projects/vig/yiwenc/ResearchProjects/lightingDiffusion/3dgs/torchSplattingMod
conda activate /projects/vig/yiwenc/all_env/torchSplatting
