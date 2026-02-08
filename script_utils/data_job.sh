#!/bin/bash
#SBATCH -J prep_data
#SBATCH -A COMPUTERLAB-SL3-CPU   # Matches your -A
#SBATCH -p sapphire              # Matches your -p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # Matches your --cpus-per-task
#SBATCH --time=05:00:00          # I increased this to 4h (1h might be tight for downloading)
#SBATCH --output=slurm_prep_%j.out

# 1. Load Environment (Optional but recommended if your .bashrc isn't loaded)
# Your prepare_data.sh handles conda, but loading the module first is safe:
. /etc/profile.d/modules.sh
module purge
module load miniconda/3

# 2. Navigate to your project folder
# (Assuming you submit this from your home directory or the la-proteina folder)
cd $HOME/la-proteina

# 3. Run the script
bash script_utils/prepare_data.sh