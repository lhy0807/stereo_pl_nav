#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:v100-sxm2:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --output=myjob.%j.out
#SBATCH --error=myjob.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hongyu@ccs.neu.edu

module load gcc/10.1.0
. "/shared/centos7/anaconda3/2022.01/etc/profile.d/conda.sh"
conda activate /scratch/li.hongyu1/torch/

python main.py