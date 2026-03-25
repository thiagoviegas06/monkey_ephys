#!/bin/bash
#SBATCH --job-name=kin_train_eval_model
#SBATCH --account=torch_pr_173_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:h200:1
#SBATCH --output=kin_logs/%x_%j.out
#SBATCH --error=kin_logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dr3432@nyu.edu

# Ensure log directory exists locally before Slurm starts redirection (if possible)
# Note: Slurm needs the directory to exist at the time the job is queued/started.
mkdir -p kin_logs

set -euo pipefail

SIF="/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
OVL="/scratch/dr3432/pytorch/pytorch_env.ext3:ro"

echo "Using Singularity Image: $SIF"
echo "Using Overlay: $OVL"
echo "Job Started: $(date)"

# Use singularity to run the training and evaluation in the specified environment
singularity exec --nv \
	--overlay $OVL \
	--fakeroot \
	$SIF \
	/bin/bash <<'EOF'
set -euo pipefail

# Verify environment script exists
if [ -f /ext3/env.sh ]; then
    source /ext3/env.sh
else
    echo "Error: /ext3/env.sh not found inside overlay!"
    exit 1
fi

echo "Python version: $(python --version)"
echo "Starting Training Phase..."
python src_kin/train.py 

echo "Starting Evaluation Phase..."
python src_kin/eval.py 

echo "Job Completed: $(date)"
EOF
