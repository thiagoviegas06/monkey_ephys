#!/bin/bash
#SBATCH --job-name=train_eval_model
#SBATCH --account=torch_pr_173_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:h200:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=dr3432@nyu.edu

set -euo pipefail

mkdir -p logs


SIF="/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
OVL="/scratch/dr3432/pytorch/pytorch_env.ext3:ro"

echo $SIF
echo $OVL

singularity exec --nv \
	--overlay $OVL \
	--fakeroot \
	$SIF \
	/bin/bash <<'EOF'
set -euo pipefail

source /ext3/env.sh

python src_mae/train.py --window-size 400
python src_mae/eval.py --window-size 400 --model-path ./checkpoints_400/best_model_tcn_transformer.pth


EOF
