#!/bin/bash
#SBATCH --job-name=train_sort_of_clevr
#SBATCH --gres=gpu:1             # Number of GPUs (per node)
#SBATCH --mem=60G               # memory (per node)
#SBATCH --time=0-8:00            # time (DD-HH:MM)

###########cluster information above this line


###load environment 

module load anaconda/3
module load cuda/10.1
conda activate AIAYNTransformer




embed_dim=$1
num_layers=$2
share_vanilla_parameters=$3
use_topk=$4
topk=$5
shared_memory_attention=$6
mem_slots=$7
null_attention=False
seed=${8}
Method=$9



save_dir=Exp-$embed_dim-$num_layers-$Method-$share_vanilla_parameters-$use_topk-$topk-$shared_memory_attention-$mem_slots-$null_attention-$seed

mkdir $save_dir

python main.py --model Transformer --epochs 100 --embed_dim $embed_dim --num_layers $num_layers \
			   --Method $Method --share_vanilla_parameters $share_vanilla_parameters \
			   --use_topk $use_topk --topk $topk --shared_memory_attention $shared_memory_attention \
			   --save_dir $save_dir --mem_slots $mem_slots --null_attention $null_attention \
			   --seed $seed



