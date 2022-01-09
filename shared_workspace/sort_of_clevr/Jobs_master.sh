#!/bin/bash






#sh run_transformer.sh 256 4 False False 5 False 8 1 Vanilla


#sh run_transformer.sh 256 4 False False 5 False 8 1 Block

declare -a All_Methods=("Block" "Vanilla")
declare -a All_seeds=(1 2 3)

for Method in "${All_Methods[@]}"
do

	for seed in "${All_seeds[@]}"
	do

		sbatch run_transformer.sh 256 4 False False 5 False 8 $seed $Method
	done
done



