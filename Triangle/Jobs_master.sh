#!/bin/bash

#sh run.sh 4 256 512 True True 20 True 1 8 Vanilla
# run.sh 4 256 512 True True 20 True 1 8 Block
# run.sh 4 256 512 True True 20 True 1 8 Hierachy

#./run.sh 4 256 512 True True 20 True 1 8 VanillaPerceiver
# ./run.sh 4 256 512 True True 20 True 1 8 HierachicalPerceiver




declare -a All_Methods=("VanillaPerceiver" "HierachicalPerceiver" "Hierachy" "Block" "Vanilla")
#declare -a All_Methods=("VanillaPerceiver" "HierachicalPerceiver" )


declare -a All_seeds=(1)

for Method in "${All_Methods[@]}"
do

	for seed in "${All_seeds[@]}"
	do

		sbatch run.sh 4 256 512 True True 20 True 1 8 $Method
	done
done

