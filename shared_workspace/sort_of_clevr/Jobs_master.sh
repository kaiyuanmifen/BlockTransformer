#!/bin/bash






#sh run_transformer.sh 256 4 False False 5 False 8 1 Vanilla


#sh run_transformer.sh 256 4 False False 5 False 8 1 Block

#sh run_transformer.sh 256 4 False False 5 False 8 1 Hierachy


#sh run_perceiver.sh 256 4 False False 5 False 8 1 VanillaPerceiver

#sh run_perceiver.sh 256 4 False False 5 False 8 1 HierachicalPerceiver

#####transformer version

# #declare -a All_Methods=("Hierachy" "Block" "Vanilla")
# declare -a All_Methods=("Hierachy" "Block" "Vanilla")
# declare -a All_seeds=(2 3)

# for Method in "${All_Methods[@]}"
# do

# 	for seed in "${All_seeds[@]}"
# 	do

# 		sbatch run_transformer.sh 256 4 False False 5 False 8 $seed $Method
# 	done
# done



####perceiver version
declare -a All_Methods=("HierachicalPerceiver" "VanillaPerceiver")
declare -a All_seeds=(1)

for Method in "${All_Methods[@]}"
do

	for seed in "${All_seeds[@]}"
	do

		sbatch run_perceiver.sh 256 4 False False 5 False 8 $seed $Method
	done
done

