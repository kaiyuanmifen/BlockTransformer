# block transformer with hierachical global workspace

This repository contains the code to the project  block transformer with hierachical global workspace with an option to use Gflownet as second layer of GWS ( to be implemented)


## Install relevant libraries
```
pip install -r requirements.txt 
```
## Detecting Equilateral Triangles 
Folder: Triangle/

The following commands to be executed from inside in the `Triangle` folder.

```
sh run.sh num_layers h_dim ffn_dim share_vanilla_parameters use_topk topk shared_memory_attention seed mem_slots Method 

Where options of "Method" including  "Vanilla" "Block" "Hierachy"  "VanillaPerceiver" "HierachicalPerceiver" 
correspinding to vanilla transformer, block transformer, hierachical transformer, Vanilla perceiver and Hierachical perceiver

use_topk: Whether to use top-k competition

topk: Value of k in top-k competition

shared_memory_attention: Whether to use shared workspace

mem_slots: Number of slots in memory
```

To submit the jobs in slurm cluster
```
./Jobs_master.sh

```

## Sort-of-CLEVR
The following commands to be executed from inside in the `sort_of_clevr` folder.

Dataset generation:
```
python sort_of_clevr_generator.py
```

```
sh run_transformer.sh h_dim num_layers share_vanilla_parameters use_topk topk shared_memory_attention mem_slots seed
```
To reproduce experiments in paper:
```
TR + HSW
sh run_transformer.sh 256 4 True True 5 True 8 1 False

TR
sh run_transformer.sh 256 4 True False 5 False 8 1 False

STR
sh run_transformer.sh 256 4 True True 5 False 8 1 False

TR + HC
sh run_transformer.sh 256 4 False False 5 False 8 1 False

ISAB
sh run_transformer.sh 256 4 False False 5 False 8 1 True


```

