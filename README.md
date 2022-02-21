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
Run transformer model 
```
./run_transformer.sh h_dim num_layers share_vanilla_parameters use_topk topk shared_memory_attention mem_slots seed Method
```

Run Perceiver model 
```
./run_perceiver.sh h_dim num_layers share_vanilla_parameters use_topk topk shared_memory_attention mem_slots seed Method
```

sh run.sh num_layers h_dim ffn_dim share_vanilla_parameters use_topk topk shared_memory_attention seed mem_slots Method 

Where options of "Method" including  "Vanilla" "Block" "Hierachy"  "VanillaPerceiver" "HierachicalPerceiver" 
correspinding to vanilla transformer, block transformer, hierachical transformer, Vanilla perceiver and Hierachical perceiver

Submit jobs in slurm clusters
```
./Jobs_master.sh

```



