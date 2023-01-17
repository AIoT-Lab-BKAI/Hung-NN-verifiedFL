import argparse
import os
import random
import numpy as np
import torch
import wandb
from pathlib import Path

def set_seed(seed):
    random.seed(1+seed)
    np.random.seed(21+seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(12+seed)
    torch.cuda.manual_seed_all(123+seed)
    

def read_arguments(algorithm):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--wandb", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--client_per_round", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--data_folder", type=str, default="./data")
    parser.add_argument("--idx_folder", type=str, default="./jsons/dataset_idx/mnist/dirichlet/dir_0.1_sparse")
    parser.add_argument("--log_folder", type=str, default="./records")
    args = parser.parse_args()
    set_seed(args.seed)
    
    groupname = " ".join(args.idx_folder.split('/')[3:])
    print("Group name:", groupname)
    print("Run name:", algorithm)
        
    if args.wandb:
        wandb.init(
            project="over-param-FL", 
            entity="aiotlab",
            group=groupname,
            name=algorithm,
            config=args
        )
        
    return args