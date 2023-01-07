import argparse
import os
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(1+seed)
    np.random.seed(21+seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(12+seed)
    torch.cuda.manual_seed_all(123+seed)
    

def read_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--contrastive", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--exp_folder", type=str, default="./jsons/baseline/simple_3")
    
    args = parser.parse_args()
    set_seed(args.seed)
    return args