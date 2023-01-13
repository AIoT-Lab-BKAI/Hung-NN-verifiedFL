from utils.train_smt import test, NumpyEncoder
from utils.reader import read_jsons
from utils.parser import read_arguments

from pathlib import Path
from torch.utils.data import DataLoader
from utils.FIM2 import MLPv2
from utils.FIM3 import MLPv3
from utils import fmodule
import torch, json, os, numpy as np, copy, random
import torch.nn.functional as F
import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
    
if __name__ == "__main__":
    args = read_arguments(algorithm=os.path.basename(__file__).split('.py')[0])
    print(args)
    batch_size = args.batch_size
    epochs = args.epochs
    
    _, _, _, global_testing_dataset, singleset = read_jsons(args.idx_folder, args.data_folder, args.dataset)
    
    if args.dataset == "mnist":
        global_model = MLPv2().to(device)
    elif args.dataset == "cifar10":
        global_model = MLPv3().to(device)
    else:
        raise NotImplementedError
    
    results = {}
    
    train_dataloader = DataLoader(singleset, batch_size=batch_size, shuffle=True, drop_last=False)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(global_model.parameters(), lr=1e-3)
    
    for t in range(500):
        print(f"Epoch {t+1} ...", end="")
        train(train_dataloader, global_model, loss_fn, optimizer)
        acc, _ = test(global_model, global_testing_dataset)
        print(f"Done! Avg. acc {acc:>.3f}")
        
    if not Path(f"{args.log_folder}/{args.idx_folder}/singleset").exists():
        os.makedirs(f"{args.log_folder}/{args.idx_folder}/singleset")
    
    results['fin_acc'] = acc
    json.dump(results, open(f"{args.log_folder}/{args.idx_folder}/singleset/results.json", "w"),cls=NumpyEncoder)
    