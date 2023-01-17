from utils.train_smt import test
from utils.reader import read_jsons
from utils.parser import read_arguments

from utils import fmodule
from torch.utils.data import DataLoader
from utils.FIM2 import MLP2, FIM2_step
from utils.FIM3 import MLP3, FIM3_step
import torch, numpy as np, copy, json
import torch.nn.functional as F
from pathlib import Path
import os


device = "cuda" if torch.cuda.is_available() else "cpu"


def train(dataloader, model, optimizer):   
    model = model.cuda()
    model.train()
    losses = []
    loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
        
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = F.log_softmax(model(X), dim=1) 
        ground = F.softmax(F.one_hot(y, 10) * 1.0, dim=1)
        loss = loss_fn(pred, ground)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        
    return losses


if __name__ == "__main__":
    args = read_arguments(algorithm=os.path.basename(__file__).split('.py')[0])
    print(args)
    batch_size = args.batch_size
    epochs = args.epochs
        
    num_client, clients_training_dataset, clients_testing_dataset, global_testing_dataset, singleset = read_jsons(args.idx_folder, args.data_folder, args.dataset)
    client_id_list = [i for i in range(num_client)]
    total_sample = np.sum([len(dataset) for dataset in clients_training_dataset])
    results = {}
    
    if args.dataset == "mnist":
        global_model = MLP2().to(device)
        fim_step = FIM2_step
    elif args.dataset == "cifar10":
        global_model = MLP3().to(device)
        fim_step = FIM3_step
    else:
        raise NotImplementedError
    
    if args.load_model_path is None:
        raise ValueError("To run fim, load_model_path must not be None")
    
    global_model.load_state_dict(torch.load(args.load_model_path))
    
    print("Origin model testing... ", end="")
    acc, cfmtx = test(global_model, global_testing_dataset, device)
    print(f"Done! Avg. acc {acc:>.3f}")
    results['origin'] = acc
    
    global_model = fim_step(global_model, clients_training_dataset, client_id_list, eta=1, device=device)
    print("Origin+fim testing... ", end="")
    acc, cfmtx = test(global_model, global_testing_dataset, device)
    print(f"Done! Avg. acc {acc:>.3f}")
    results['+fim'] = acc
    
    save_path = os.path.join(*args.load_model_path.split('/')[:-1], "fim.json")
    json.dump(results, open(save_path, "w"))