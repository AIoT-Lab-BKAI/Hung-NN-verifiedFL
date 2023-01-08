from utils.train_smt import test
from utils.reader import read_jsons
from utils.parser import read_arguments

from utils import fmodule
from pathlib import Path
from torch.utils.data import DataLoader
from utils.FIM2 import MLP2, FIM2_step
from utils.FIM3 import MLP3, FIM3_step
import torch, json, os, numpy as np, copy, random
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"


def train(dataloader, model, optimizer):   
    model = model.cuda()
    model.train()
    losses = []
        
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = F.softmax(model(X), dim=1) 
        ground = F.softmax(F.one_hot(y, 10) * 1.0, dim=1) 
        loss = torch.sum(ground * torch.log(ground/pred)) / (ground.shape[0] * ground.shape[1])
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        
    return losses


if __name__ == "__main__":
    args = read_arguments()
    print(args)
    batch_size = args.batch_size
    epochs = args.epochs
        
    num_client, clients_training_dataset, clients_testing_dataset, global_testing_dataset, singleset = read_jsons(args.exp_folder, args.dataset)
    client_id_list = [i for i in range(num_client)]
    total_sample = np.sum([len(dataset) for dataset in clients_training_dataset])
    
    if args.dataset == "mnist":
        global_model = MLP2().to(device)
        fim_step = FIM2_step
    elif args.dataset == "cifar10":
        global_model = MLP3().to(device)
        fim_step = FIM3_step
    else:
        raise NotImplementedError
    
    
    centroid = copy.deepcopy(global_model)
    
    for cur_round in range(args.round):
        print("============ Round {} ==============".format(cur_round))
        client_models_this_round = []
        client_id_list_this_round = np.random.choice(client_id_list, size=10, replace=False).tolist()
        total_sample_this_round = np.sum([len(clients_training_dataset[i]) for i in client_id_list_this_round])
        impact_factors = [len(clients_training_dataset[client_id])/total_sample_this_round for client_id in client_id_list_this_round]
        
        # Local training
        for client_id in client_id_list_this_round:
            print(f"Client {client_id:>2d} training... ", end="")
            # Training process
            my_training_dataset = clients_training_dataset[client_id]
            my_testing_dataset = clients_testing_dataset[client_id]
        
            local_model = copy.deepcopy(global_model)
            
            train_dataloader = DataLoader(my_training_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=1e-3)
            
            epoch_loss = []
            for t in range(epochs):
                epoch_loss.append(np.mean(train(train_dataloader, local_model, optimizer)))
            
            client_models_this_round.append(local_model)
            
            # Testing the local_model
            test_acc, _ = test(local_model, my_testing_dataset)
            
            norm_diff = (local_model - global_model).norm()
            print(f"Done! Aver. round loss: {np.mean(epoch_loss):>.3f}, test acc {test_acc:>.3f}, shift length {norm_diff:>.5f}")

        centroid = fmodule._model_sum([model * pk for model, pk in zip(client_models, impact_factors)])
        global_model = fim_step(centroid, clients_training_dataset, client_id_list, eta=1).to(device)
        
        # Testing
        print("Server testing... ", end="")
        acc, cfmtx = test(global_model, global_testing_dataset, device)
        print(f"Done! Avg. acc {acc:>.3f}")

    # print_cfmtx(cfmtx)
    