import argparse
import copy
import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.aggregate import aggregate, check_representations
from utils.dataloader import CustomDataset
from utils.model import NeuralNetwork
from utils.train_smt import NumpyEncoder, batch_similarity, print_cfmtx, test

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_mask(dim, dataset):
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    mask = torch.zeros([dim, dim])
    for X, y in train_dataloader:
        label = y.item()
        mask[label, label] = 1
    return mask

def train(dataloader, model, loss_fn, optimizer, other_condensed_rep):  
    mask = model.lowrank_mtx
    label_list = torch.diag(mask).nonzero().flatten().tolist()
     
    model = model.cuda()
    model.train()
    losses = []
        
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred, rep = model.pred_and_rep(X)
        
        rep_list = []
        
        for label in label_list:
            label_rep = rep[y==label]
            if len(label_rep):
                rep_list.append(label_rep)
        
        classification_loss = loss_fn(pred, y) 
    
        same_interclass_distance = 0
        for label_rep in rep_list:
            same_interclass_distance += 0.5 * torch.sum(torch.pow(batch_similarity(label_rep, label_rep), 2))/(label_rep.shape[0] * label_rep.shape[0])
    
        diff_interclass_distance = 0
        for i in range(len(rep_list)):
            for j in range(i + 1, len(rep_list)):
                diff_interclass_distance += 0.5 * torch.sum(torch.pow(batch_similarity(rep_list[i], rep_list[j]), 2))/(rep_list[i].shape[0] * rep_list[j].shape[0])
        
        diff_interclient_distance = torch.sum(torch.pow(batch_similarity(rep, other_condensed_rep),2))/(rep.shape[0] * other_condensed_rep.shape[0])
        
        classification_loss = loss_fn(pred, y)
        loss = classification_loss + same_interclass_distance - diff_interclass_distance - diff_interclient_distance
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        
    return losses

def train_representation(dataloader, model, condense_representation, other_condensed_rep):
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred, rep = model.pred_and_rep(X)
        condense_representation.retain_grad()
        sim = batch_similarity(rep, condense_representation)
        # sim_to_others = batch_similarity(condense_representation, other_condensed_rep)
        condense_represent_loss = torch.sum(1 - sim)  #+ 0.25 * torch.sum(sim_to_others)
        condense_represent_loss.backward()
        condense_representation = condense_representation - 5 * condense_representation.grad

    return condense_representation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--round", type=int, default=1)
    args = parser.parse_args()
    batch_size = args.batch_size
    epochs = args.epochs

    training_data = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
    )
    
    testing_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
    )
    
    client_id_list = [0,1,2,3,4]
    clients_dataset = [CustomDataset(training_data, json.load(open(f"./jsons/client{client_id}.json", 'r'))) for client_id in client_id_list]
    
    global_model = NeuralNetwork(bias=False).to(device)
    local_loss_record = {client_id:[] for client_id in client_id_list}
    local_cfmtx_bfag_record = {client_id:[] for client_id in client_id_list}
    local_cfmtx_afag_record = {client_id:[] for client_id in client_id_list}
    global_cfmtx_record = []
    U_cfmtx_record = []
    condense_representation_list = [torch.randn([512], requires_grad=True).unsqueeze(0) for client_id in client_id_list]
    
    clients_mask = [None for client_id in client_id_list]
    
    for cur_round in range(args.round):
        print("============ Round {} ==============".format(cur_round))
        
        client_models = []
        
        # Local training
        for client_id in client_id_list:
            print("    Client {} training... ".format(client_id), end="")
            # Training process
            mydataset = clients_dataset[client_id]
            train_dataloader = DataLoader(mydataset, batch_size=batch_size, shuffle=True, drop_last=False)
            
            if clients_mask[client_id] is None:
                clients_mask[client_id] = create_mask(dim=10, dataset=mydataset)
            
            local_model = copy.deepcopy(global_model)
            local_model.update_mask(clients_mask[client_id])

            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(local_model.parameters(), lr=1e-3)

            # Other representations
            other_clients_representation = [condense_representation_list[i] for i in client_id_list if i != client_id]
            other_condensed_rep = torch.cat(other_clients_representation, dim=0).cuda().detach()
            
            epoch_loss = []
            for t in range(epochs):
                epoch_loss.append(np.mean(train(train_dataloader, local_model, loss_fn, optimizer, other_condensed_rep)))
            local_loss_record[client_id].append(np.mean(epoch_loss))
            
            client_models.append(local_model)
            
            # Testing the local_model to its own data
            cfmtx = test(local_model, mydataset)
            local_cfmtx_bfag_record[client_id].append(cfmtx)
            
            # Regenerating representations
            condense_representation = condense_representation_list[client_id].cuda()
            for t in range(epochs):
                condense_representation = train_representation(train_dataloader, local_model, condense_representation, other_condensed_rep)
            condense_representation = condense_representation.detach().cpu()
            local_model.zero_grad()
            print(f"Done! Aver. round loss: {np.mean(epoch_loss):>.3f}")
        
        print("    # Server aggregating... ", end="")
        # Aggregation
        condensed_rep = torch.cat(condense_representation_list, dim=0).detach()
        global_model = aggregate(client_models, condensed_rep, indexes=client_id_list)
        print("Done!")
        
        # Testing
        print("    # Server testing... ", end="")
        cfmtx = test(global_model, testing_data)
        global_cfmtx_record.append(cfmtx)
        print("Done!")
        np.set_printoptions(precision=2, suppress=True)
        print('-'*20, "Classification confussion matrix", '-'*20)
        print_cfmtx(cfmtx)
        
        U_cfmtx = check_representations(global_model, condensed_rep, testing_data, "cuda")
        print('-'*20, "U selection confussion matrix", '-'*20)
        print_cfmtx(U_cfmtx)
        U_cfmtx_record.append(U_cfmtx)
    
    if not Path("records/proposal2").exists():
        os.makedirs("records/proposal2")
    
    json.dump(local_loss_record,        open("records/proposal2/local_loss_record.json", "w"),         cls=NumpyEncoder)
    json.dump(local_cfmtx_bfag_record,  open("records/proposal2/local_cfmtx_bfag_record.json", "w"),   cls=NumpyEncoder)
    json.dump(global_cfmtx_record,      open("records/proposal2/global_cfmtx_record.json", "w"),       cls=NumpyEncoder)
    json.dump(U_cfmtx_record,           open("records/proposal2/U_cfmtx_record.json", "w"),            cls=NumpyEncoder)