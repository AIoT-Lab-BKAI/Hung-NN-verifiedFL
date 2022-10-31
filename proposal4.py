import argparse
import copy
import json
import os
from pathlib import Path

import random
import utils.fmodule as fmodule
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.dataloader import CustomDataset
from utils.proposal_model import DNN_proposal, augment_model
from utils.train_smt import NumpyEncoder, print_cfmtx
from torchmetrics import ConfusionMatrix

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_mask_diagonal(dim, dataset):
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    mask = torch.zeros([dim])
    for X, y in train_dataloader:
        label = y.item()
        mask[label] = 1
    return mask

def classification_training(dataloader, model, loss_fn, optimizer, original_mask, device="cuda:1"):
    original_mask = original_mask.to(device)
    model = model.to(device)
    model.train()
    losses = []
        
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X, original_mask)
        classification_loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        classification_loss.backward()
        optimizer.step()
        losses.append(classification_loss.item())
    
    # print("classification losses", losses)
    return np.mean(losses)

def mask_training(dataloader, model, optimizer, original_mask_diagonal, device="cuda:1"):
    """
    This method trains to make the model generate a mask that is
    close to the original mask
    """
    original_mask_diagonal = original_mask_diagonal.to(device)
    model = model.to(device)
    model.train()
    
    losses = []
    storage = []
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        representation = model.encoder(X)
        mirr_mask_diagonal = model.mask_diagonal_regenerator(representation)
        storage.append((representation.detach().cpu(), mirr_mask_diagonal.detach().cpu()))
        
        loss = torch.sum(torch.pow(mirr_mask_diagonal - original_mask_diagonal, 2))/mirr_mask_diagonal.shape[0]
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        mirr_mask_diagonal = (mirr_mask_diagonal > 1/10) * 1.0
        loss = torch.sum(torch.abs(mirr_mask_diagonal - original_mask_diagonal))/mirr_mask_diagonal.shape[0]
        losses.append(loss.item())
        
    # print("Masking losses", losses)
    return np.mean(losses), storage

def representation_training(dataloader, model, loss_fn, optimizer, device="cuda:1"):
    """
    This method trains for a discriminative representation space,
    using constrastive learning
    
    Args:
        dataloader: batch_size of 2, drop_last = True
        loss_fn:    mean square error
    """
    model = model.to(device)
    model.train()
    same_class_dis = []
    different_class_dis = []
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        representations = model.encoder(X)
        
        alpha = 1.0 if y[0].item() == y[1].item() else -1.0
        distance = loss_fn(representations[0], representations[1])
        loss = alpha * distance
                
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if alpha > 0:
            same_class_dis.append(distance.detach().item())
        else:
            different_class_dis.append(distance.detach().item())
    
    return np.mean(same_class_dis) if len(same_class_dis) else 0, np.mean(different_class_dis) if len(different_class_dis) else 0

def test(model, testing_data, device="cuda:1"):
    test_loader = DataLoader(testing_data, batch_size=32, shuffle=True, drop_last=False)
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    model.eval()
    test_loss, correct = 0, 0
    confmat = ConfusionMatrix(num_classes=10).to(device)
    cmtx = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            cmtx += confmat(pred, y)

    test_loss /= num_batches
    correct /= size

    acc, cfmtx =  correct, cmtx.cpu().numpy()
    down = np.sum(cfmtx, axis=1, keepdims=True)
    down[down == 0] = 1
    cfmtx = cfmtx/down
    return cfmtx

def check_masking_loss(global_model, testing_data, loss_fn, device="cuda:1"):
    test_loader = DataLoader(testing_data, batch_size=32, shuffle=True, drop_last=False)
    global_model.to(device)
    
    label_list = [[1,2],[3,4],[5,6],[7,8],[9,0]]
    
    def masking(labels, dim=10):
        mask = torch.zeros([len(labels), dim])
        for label_idx in range(len(labels)):
            label = labels[label_idx]
            mask_units = None
            for client_label in label_list:
                if label in client_label:
                    mask_units = client_label
                    break
            
            for unit in mask_units:
                mask[label_idx, unit] = 1

        return mask
    
    total_loss = 0
    num_batch = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            mirr_mask_diagonal = global_model.mask_diagonal(X)
            mirr_mask_diagonal = (mirr_mask_diagonal > 1/10) * 1.0
            ground_mask = masking(y.tolist()).to(device)
            mirr_mask_diagonal = (mirr_mask_diagonal > 1/10) * 1.0
            total_loss += torch.sum(torch.abs(mirr_mask_diagonal - ground_mask))/mirr_mask_diagonal.shape[0]
            num_batch += 1
    
    return total_loss.cpu().numpy()/num_batch

def mask_transferring(global_model, local_representation_storage, epochs=8, device="cuda:1"):
    """
    local_representation_storage: N packs
        each pack is a list of tuple (representation, mask_diagonal)
    
    """
    global_model = global_model.to(device)
    optimizer = torch.optim.SGD(global_model.parameters(), lr=1e-3)
    
    reps = []
    mask = []
    for client_id in local_representation_storage.keys():
        for r, m in local_representation_storage[client_id]:
            reps.append(r)
            mask.append(m)
            # print("\trep shape:", r.shape)
            # print("\tmask shape:", m.shape)
    
    reps = torch.cat(reps, dim=0).to(device)
    mask = torch.cat(mask, dim=0).to(device)
    
    # print("Concat reps:", reps.shape)
    # print("Concat mask:", mask.shape)
    
    for epoch in range(epochs):
        mirr_mask_diagonal = global_model.mask_diagonal_regenerator(reps)
        loss = torch.sum(torch.pow(mirr_mask_diagonal - mask, 2))/mirr_mask_diagonal.shape[0]
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        mirr_mask_diagonal = (mirr_mask_diagonal > 1/10) * 1.0
        loss = torch.sum(torch.abs(mirr_mask_diagonal - mask))/mirr_mask_diagonal.shape[0]
        
        print(f"\tServer mask transferring epoch {epoch}, Avg. loss {loss.detach().item():>.3f}")
        
    return global_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--round", type=int, default=100)
    parser.add_argument("--warmup_round", type=int, default=20)
    args = parser.parse_args()
    batch_size = args.batch_size
    epochs = args.epochs
    
    training_data = datasets.MNIST(
        root="../data",
        train=True,
        download=False,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
    )
    
    testing_data = datasets.MNIST(
        root="../data",
        train=False,
        download=False,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
    )
    
    # client_id_list = [0]
    client_id_list = [0,1,2,3,4]
    clients_dataset = [CustomDataset(training_data, json.load(open(f"./jsons/client{client_id}.json", 'r'))) for client_id in client_id_list]
    total_sample = np.sum([len(dataset) for dataset in clients_dataset])
    
    global_model = DNN_proposal().to(device)
    local_classification_loss_record = {client_id:[] for client_id in client_id_list}
    local_mask_loss_record = {client_id:[] for client_id in client_id_list}
    local_cfmtx_bfag_record = {client_id:[] for client_id in client_id_list}
    local_cfmtx_afag_record = {client_id:[] for client_id in client_id_list}
    global_cfmtx_record = []
    global_masking_loss = []
    clients_mask_diagonal = [None for client_id in client_id_list]
    local_representation_storage = {client_id: None for client_id in client_id_list}
    mask_done = {client_id: False for client_id in client_id_list}
    warmup = True
    
    for cur_round in range(args.round + 1):
        print("============ Round {} ==============".format(cur_round))
        client_models = []
        impact_factors = [len(clients_dataset[client_id])/total_sample for client_id in client_id_list]
        
        if cur_round > args.warmup_round:
            warmup = False 
        
        # Local training
        for client_id in client_id_list:
            print("    Client {} training... ".format(client_id), end="")
            # Training process
            mydataset = clients_dataset[client_id]
            
            if clients_mask_diagonal[client_id] is None:
                clients_mask_diagonal[client_id] = create_mask_diagonal(dim=10, dataset=mydataset)
                
            local_model = copy.deepcopy(global_model)
            optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-3)
                       
            if warmup:
                # Train the representation space
                print("warming up...", end="")
                train_dataloader = DataLoader(mydataset, batch_size=2, shuffle=True, drop_last=True)
                loss_fn = torch.nn.MSELoss()
                for t in range(epochs):
                    same, diff = representation_training(train_dataloader, local_model, loss_fn, optimizer, device)
                print(f"Done!")
                
            else:
                train_dataloader = DataLoader(mydataset, batch_size=batch_size, shuffle=True, drop_last=False)
            
                # Train the mask first
                for t in range(epochs):
                    mask_loss, local_rep = mask_training(train_dataloader, local_model, optimizer, clients_mask_diagonal[client_id], "cuda")
                print(f"\n\tMasking done! Fin. round loss: {mask_loss:>.3f}")
                local_representation_storage[client_id] = local_rep
                    
                # Then train the classifier
                loss_fn = torch.nn.CrossEntropyLoss()
                epoch_loss = []
                for t in range(epochs):
                    loss = classification_training(train_dataloader, local_model, loss_fn, optimizer, clients_mask_diagonal[client_id], "cuda")
                    epoch_loss.append(loss)
                local_classification_loss_record[client_id].append(np.mean(epoch_loss))
                print(f"\tClassification done! Aver. round loss: {np.mean(epoch_loss):>.3f}")
                
                # Testing the local_model to its own data
                cfmtx = test(local_model, mydataset)
                local_cfmtx_bfag_record[client_id].append(cfmtx)
                
            client_models.append(augment_model(local_model, torch.diag_embed(clients_mask_diagonal[client_id]), impact_factors[client_id], device))
            
        print("    # Server aggregating... ", end="")
        
        # Aggregation
        global_model = fmodule._model_sum([model * pk for model, pk in zip(client_models, impact_factors)])
        print("Done!")
        if not warmup:
            mask_transferring(global_model, local_representation_storage, epochs=8, device=device)
            
        print("    # Server testing... ", end="")
        cfmtx = test(global_model, testing_data, device)
        global_cfmtx_record.append(cfmtx)
        print("Done!")
        np.set_printoptions(precision=2, suppress=True)
        if not warmup:
            print_cfmtx(cfmtx)

        masking_loss = check_masking_loss(global_model, testing_data, torch.nn.MSELoss(), device)
        global_masking_loss.append(masking_loss)
        print("Server masking loss", masking_loss)
    
    if not Path("records/proposal4").exists():
        os.makedirs("records/proposal4")
    
    json.dump(local_classification_loss_record,        open("records/proposal4/local_classification_loss_record.json", "w"),         cls=NumpyEncoder)
    json.dump(local_mask_loss_record,   open("records/proposal4/local_mask_loss_record.json", "w"),    cls=NumpyEncoder)
    json.dump(local_cfmtx_bfag_record,  open("records/proposal4/local_cfmtx_bfag_record.json", "w"),   cls=NumpyEncoder)
    json.dump(global_cfmtx_record,      open("records/proposal4/global_cfmtx_record.json", "w"),       cls=NumpyEncoder)
    json.dump(global_masking_loss,      open("records/proposal4/global_masking_loss.json", "w"),       cls=NumpyEncoder)
    