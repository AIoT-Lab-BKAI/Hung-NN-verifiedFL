import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.dataloader import CustomDataset
from utils.elements import ProposedNet, MaskGenerator, augmented_classifier
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

def classifier_training(dataloader, model:ProposedNet, original_mask_diagonal, device="cuda:1"):
    original_mask_diagonal = original_mask_diagonal.to(device)
    model = model.to(device)
    model.train()
    losses = []

    optimizer = torch.optim.Adam(model.classifier.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X, original_mask_diagonal)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    # print("classification losses", losses)
    return np.mean(losses)

def mask_training(dataloader, model:ProposedNet, original_mask_diagonal, device="cuda:1"):
    """
    This method trains to make the model generate a mask that is
    close to the original mask
    """
    original_mask_diagonal = original_mask_diagonal.to(device)
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.mask_generator.parameters(), lr=1e-3)
    
    storage = []
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        r_x = model.feature_extractor(X).detach()
        dm_x = model.mask_generator(r_x)
        
        storage.append((r_x.detach().cpu(), dm_x.detach().cpu()))
        
        loss = torch.sum(torch.pow(dm_x - original_mask_diagonal, 2))/dm_x.shape[0]
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        dm_x = (dm_x > 1/10) * 1.0
        loss = torch.sum(torch.abs(dm_x - original_mask_diagonal))/dm_x.shape[0]
        
    return loss, storage

def representation_training(dataset, model:ProposedNet, device="cuda:1"):
    """
    This method trains for a discriminative representation space,
    using constrastive learning
    """
    
    feature_extractor = model.feature_extractor.to(device)
    feature_extractor.train()
    same_class_dis = []
    different_class_dis = []
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(feature_extractor.parameters(), lr=1e-3)
    loss_fn=torch.nn.MSELoss()
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        representations = feature_extractor(X)
        
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

def check_masking_loss(global_model:ProposedNet, testing_data, device="cuda:1"):
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
            r_x = global_model.feature_extractor(X)
            m_x = global_model.mask_generator(r_x)
            m_x = (m_x > 1/10) * 1.0
            ground_mask = masking(y.tolist()).to(device)
            total_loss += torch.sum(torch.abs(m_x - ground_mask))/m_x.shape[0]
            num_batch += 1
    
    return total_loss.cpu().numpy()/num_batch


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
    
    local_mask_loss_record = {client_id:[] for client_id in client_id_list}
    local_cfmtx_bfag_record = {client_id:[] for client_id in client_id_list}
    global_cfmtx_record = []
    global_masking_loss = []
    clients_mask_diagonal = [None for client_id in client_id_list]
    local_representation_storage = {client_id: None for client_id in client_id_list}
    warmup = True
    
    mg_done = [False for client_id in client_id_list]
    local_mgs = [MaskGenerator() for client_id in client_id_list]    
    global_model = ProposedNet().to(device)
    
    for cur_round in range(args.round + 1):
        print("============ Round {} ==============".format(cur_round))
        # client_id_round = np.random.choice(client_id_list, len(client_id_list), replace=False).tolist()
        client_id_round = client_id_list.copy()
        print("Client this round: ", client_id_round)
        
        client_models = []
        impact_factors = [len(clients_dataset[client_id])/total_sample for client_id in client_id_round]
        
        if cur_round > args.warmup_round:
            warmup = False 
        
        # Local training
        for client_id in client_id_round:
            print("\tClient {} training: ".format(client_id), end="")

            # Prepare dataset and mask
            mydataset = clients_dataset[client_id]
            if clients_mask_diagonal[client_id] is None:
                clients_mask_diagonal[client_id] = create_mask_diagonal(dim=10, dataset=mydataset)
                
            # Make a copy of the global model
            local_model = copy.deepcopy(global_model)
            local_model.mask_generator = local_mgs[client_id]

            if warmup:
                # Train the representation space
                print("Warming up...", end="")
                for t in range(epochs):
                    same, diff = representation_training(mydataset, local_model, device)
                print(f"Done!")
                
            else:
                train_dataloader = DataLoader(mydataset, batch_size=batch_size, shuffle=True, drop_last=False)

                # Train the mask first
                if not mg_done[client_id]:
                    print("\n\t  Mask training...", end="")
                    for t in range(epochs):
                        last_loss, storage = mask_training(train_dataloader, local_model, clients_mask_diagonal[client_id], device)
                    local_representation_storage[client_id] = storage
                    
                    if last_loss < 0.01:
                        mg_done[client_id] = True
                    print(f"Done, fin. loss = {last_loss:>.3f}", end="")
                        
                # Then train the classifier
                print("\n\t  Classifier training...", end="")
                for t in range(epochs):
                    mean_loss = classifier_training(train_dataloader, local_model, clients_mask_diagonal[client_id], device)
                print(f"Done, avg. loss = {mean_loss:>.3f}")
                
                # Testing the local_model to its own data
                cfmtx = test(local_model, mydataset)
                local_cfmtx_bfag_record[client_id].append(cfmtx)
            
            # Augment the classifier
            local_model.classifier = augmented_classifier(local_model.classifier, torch.diag_embed(clients_mask_diagonal[client_id]), impact_factors[client_id], device)
            client_models.append(local_model)
            
        # Aggregation
        print("\t# Server aggregating... ")
        if warmup:
            """
            If in phase 01, aggregate the feature extractor only
            """
            global_model.feature_extractor = ProposedNet.aggregate_fe([model.feature_extractor for model in client_models], impact_factors, device)
            
        else:
            """
            If in phase 02, aggregate the mask generator and classifier
            """
            global_model.classifier = ProposedNet.aggregate_cl([model.classifier for model in client_models], impact_factors, device)
            global_model.mask_generator = ProposedNet.aggregate_mg(global_model.mask_generator, [model.mask_generator for model in client_models],
                                                                   local_representation_storage, epochs=epochs, device=device)
        # print("Done!")
        
        # Testing
        print("\t# Server testing... ", end="")
        cfmtx = test(global_model, testing_data, device)
        global_cfmtx_record.append(cfmtx)
        print("Done!")
        np.set_printoptions(precision=2, suppress=True)
        if not warmup:
            print_cfmtx(cfmtx)

        masking_loss = check_masking_loss(global_model, testing_data, device)
        global_masking_loss.append(masking_loss)
        print("Server masking loss", masking_loss)
    
    if not Path("records/proposal4").exists():
        os.makedirs("records/proposal4")
    
    json.dump(local_mask_loss_record,   open("records/proposal4/local_mask_loss_record.json", "w"),    cls=NumpyEncoder)
    json.dump(local_cfmtx_bfag_record,  open("records/proposal4/local_cfmtx_bfag_record.json", "w"),   cls=NumpyEncoder)
    json.dump(global_cfmtx_record,      open("records/proposal4/global_cfmtx_record.json", "w"),       cls=NumpyEncoder)
    json.dump(global_masking_loss,      open("records/proposal4/global_masking_loss.json", "w"),       cls=NumpyEncoder)
    