from utils.train_smt import gen_representation, test, batch_similarity, print_cfmtx, NumpyEncoder
from utils.aggregate import aggregate, check_representations
from pathlib import Path
from torch.utils.data import DataLoader
from utils.dataloader import CustomDataset
from torchvision import datasets, transforms
from utils.model import NeuralNetwork
from copy import deepcopy

import torch, argparse, json, os, numpy as np, copy

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_mask(dim, dataset):
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    mask = torch.zeros([dim, dim])
    for X, y in train_dataloader:
        label = y.item()
        mask[label, label] = 1
    return mask


def train(dataloader, model, loss_fn, optimizer, fct=0.01):
    base_model = deepcopy(model).cuda()
    base_model.freeze_grad()
    
    model = model.cuda()
    model.train()
    losses = []
        
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred, rep = model.pred_and_rep(X)
        _, rep_base = base_model.pred_and_rep(X)
        
        sim = batch_similarity(rep, rep_base)
        loss = loss_fn(pred, y) + fct * torch.sum(sim - torch.diag(sim)) / (X.shape[0] **2)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        
    return losses

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
    
    for cur_round in range(args.round):
        print("============ Round {} ==============".format(cur_round))
        condense_representation_list = []
        client_models = []
        
        # Local training
        for client_id in client_id_list:
            print("    Client {} training... ".format(client_id), end="")
            # Training process
            mydataset = clients_dataset[client_id]
            train_dataloader = DataLoader(mydataset, batch_size=batch_size, shuffle=True, drop_last=False)
            
            U = create_mask(dim=10, dataset=mydataset)
                
            local_model = copy.deepcopy(global_model)
            local_model.update_mask(U)

            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(local_model.parameters(), lr=1e-3)

            epoch_loss = []
            for t in range(epochs):
                epoch_loss.append(np.mean(train(train_dataloader, local_model, loss_fn, optimizer, fct=0.1)))
            local_loss_record[client_id].append(np.mean(epoch_loss))
            
            client_models.append(local_model)
            
            # Testing the local_model to its own data
            cfmtx = test(local_model, mydataset)
            local_cfmtx_bfag_record[client_id].append(cfmtx)
            
            # Generating representations
            condense_representation = torch.randn([512], requires_grad=True).unsqueeze(0).cuda()
            for t in range(epochs):
                condense_representation = gen_representation(train_dataloader, local_model, condense_representation)
            condense_representation_list.append(condense_representation.detach().cpu())
            local_model.zero_grad()
            print(f"Done! Aver. round loss: {np.mean(epoch_loss):>.3f}")
        
        print("    # Server aggregating... ", end="")
        # Aggregation
        condensed_rep = torch.cat(condense_representation_list, dim=0)
        global_model = aggregate(client_models, condensed_rep, indexes=client_id_list)
        print("Done!")
        
        # Testing
        print("    # Server testing... ", end="")
        cfmtx = test(global_model, testing_data)
        global_cfmtx_record.append(cfmtx)
        print("Done!")
        np.set_printoptions(precision=2, suppress=True)
        print_cfmtx(cfmtx)
        
        U_cfmtx = check_representations(global_model, condensed_rep, testing_data, "cuda")
        U_cfmtx_record.append(U_cfmtx)
    
    if not Path("records/proposal").exists():
        os.makedirs("records/proposal")
    
    json.dump(local_loss_record,        open("records/proposal/local_loss_record.json", "w"),         cls=NumpyEncoder)
    json.dump(local_cfmtx_bfag_record,  open("records/proposal/local_cfmtx_bfag_record.json", "w"),   cls=NumpyEncoder)
    json.dump(global_cfmtx_record,      open("records/proposal/global_cfmtx_record.json", "w"),       cls=NumpyEncoder)
    json.dump(U_cfmtx_record,           open("records/proposal/U_cfmtx_record.json", "w"),            cls=NumpyEncoder)