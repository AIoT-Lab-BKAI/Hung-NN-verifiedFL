from utils.train_smt import test, print_cfmtx, NumpyEncoder
from pathlib import Path
from torch.utils.data import DataLoader
from utils.dataloader import CustomDataset
from torchvision import datasets, transforms
from utils.base_model import NeuralNetwork
from utils import fmodule
import torch, argparse, json, os, numpy as np, copy

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(dataloader, model, loss_fn, optimizer, gradL, alpha=0.5):  
    src_model = copy.deepcopy(model).to(device)
    src_model.freeze_grad()
    model = model.to(device)
    model.train()
         
    losses = []
        
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        l1 = loss_fn(pred, y)
        l2 = 0
        l3 = 0
        for pgl, pm, ps in zip(gradL.parameters(), model.parameters(), src_model.parameters()):
            l2 += torch.dot(pgl.view(-1), pm.view(-1))
            l3 += torch.sum(torch.pow(pm-ps,2))
        loss = l1 - l2 + 0.5 * alpha * l3
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    
    gradL = gradL - alpha * (model - src_model)
    return losses

def aggregate(current_model, models, h, alpha=0.5, total_clients=5):
    h = h - alpha * (1.0 / total_clients * fmodule._model_sum(models) - current_model)
    new_model = fmodule._model_average(models) - 1.0 / alpha * h
    return new_model

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
        download=False,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
    )
    
    testing_data = datasets.MNIST(
        root="data",
        train=False,
        download=False,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
    )
    
    client_id_list = [0,1,2,3,4]
    clients_dataset = [CustomDataset(training_data, json.load(open(f"./jsons/client{client_id}.json", 'r'))) for client_id in client_id_list]
    impact_factors = [1/len(dataset) for dataset in clients_dataset]
    
    global_model = NeuralNetwork().to(device)
    local_loss_record = {client_id:[] for client_id in client_id_list}
    local_cfmtx_bfag_record = {client_id:[] for client_id in client_id_list}
    local_cfmtx_afag_record = {client_id:[] for client_id in client_id_list}
    global_cfmtx_record = []
    U_cfmtx_record = []
    
    
    server_h = global_model.zeros_like()
    clients_gradLs = [global_model.zeros_like() for client_id in client_id_list]
    
    for cur_round in range(args.round):
        print("============ Round {} ==============".format(cur_round))
        client_models = []
        
        # Local training
        for client_id in client_id_list:
            print("    Client {} training... ".format(client_id), end="")
            # Training process
            mydataset = clients_dataset[client_id]
            train_dataloader = DataLoader(mydataset, batch_size=batch_size, shuffle=True, drop_last=False)
            
            local_model = copy.deepcopy(global_model)
            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(local_model.parameters(), lr=1e-3)
            
            epoch_loss = []
            for t in range(epochs):
                epoch_loss.append(np.mean(train(train_dataloader, local_model, loss_fn, optimizer, gradL=clients_gradLs[client_id])))
            local_loss_record[client_id].append(np.mean(epoch_loss))
            
            client_models.append(local_model)
            
            # Testing the local_model to its own data
            cfmtx = test(local_model, mydataset)
            local_cfmtx_bfag_record[client_id].append(cfmtx)
            print(f"Done! Aver. round loss: {np.mean(epoch_loss):>.3f}")
            
        print("    # Server aggregating... ", end="")
        # Aggregation
        global_model = aggregate(global_model, client_models, server_h, total_clients=len(client_id_list))
        print("Done!")
        
        print("    # Server testing... ", end="")
        cfmtx = test(global_model, testing_data)
        global_cfmtx_record.append(cfmtx)
        print("Done!")
        np.set_printoptions(precision=2, suppress=True)
        print_cfmtx(cfmtx)
        
        if not Path("records/feddyn").exists():
            os.makedirs("records/feddyn")
        
        json.dump(local_loss_record,        open("records/feddyn/local_loss_record.json", "w"),         cls=NumpyEncoder)
        json.dump(local_cfmtx_bfag_record,  open("records/feddyn/local_cfmtx_bfag_record.json", "w"),   cls=NumpyEncoder)
        json.dump(global_cfmtx_record,      open("records/feddyn/global_cfmtx_record.json", "w"),       cls=NumpyEncoder)