from utils.train_smt import test
from pathlib import Path
from torch.utils.data import DataLoader
from utils.dataloader import CustomDataset
from torchvision import datasets, transforms
from utils.base_model import NeuralNetwork
from utils import fmodule
import torch, argparse, json, os, numpy as np, copy

device = "cuda" if torch.cuda.is_available() else "cpu"

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def print_cfmtx(mtx):
    for i in range(mtx.shape[0]):
        for j in range(mtx.shape[1]):
            if i == j:
                print(f"\033[48;5;225m{mtx[i,j]:>.3f}\033[0;0m", end="  ")
            else:
                print(f"{mtx[i,j]:>.3f}", end="  ")
        print()
    return

def train(train_dataloader, local_model, loss_fn, optimizer, cg, c):
    losses = []
    num_batches = 0
    for batch_idx, (X, y) in enumerate(train_dataloader):
        local_model.zero_grad()
        X, y = X.to(device), y.to(device)
        
        pred = local_model(X)
        loss = loss_fn(pred, y)
        
        loss.backward()
        for pm, pcg, pc in zip(local_model.parameters(), cg.parameters(), c.parameters()):
            pm.grad = pm.grad - pc + pcg
        optimizer.step()
        num_batches += 1
    
    losses.append(loss.item())
    return losses, num_batches


def aggregate(global_model, cg, dys, dcs, eta=1.0, total_client=5):  # c_list is c_i^+
    dw = fmodule._model_average(dys)
    dc = fmodule._model_average(dcs)
    new_model = global_model + eta * dw
    new_c = cg + 1.0 * len(dcs) / total_client * dc
    return new_model, new_c


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
    
    global_model = NeuralNetwork().to(device)
    local_loss_record = {client_id:[] for client_id in client_id_list}
    local_cfmtx_bfag_record = {client_id:[] for client_id in client_id_list}
    local_cfmtx_afag_record = {client_id:[] for client_id in client_id_list}
    global_cfmtx_record = []
    U_cfmtx_record = []
    
    client_cs = [global_model.zeros_like() for i in client_id_list]
    cg = global_model.zeros_like()
    cg.freeze_grad()
    
    for cur_round in range(args.round):
        print("============ Round {} ==============".format(cur_round))
        client_models = []
        # global_model.freeze_grad()
        
        client_dys = []
        client_dcs = []
    
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
            K = 0
            for t in range(epochs):
                losses, num_batch = train(train_dataloader, local_model, loss_fn, optimizer, cg, client_cs[client_id])
                epoch_loss.append(np.mean(losses))
                K += num_batch
            
            with torch.no_grad():
                dy = local_model - global_model
                dc = -1.0 / (K * 1e-3) * dy - cg
                client_cs[client_id] = client_cs[client_id] + dc
                client_dys.append(dy)
                client_dcs.append(dc)
    
            local_loss_record[client_id].append(np.mean(epoch_loss))
            client_models.append(local_model)
            
            # Testing the local_model to its own data
            cfmtx = test(local_model, mydataset)
            local_cfmtx_bfag_record[client_id].append(cfmtx)
            print(f"Done! Aver. round loss: {np.mean(epoch_loss):>.3f}")
        
        print("    # Server aggregating... ", end="")
        # Server aggregation
        global_model, cg = aggregate(global_model, cg, client_dys, client_dcs, total_client=len(client_id_list))
        print("Done!")
        
        print("    # Server testing... ", end="")
        cfmtx = test(global_model, testing_data)
        global_cfmtx_record.append(cfmtx)
        print("Done!")
        np.set_printoptions(precision=2, suppress=True)
        print_cfmtx(cfmtx)
        
        if not Path("records/scaffold").exists():
            os.makedirs("records/scaffold")
        
        json.dump(local_loss_record,        open("records/scaffold/local_loss_record.json", "w"),         cls=NumpyEncoder)
        json.dump(local_cfmtx_bfag_record,  open("records/scaffold/local_cfmtx_bfag_record.json", "w"),   cls=NumpyEncoder)
        json.dump(global_cfmtx_record,      open("records/scaffold/global_cfmtx_record.json", "w"),       cls=NumpyEncoder)