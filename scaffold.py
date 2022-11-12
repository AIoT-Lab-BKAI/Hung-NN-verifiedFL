from utils.train_smt import test, print_cfmtx, NumpyEncoder
from utils.reader import read_jsons
from utils.parser import read_arguments

from pathlib import Path
from torch.utils.data import DataLoader
from utils.dataloader import CustomDataset
from torchvision import datasets, transforms
from utils.base_model import NeuralNetwork
from utils import fmodule
import torch, argparse, json, os, numpy as np, copy

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    args = read_arguments()
    print(args)
    batch_size = args.batch_size
    epochs = args.epochs
    
    num_client, clients_training_dataset, clients_testing_dataset, global_testing_dataset = read_jsons(args.exp_folder, args.dataset)
    client_id_list = [i for i in range(num_client)]
    
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
            my_training_dataset = clients_training_dataset[client_id]
            my_testing_dataset = clients_testing_dataset[client_id]
            
            train_dataloader = DataLoader(my_training_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
            
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
            cfmtx = test(local_model, my_testing_dataset)
            local_cfmtx_bfag_record[client_id].append(cfmtx)
            print(f"Done! Aver. round loss: {np.mean(epoch_loss):>.3f}, acc {np.mean(np.diag(cfmtx)):>.3f}")
        
        print("    # Server aggregating... ", end="")
        # Server aggregation
        global_model, cg = aggregate(global_model, cg, client_dys, client_dcs, total_client=len(client_id_list))
        print("Done!")
        
        print("    # Server testing... ", end="")
        cfmtx = test(global_model, global_testing_dataset)
        global_cfmtx_record.append(cfmtx)
        print(f"Done! Avg. acc {np.mean(np.diag(cfmtx)):>.3f}")
        # np.set_printoptions(precision=2, suppress=True)
        # print_cfmtx(cfmtx)
        
        if not Path(f"records/{args.exp_folder}/scaffold").exists():
            os.makedirs(f"records/{args.exp_folder}/scaffold")
        
        json.dump(local_loss_record,        open(f"records/{args.exp_folder}/scaffold/local_loss_record.json", "w"),         cls=NumpyEncoder)
        json.dump(local_cfmtx_bfag_record,  open(f"records/{args.exp_folder}/scaffold/local_cfmtx_bfag_record.json", "w"),   cls=NumpyEncoder)
        json.dump(global_cfmtx_record,      open(f"records/{args.exp_folder}/scaffold/global_cfmtx_record.json", "w"),       cls=NumpyEncoder)