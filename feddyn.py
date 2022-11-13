from utils.train_smt import test, print_cfmtx, NumpyEncoder, check_global_contrastive
from utils.reader import read_jsons
from utils.parser import read_arguments

from pathlib import Path
from torch.utils.data import DataLoader
from utils.base_model import NeuralNetwork
from utils import fmodule
import torch, json, os, numpy as np, copy

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
    args = read_arguments()
    print(args)
    batch_size = args.batch_size
    epochs = args.epochs
    
    num_client, clients_training_dataset, clients_testing_dataset, global_testing_dataset, singleset = read_jsons(args.exp_folder, args.dataset)
    client_id_list = [i for i in range(num_client)]
    total_sample = np.sum([len(dataset) for dataset in clients_training_dataset])
    
    global_model = NeuralNetwork().to(device)
    local_loss_record = {client_id:[] for client_id in client_id_list}
    local_acc_bfag_record = {client_id:[] for client_id in client_id_list}
    local_acc_afag_record = {client_id:[] for client_id in client_id_list}
    global_constrastive_info = {"same": [], "diff": [], "sim_mtx": []}
    
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
            my_training_dataset = clients_training_dataset[client_id]
            my_testing_dataset = clients_testing_dataset[client_id]
            
            local_model = copy.deepcopy(global_model)
            # Testing the global_model to the local data
            acc, cfmtx = test(global_model, my_testing_dataset)
            local_acc_afag_record[client_id].append(acc)
            
            train_dataloader = DataLoader(my_training_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(local_model.parameters(), lr=1e-3)
            
            epoch_loss = []
            for t in range(epochs):
                epoch_loss.append(np.mean(train(train_dataloader, local_model, loss_fn, optimizer, gradL=clients_gradLs[client_id])))
            local_loss_record[client_id].append(np.mean(epoch_loss))
            
            client_models.append(local_model)
            
            # Testing the local_model to its own data
            acc, cfmtx = test(local_model, my_testing_dataset)
            local_acc_bfag_record[client_id].append(acc)
            print(f"Done! Aver. round loss: {np.mean(epoch_loss):>.3f}, acc {acc:>.3f}")
            
        print("    # Server aggregating... ", end="")
        # Aggregation
        global_model = aggregate(global_model, client_models, server_h, total_clients=len(client_id_list))
        print("Done!")
        
        print("    # Server testing... ", end="")
        acc, cfmtx = test(global_model, global_testing_dataset)
        global_cfmtx_record.append(cfmtx)
        
        same, diff, sim_mtx = check_global_contrastive(global_model, global_testing_dataset, device)
        global_constrastive_info["same"].append(same)
        global_constrastive_info["diff"].append(diff)
        global_constrastive_info["sim_mtx"].append(sim_mtx)
        print(f"Done! Avg. acc {acc:>.3f}, same {same:>.3f}, diff {diff:>.3f}")

        
    if not Path(f"records/{args.exp_folder}/feddyn").exists():
        os.makedirs(f"records/{args.exp_folder}/feddyn")
    
    json.dump(local_loss_record,        open(f"records/{args.exp_folder}/feddyn/local_loss_record.json", "w"),         cls=NumpyEncoder)
    json.dump(local_acc_bfag_record,    open(f"records/{args.exp_folder}/feddyn/local_acc_bfag_record.json", "w"),     cls=NumpyEncoder)
    json.dump(local_acc_afag_record,    open(f"records/{args.exp_folder}/feddyn/local_acc_afag_record.json", "w"),     cls=NumpyEncoder)
    json.dump(global_cfmtx_record,      open(f"records/{args.exp_folder}/feddyn/global_cfmtx_record.json", "w"),       cls=NumpyEncoder)
    json.dump(global_constrastive_info, open(f"records/{args.exp_folder}/feddyn/global_constrastive_info.json", "w"),  cls=NumpyEncoder)
    