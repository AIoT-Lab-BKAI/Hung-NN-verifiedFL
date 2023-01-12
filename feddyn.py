from utils.train_smt import test, print_cfmtx, NumpyEncoder, check_global_contrastive
from utils.reader import read_jsons
from utils.parser import read_arguments

from pathlib import Path
from torch.utils.data import DataLoader
from utils.FIM2 import MLP2
from utils.FIM3 import MLP3
from utils import fmodule
import torch, json, os, numpy as np, copy
import wandb

process_device = "cuda:0" if torch.cuda.is_available() else "cpu"
buffer_device_1 = "cuda:1" if torch.cuda.is_available() else "cpu"
buffer_device_2 = "cuda:2" if torch.cuda.is_available() else "cpu"
buffer_device_3 = "cuda:3" if torch.cuda.is_available() else "cpu"


def train(dataloader, model, src_model, loss_fn, optimizer, gradL, alpha=0.5):      
    model = model.to(process_device)
    model.train()
         
    losses = []
        
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(process_device), y.to(process_device)

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
    
    # gradL = gradL - alpha * (model - src_model)
    for p,q,k in zip(gradL.parameters(), model.parameters(), src_model.parameters()):
        p = p - alpha * (q - k)
        
    return losses

def aggregate(current_model, aver_model, h, rate, alpha=0.5):
    h = h - alpha * (rate * aver_model - current_model)
    new_model = aver_model - 1.0 / alpha * h
    return new_model

if __name__ == "__main__":
    args = read_arguments(algorithm=os.path.basename(__file__).split('.py')[0])
    print(args)
    batch_size = args.batch_size
    epochs = args.epochs
    
    num_client, clients_training_dataset, clients_testing_dataset, global_testing_dataset, singleset = read_jsons(args.idx_folder, args.data_folder, args.dataset)
    client_id_list = [i for i in range(num_client)]
    total_sample = np.sum([len(dataset) for dataset in clients_training_dataset])
    
    if args.dataset == "mnist":
        global_model = MLP2().to(process_device)
    elif args.dataset == "cifar10":
        global_model = MLP3().to(process_device)
    else:
        raise NotImplementedError
    
    local_loss_record = {client_id:[] for client_id in client_id_list}
    local_acc_bfag_record = {client_id:[] for client_id in client_id_list}
    local_acc_afag_record = {client_id:[] for client_id in client_id_list}
    
    global_cfmtx_record = []
    U_cfmtx_record = []
    max_acc = 0
    
    server_h = global_model.zeros_like()
    clients_gradLs = {client_id: global_model.zeros_like().cpu() for client_id in client_id_list}
    torch.manual_seed(0)
    for client_id in client_id_list:
        if client_id < len(client_id_list) * 5/13:
            print("Init client", client_id, "to ", buffer_device_1)
            clients_gradLs[client_id] = global_model.zeros_like().to(buffer_device_1)
        elif client_id < 2 * len(client_id_list) * 5/13:
            print("Init client", client_id, "to ", buffer_device_2)
            clients_gradLs[client_id] = global_model.zeros_like().to(buffer_device_2)
        else:
            print("Init client", client_id, "to ", buffer_device_3)
            clients_gradLs[client_id] = global_model.zeros_like().to(buffer_device_3)
    
    client_per_round = len(client_id_list)
    
    for cur_round in range(args.round):
        print("============ Round {} ==============".format(cur_round))
        client_id_list_this_round = sorted(np.random.choice(client_id_list, size=client_per_round, replace=False).tolist())
        impact_factors = {client_id: 1.0/client_per_round for client_id in client_id_list_this_round}
        
        aver_model = global_model.zeros_like()
        src_model = copy.deepcopy(global_model)
        src_model.freeze_grad()
        
        inference_acc = []
        training_loss = []
        
        # Local training
        for client_id in client_id_list_this_round:
            if args.verbose:
                print("Client {} training... ".format(client_id), end="")
            # Training process
            my_training_dataset = clients_training_dataset[client_id]
            my_testing_dataset = clients_testing_dataset[client_id]
            
            local_model = copy.deepcopy(global_model)
            # Testing the global_model to the local data
            acc, cfmtx = test(global_model, my_testing_dataset, device=local_model.get_device())
            local_acc_afag_record[client_id].append(acc)
            inference_acc.append(acc)
            
            train_dataloader = DataLoader(my_training_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(local_model.parameters(), lr=1e-3)
            
            epoch_loss = []
            origin_device = clients_gradLs[client_id].get_device()
            clients_gradLs[client_id] = clients_gradLs[client_id].to(process_device)
            for t in range(epochs):
                epoch_loss.append(np.mean(train(train_dataloader, local_model, src_model, loss_fn, optimizer, gradL=clients_gradLs[client_id])))
            
            clients_gradLs[client_id] = clients_gradLs[client_id].to(origin_device)
            local_loss_record[client_id].append(np.mean(epoch_loss))
            training_loss.append(local_loss_record[client_id][-1])
            
            # Testing the local_model to its own data
            acc, cfmtx = test(local_model, my_testing_dataset)
            local_acc_bfag_record[client_id].append(acc)
            if args.verbose:
                print(f"Done! Aver. round loss: {np.mean(epoch_loss):>.3f}, acc {acc:>.3f}")
            aver_model = fmodule._model_sum([aver_model, impact_factors[client_id] * local_model])
            
        print("# Server aggregating... ", end="")
        # Aggregation
        global_model = aggregate(global_model, aver_model, server_h, rate=client_per_round/len(client_id_list))
        print("Done!")
        
        print("# Server testing... ", end="")
        acc, cfmtx = test(global_model, global_testing_dataset, device=local_model.get_device())
        global_cfmtx_record.append(cfmtx)
        print(f"Done! Avg. acc {acc:>.3f}")

        max_acc = max(max_acc, acc)
        if args.wandb:
            wandb.log({
                    "Mean inference accuracy": np.mean(inference_acc),
                    "Mean training loss": np.mean(training_loss),
                    "Global accuracy": acc,
                    "Max accuracy": max_acc
                })
        
    if not Path(f"records/{args.idx_folder}/E{epochs}/feddyn").exists():
        os.makedirs(f"records/{args.idx_folder}/E{epochs}/feddyn")
    
    json.dump(local_loss_record,        open(f"records/{args.idx_folder}/E{epochs}/feddyn/local_loss_record.json", "w"),         cls=NumpyEncoder)
    json.dump(local_acc_bfag_record,    open(f"records/{args.idx_folder}/E{epochs}/feddyn/local_acc_bfag_record.json", "w"),     cls=NumpyEncoder)
    json.dump(local_acc_afag_record,    open(f"records/{args.idx_folder}/E{epochs}/feddyn/local_acc_afag_record.json", "w"),     cls=NumpyEncoder)
    json.dump(global_cfmtx_record,      open(f"records/{args.idx_folder}/E{epochs}/feddyn/global_cfmtx_record.json", "w"),       cls=NumpyEncoder)
    