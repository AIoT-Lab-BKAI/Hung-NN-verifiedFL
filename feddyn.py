from utils.train_smt import test, print_cfmtx, NumpyEncoder, check_global_contrastive
from utils.reader import read_jsons
from utils.parser import read_arguments

from pathlib import Path
from torch.utils.data import DataLoader
from utils.FIM2 import MLP2
from utils.FIM3 import MLP3
from utils import fmodule
import torch, json, os, numpy as np, copy

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(dataloader, model, src_model, loss_fn, optimizer, gradL, alpha=0.5):      
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
    
    # gradL = gradL - alpha * (model - src_model)
    for p,q,k in zip(gradL.parameters(), model.parameters(), src_model.parameters()):
        p = p - alpha * (q - k)
        
    return losses

def aggregate(current_model, aver_model, h, rate, alpha=0.5):
    h = h - alpha * (rate * aver_model - current_model)
    new_model = aver_model - 1.0 / alpha * h
    return new_model

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
    elif args.dataset == "cifar10":
        global_model = MLP3().to(device)
    else:
        raise NotImplementedError
    
    local_loss_record = {client_id:[] for client_id in client_id_list}
    local_acc_bfag_record = {client_id:[] for client_id in client_id_list}
    local_acc_afag_record = {client_id:[] for client_id in client_id_list}
    
    global_cfmtx_record = []
    U_cfmtx_record = []
    
    server_h = global_model.zeros_like()
    clients_gradLs = [global_model.zeros_like() for client_id in client_id_list]
    
    client_per_round = len(client_id_list)
    
    for cur_round in range(args.round):
        print("============ Round {} ==============".format(cur_round))
        client_id_list_this_round = np.random.choice(client_id_list, size=client_per_round, replace=False).tolist()
        total_sample_this_round = np.sum([len(clients_training_dataset[i]) for i in client_id_list_this_round])
        impact_factors = [1.0/client_per_round for client_id in client_id_list_this_round]
        
        aver_model = global_model.zeros_like()
        src_model = copy.deepcopy(global_model)
        src_model.freeze_grad()
        
        # Local training
        for client_id in sorted(client_id_list_this_round):
            print("Client {} training... ".format(client_id), end="")
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
                epoch_loss.append(np.mean(train(train_dataloader, local_model, src_model, loss_fn, optimizer, gradL=clients_gradLs[client_id])))
            local_loss_record[client_id].append(np.mean(epoch_loss))
                        
            # Testing the local_model to its own data
            acc, cfmtx = test(local_model, my_testing_dataset)
            local_acc_bfag_record[client_id].append(acc)
            print(f"Done! Aver. round loss: {np.mean(epoch_loss):>.3f}, acc {acc:>.3f}")
            aver_model = fmodule._model_sum([aver_model, impact_factors[client_id] * local_model])
            
        print("# Server aggregating... ", end="")
        # Aggregation
        global_model = aggregate(global_model, aver_model, server_h, rate=client_per_round/len(client_id_list))
        print("Done!")
        
        print("# Server testing... ", end="")
        acc, cfmtx = test(global_model, global_testing_dataset)
        global_cfmtx_record.append(cfmtx)
        print(f"Done! Avg. acc {acc:>.3f}")

        
    if not Path(f"records/{args.exp_folder}/feddyn").exists():
        os.makedirs(f"records/{args.exp_folder}/feddyn")
    
    json.dump(local_loss_record,        open(f"records/{args.exp_folder}/feddyn/local_loss_record.json", "w"),         cls=NumpyEncoder)
    json.dump(local_acc_bfag_record,    open(f"records/{args.exp_folder}/feddyn/local_acc_bfag_record.json", "w"),     cls=NumpyEncoder)
    json.dump(local_acc_afag_record,    open(f"records/{args.exp_folder}/feddyn/local_acc_afag_record.json", "w"),     cls=NumpyEncoder)
    json.dump(global_cfmtx_record,      open(f"records/{args.exp_folder}/feddyn/global_cfmtx_record.json", "w"),       cls=NumpyEncoder)
    