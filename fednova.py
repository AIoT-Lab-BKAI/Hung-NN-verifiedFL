from utils.train_smt import test, NumpyEncoder
from utils.reader import read_jsons
from utils.parser import read_arguments

from pathlib import Path
from torch.utils.data import DataLoader
from utils.FIM2 import MLPv2
from utils.FIM3 import MLPv3
from utils import fmodule
import torch, json, os, numpy as np, copy, random
import torch.nn.functional as F
import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(dataloader, model, loss_fn, optimizer):   
    model = model.cuda()
    model.train()
    losses = []
        
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        
    return losses


if __name__ == "__main__":
    args = read_arguments(algorithm=os.path.basename(__file__).split('.py')[0])
    print(args)
    batch_size = args.batch_size
    epochs = args.epochs
    
    num_client, clients_training_dataset, clients_testing_dataset, global_testing_dataset, singleset = read_jsons(args.idx_folder, args.data_folder, args.dataset)
    client_id_list = [i for i in range(num_client)]
    total_sample = np.sum([len(dataset) for dataset in clients_training_dataset])
    
    if args.dataset == "mnist":
        global_model = MLPv2().to(device)
    elif args.dataset == "cifar10":
        global_model = MLPv3().to(device)
    else:
        raise NotImplementedError
        
    local_loss_record = {client_id:[] for client_id in client_id_list}
    local_acc_bfag_record = {client_id:[] for client_id in client_id_list}
    local_acc_afag_record = {client_id:[] for client_id in client_id_list}
    
    global_cfmtx_record = []
    U_cfmtx_record = []
    max_acc = 0
    
    client_taus = {client_id: epochs * np.ceil(len(clients_training_dataset[client_id])/batch_size) for client_id in client_id_list}

    client_per_round = len(client_id_list)
    for cur_round in range(args.round):
        print("============ Round {} ==============".format(cur_round))
        client_id_list_this_round = sorted(np.random.choice(client_id_list, size=client_per_round, replace=False).tolist())
        total_sample_this_round = np.sum([len(clients_training_dataset[i]) for i in client_id_list_this_round])
        
        impact_factors = {client_id: len(clients_training_dataset[client_id])/total_sample_this_round for client_id in client_id_list_this_round}
        delta = global_model.zeros_like()
        tau_eff = 0
        
        inference_acc = []
        training_loss = []
        
        # Local training
        for client_id in sorted(client_id_list_this_round):
            if args.verbose:
                print("    Client {} training... ".format(client_id), end="")
            # Training process
            my_training_dataset = clients_training_dataset[client_id]
            my_testing_dataset = clients_testing_dataset[client_id]
            
            local_model = copy.deepcopy(global_model)
            # Testing the global_model to the local data
            acc, cfmtx = test(local_model, my_testing_dataset)
            local_acc_afag_record[client_id].append(acc)
            inference_acc.append(acc)
            
            train_dataloader = DataLoader(my_training_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(local_model.parameters(), lr=args.learning_rate)
            
            epoch_loss = []
            for t in range(epochs):
                epoch_loss.append(np.mean(train(train_dataloader, local_model, loss_fn, optimizer)))
            local_loss_record[client_id].append(np.mean(epoch_loss))
            training_loss.append(local_loss_record[client_id][-1])
                        
            # Testing the local_model to its own data
            acc, cfmtx = test(local_model, my_testing_dataset)
            local_acc_bfag_record[client_id].append(acc)
            if args.verbose:
                print(f"Done! Aver. round loss: {np.mean(epoch_loss):>.3f}, acc {acc:>.3f}")
            
            delta = fmodule._model_sum([delta, impact_factors[client_id]/client_taus[client_id] * (local_model - global_model)])
            tau_eff += impact_factors[client_id] * client_taus[client_id]
            
        print("    # Server aggregating... ", end="")
        # Aggregation
        global_model = global_model + tau_eff * delta
        print("Done!")
        
        print("    # Server testing... ", end="")
        acc, cfmtx = test(global_model, global_testing_dataset)
        global_cfmtx_record.append(cfmtx)
    
        print(f"Done! Avg. acc {acc:>.3f}")
        # print_cfmtx(cfmtx)
        
        max_acc = max(max_acc, acc)
        if args.wandb:
            wandb.log({
                    "Mean inference accuracy": np.mean(inference_acc),
                    "Mean training loss": np.mean(training_loss),
                    "Global accuracy": acc,
                    "Max accuracy": max_acc
                })
        
    if not Path(f"{args.log_folder}/{args.idx_folder}/E{epochs}/R{args.round}/fednova").exists():
        os.makedirs(f"{args.log_folder}/{args.idx_folder}/E{epochs}/R{args.round}/fednova")
    
    json.dump(local_loss_record,        open(f"{args.log_folder}/{args.idx_folder}/E{epochs}/R{args.round}/fednova/local_loss_record.json", "w"),         cls=NumpyEncoder)
    json.dump(local_acc_bfag_record,    open(f"{args.log_folder}/{args.idx_folder}/E{epochs}/R{args.round}/fednova/local_acc_bfag_record.json", "w"),     cls=NumpyEncoder)
    json.dump(local_acc_afag_record,    open(f"{args.log_folder}/{args.idx_folder}/E{epochs}/R{args.round}/fednova/local_acc_afag_record.json", "w"),     cls=NumpyEncoder)
    json.dump(global_cfmtx_record,      open(f"{args.log_folder}/{args.idx_folder}/E{epochs}/R{args.round}/fednova/global_cfmtx_record.json", "w"),       cls=NumpyEncoder)
    if args.save_model:
        print("Saving model ...", end="")
        torch.save(global_model.state_dict(), f"{args.log_folder}/{args.idx_folder}/E{epochs}/R{args.round}/fednova/global_model.pth")
        print("Done!")