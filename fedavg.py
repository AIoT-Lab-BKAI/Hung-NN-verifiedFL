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
    
    client_per_round = len(client_id_list)
    for cur_round in range(args.round):
        print("============ Round {} ==============".format(cur_round))
        client_id_list_this_round = sorted(np.random.choice(client_id_list, size=client_per_round, replace=False).tolist())
        total_sample_this_round = np.sum([len(clients_training_dataset[i]) for i in client_id_list_this_round])
        impact_factors = {client_id: len(clients_training_dataset[client_id])/total_sample_this_round for client_id in client_id_list_this_round}
        
        aver_model = global_model.zeros_like()
        # Local training
        inference_acc = []
        training_loss = []
        
        for client_id in client_id_list_this_round:
            if args.verbose:
                print("Client {} training... ".format(client_id), end="")
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
            optimizer = torch.optim.SGD(local_model.parameters(), lr=1e-3)
            
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
            
            aver_model = fmodule._model_sum([aver_model, impact_factors[client_id] * local_model])
            
        print("# Server aggregating... ", end="")
        # Aggregation
        global_model = copy.deepcopy(aver_model)
        print("Done!")
        
        print("# Server testing... ", end="")
        acc, cfmtx = test(global_model, global_testing_dataset)
        global_cfmtx_record.append(cfmtx)
    
        print(f"Done! Avg. acc {acc:>.3f}")
        
        if args.wandb:
            wandb.log({
                    "Mean inference accuracy": np.mean(inference_acc),
                    "Mean training loss": np.mean(training_loss),
                    "Global accuracy": acc,
                })
        
    if not Path(f"records/{args.idx_folder}/E{epochs}/fedavg").exists():
        os.makedirs(f"records/{args.idx_folder}/E{epochs}/fedavg")
    
    json.dump(local_loss_record,        open(f"records/{args.idx_folder}/E{epochs}/fedavg/local_loss_record.json", "w"),         cls=NumpyEncoder)
    json.dump(local_acc_bfag_record,    open(f"records/{args.idx_folder}/E{epochs}/fedavg/local_acc_bfag_record.json", "w"),     cls=NumpyEncoder)
    json.dump(local_acc_afag_record,    open(f"records/{args.idx_folder}/E{epochs}/fedavg/local_acc_afag_record.json", "w"),     cls=NumpyEncoder)
    json.dump(global_cfmtx_record,      open(f"records/{args.idx_folder}/E{epochs}/fedavg/global_cfmtx_record.json", "w"),       cls=NumpyEncoder)
    