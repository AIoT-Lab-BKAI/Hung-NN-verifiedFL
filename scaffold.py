from utils.train_smt import test, NumpyEncoder
from utils.reader import read_jsons
from utils.parser import read_arguments

from pathlib import Path
from torch.utils.data import DataLoader
from utils.FIM2 import MLPv2
from utils.FIM3 import MLPv3
from utils import fmodule
import torch, json, os, numpy as np, copy, random
import wandb


process_device = "cuda:0" if torch.cuda.is_available() else "cpu"
buffer_device_1 = "cuda:1" if torch.cuda.is_available() else "cpu"
buffer_device_2 = "cuda:2" if torch.cuda.is_available() else "cpu"
buffer_device_3 = "cuda:3" if torch.cuda.is_available() else "cpu"

def train(train_dataloader, local_model, loss_fn, optimizer, cg, c):
    losses = []
    num_batches = 0
    for batch_idx, (X, y) in enumerate(train_dataloader):
        local_model.zero_grad()
        X, y = X.to(process_device), y.to(process_device)
        
        pred = local_model(X)
        loss = loss_fn(pred, y)
        
        loss.backward()
        for pm, pcg, pc in zip(local_model.parameters(), cg.parameters(), c.parameters()):
            pm.grad = pm.grad - pc + pcg
        optimizer.step()
        num_batches += 1
    
    losses.append(loss.item())
    return losses, num_batches


def aggregate(global_model, cg, aver_dys, aver_dcs, eta=1.0, rate=1.0):   
    new_model = global_model + eta * aver_dys
    new_c = cg + 1.0 * rate * aver_dcs
    return new_model, new_c


if __name__ == "__main__":
    args = read_arguments(algorithm=os.path.basename(__file__).split('.py')[0])
    print(args)
    batch_size = args.batch_size
    epochs = args.epochs
        
    num_client, clients_training_dataset, clients_testing_dataset, global_testing_dataset, singleset = read_jsons(args.idx_folder, args.data_folder, args.dataset)
    client_id_list = [i for i in range(num_client)]
    
    if args.dataset == "mnist":
        global_model = MLPv2().to(process_device)
    elif args.dataset == "cifar10":
        global_model = MLPv3().to(process_device)
    else:
        raise NotImplementedError
    
    local_loss_record = {client_id:[] for client_id in client_id_list}
    local_acc_bfag_record = {client_id:[] for client_id in client_id_list}
    local_acc_afag_record = {client_id:[] for client_id in client_id_list}
    
    global_cfmtx_record = []
    U_cfmtx_record = []
    max_acc = 0
    
    client_cs = {}
    torch.manual_seed(0)
    for client_id in client_id_list:
        if client_id < len(client_id_list) * 5/13:
            print("Init client", client_id, "to ", buffer_device_1)
            client_cs[client_id] = global_model.zeros_like().to(buffer_device_1)
        elif client_id < 2 * len(client_id_list) * 5/13:
            print("Init client", client_id, "to ", buffer_device_2)
            client_cs[client_id] = global_model.zeros_like().to(buffer_device_2)
        else:
            print("Init client", client_id, "to ", buffer_device_3)
            client_cs[client_id] = global_model.zeros_like().to(buffer_device_3)
            
    # client_cs = {client_id: global_model.zeros_like().cpu() for client_id in client_id_list}
    cg = global_model.zeros_like()
    cg.freeze_grad()
    
    client_per_round = len(client_id_list)
    for cur_round in range(args.round):
        print("============ Round {} ==============".format(cur_round))
        
        aver_dys = global_model.zeros_like()
        aver_dcs = global_model.zeros_like()
        
        client_id_list_this_round = sorted(np.random.choice(client_id_list, size=client_per_round, replace=False).tolist())
        total_sample_this_round = np.sum([len(clients_training_dataset[i]) for i in client_id_list_this_round])
        impact_factors = {client_id: 1/client_per_round for client_id in client_id_list_this_round}
    
        inference_acc = []
        training_loss = []
        
        # Local training
        for client_id in client_id_list_this_round:
            if args.verbose:
                print("    Client {} training... ".format(client_id), end="")
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
            K = 0
            
            origin_device = client_cs[client_id].get_device()
            client_cs[client_id] = client_cs[client_id].to(process_device)
            for t in range(epochs):
                losses, num_batch = train(train_dataloader, local_model, loss_fn, optimizer, cg, client_cs[client_id])
                epoch_loss.append(np.mean(losses))
                K += num_batch
            
            with torch.no_grad():
                dy = local_model - global_model
                dc = -1.0 / (K * 1e-3) * dy - cg
                client_cs[client_id] = client_cs[client_id] + dc
                aver_dys = fmodule._model_sum([aver_dys, impact_factors[client_id] * dy])
                aver_dcs = fmodule._model_sum([aver_dcs, impact_factors[client_id] * dc])
    
            local_loss_record[client_id].append(np.mean(epoch_loss))
            training_loss.append(local_loss_record[client_id][-1])
            
            client_cs[client_id] = client_cs[client_id].to(origin_device)
            
            # Testing the local_model to its own data
            acc, cfmtx = test(local_model, my_testing_dataset, device=local_model.get_device())
            local_acc_bfag_record[client_id].append(acc)
            if args.verbose:
                print(f"Done! Aver. round loss: {np.mean(epoch_loss):>.3f}, acc {acc:>.3f}")
        
        print("# Server aggregating... ", end="")
        # Server aggregation
        global_model, cg = aggregate(global_model, cg, aver_dys, aver_dcs, rate=client_per_round/len(client_id_list))
        print("Done!")
        
        print("# Server testing... ", end="")
        acc, cfmtx = test(global_model, global_testing_dataset)
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
        
    if not Path(f"records/{args.idx_folder}/E{epochs}/R{args.round}/scaffold").exists():
        os.makedirs(f"records/{args.idx_folder}/E{epochs}/R{args.round}/scaffold")
    
    json.dump(local_loss_record,        open(f"records/{args.idx_folder}/E{epochs}/R{args.round}/scaffold/local_loss_record.json", "w"),         cls=NumpyEncoder)
    json.dump(local_acc_bfag_record,    open(f"records/{args.idx_folder}/E{epochs}/R{args.round}/scaffold/local_acc_bfag_record.json", "w"),     cls=NumpyEncoder)
    json.dump(local_acc_afag_record,    open(f"records/{args.idx_folder}/E{epochs}/R{args.round}/scaffold/local_acc_afag_record.json", "w"),     cls=NumpyEncoder)
    json.dump(global_cfmtx_record,      open(f"records/{args.idx_folder}/E{epochs}/R{args.round}/scaffold/global_cfmtx_record.json", "w"),       cls=NumpyEncoder)
    