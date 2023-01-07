from utils.train_smt import test, NumpyEncoder
from utils.reader import read_jsons
from utils.parser import read_arguments

from pathlib import Path
from torch.utils.data import DataLoader
from utils.FIM2 import MLP2
from utils import fmodule
import torch, json, os, numpy as np, copy, random
import torch.nn.functional as F

def set_seed(seed):
    random.seed(1+seed)
    np.random.seed(21+seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(12+seed)
    torch.cuda.manual_seed_all(123+seed)

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
        # KL
        # pred = F.softmax(pred, dim=1) 
        # ground = F.softmax(F.one_hot(y, 10) * 1.0, dim=1) 
        # loss = torch.sum(ground * torch.log(ground/pred)) / (ground.shape[0] * ground.shape[1])
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        
    return losses

if __name__ == "__main__":
    args = read_arguments()
    print(args)
    batch_size = args.batch_size
    epochs = args.epochs
    
    set_seed(args.seed)
    
    num_client, clients_training_dataset, clients_testing_dataset, global_testing_dataset, singleset = read_jsons(args.exp_folder, args.dataset)
    client_id_list = [i for i in range(num_client)]
    total_sample = np.sum([len(dataset) for dataset in clients_training_dataset])
    
    global_model = MLP2().to(device)
    local_loss_record = {client_id:[] for client_id in client_id_list}
    local_acc_bfag_record = {client_id:[] for client_id in client_id_list}
    local_acc_afag_record = {client_id:[] for client_id in client_id_list}
    global_constrastive_info = {"same": [], "diff": [], "sim_mtx": []}
    
    global_cfmtx_record = []
    U_cfmtx_record = []
    
    for cur_round in range(args.round):
        print("============ Round {} ==============".format(cur_round))
        client_models = []
        impact_factors = [len(clients_training_dataset[client_id])/total_sample for client_id in client_id_list]
        
        # Local training
        for client_id in client_id_list:
            print("    Client {} training... ".format(client_id), end="")
            # Training process
            my_training_dataset = clients_training_dataset[client_id]
            my_testing_dataset = clients_testing_dataset[client_id]
            
            local_model = copy.deepcopy(global_model)
            # Testing the global_model to the local data
            acc, cfmtx = test(local_model, my_testing_dataset)
            local_acc_afag_record[client_id].append(acc)
            
            train_dataloader = DataLoader(my_training_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
            loss_fn = torch.nn.CrossEntropyLoss()
            # loss_fn = torch.nn.KLDivLoss(reduction='batchmean') # reduction='batchmean'
            optimizer = torch.optim.SGD(local_model.parameters(), lr=1e-3)
            
            epoch_loss = []
            for t in range(epochs):
                epoch_loss.append(np.mean(train(train_dataloader, local_model, loss_fn, optimizer)))
            local_loss_record[client_id].append(np.mean(epoch_loss))
            
            client_models.append(local_model)
            
            # Testing the local_model to its own data
            acc, cfmtx = test(local_model, my_testing_dataset)
            local_acc_bfag_record[client_id].append(acc)
            print(f"Done! Aver. round loss: {np.mean(epoch_loss):>.3f}, acc {acc:>.3f}")
            
        print("    # Server aggregating... ", end="")
        # Aggregation
        global_model = fmodule._model_sum([model * pk for model, pk in zip(client_models, impact_factors)])
        print("Done!")
        
        print("    # Server testing... ", end="")
        acc, cfmtx = test(global_model, global_testing_dataset)
        global_cfmtx_record.append(cfmtx)
        
        # same, diff, sim_mtx = check_global_contrastive(global_model, singleset, device)
        # global_constrastive_info["same"].append(same)
        # global_constrastive_info["diff"].append(diff)
        # global_constrastive_info["sim_mtx"].append(sim_mtx)
        same, diff = 0, 0
        print(f"Done! Avg. acc {acc:>.3f}, same {same:>.3f}, diff {diff:>.3f}")
        # print_cfmtx(cfmtx)
        
    if not Path(f"records/{args.exp_folder}/fedavg").exists():
        os.makedirs(f"records/{args.exp_folder}/fedavg")
    
    json.dump(local_loss_record,        open(f"records/{args.exp_folder}/fedavg/local_loss_record.json", "w"),         cls=NumpyEncoder)
    json.dump(local_acc_bfag_record,    open(f"records/{args.exp_folder}/fedavg/local_acc_bfag_record.json", "w"),     cls=NumpyEncoder)
    json.dump(local_acc_afag_record,    open(f"records/{args.exp_folder}/fedavg/local_acc_afag_record.json", "w"),     cls=NumpyEncoder)
    json.dump(global_cfmtx_record,      open(f"records/{args.exp_folder}/fedavg/global_cfmtx_record.json", "w"),       cls=NumpyEncoder)
    json.dump(global_constrastive_info, open(f"records/{args.exp_folder}/fedavg/global_constrastive_info.json", "w"),  cls=NumpyEncoder)
    