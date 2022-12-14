from utils.train_smt import test, print_cfmtx, NumpyEncoder
from pathlib import Path
from torch.utils.data import DataLoader
from utils.dataloader import CustomDataset
from torchvision import datasets, transforms
from utils.proposal_6.model import Model
from utils import fmodule
import torch, argparse, json, os, numpy as np, copy

device = "cuda" if torch.cuda.is_available() else "cpu"

def training(dataset, local_model:Model, global_model:Model, pk, round, batch_size, contrastive, device):
    
    global_model = global_model.to(device)
    local_model = local_model.to(device)
    local_model.train()
    
    inter_client_losses = []
    same_class_dis = []
    different_class_dis = []
    epoch_loss = []
    
    ex_model = None
    if round > 0:
        ex_model = fmodule._model_sub(global_model, pk * local_model)
        ex_model.freeze_grad()
    
    dataloder = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for batch, (X, y) in enumerate(dataloder):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        rep = local_model.feature_extractor(X)
        pred = local_model.classifier(rep)
        loss = loss_fn(pred, y)
        epoch_loss.append(loss.detach().item())
        
        """
        Intra client: contrasitive between client's samples
        """
        simi_loss = 0
        for i in range(0, rep.shape[0] - 1, 2):
            similarity = rep[i] @ rep[i+1]
            if y[i].detach().item() == y[i+1].detach().item():
                simi_loss += 1 - similarity
                same_class_dis.append(similarity.detach().item())
            else:
                simi_loss += similarity
                different_class_dis.append(similarity.detach().item())
        simi_loss = simi_loss * 2 / rep.shape[0]
        
        """
        Inter client: the logits must be zeros vectors
        """
        inter_client_loss = 0
        if ex_model is not None:
            logits = ex_model.classifier(rep)
            inter_client_loss = 0.5 * torch.sum(torch.pow(logits, 2)) / (logits.shape[0])
            inter_client_losses.append(inter_client_loss.detach().item())
        
        # Backpropagation
        if contrastive:
            loss += 0.1 * simi_loss + 0.01 * inter_client_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return np.mean(epoch_loss), \
        np.mean(same_class_dis) if len(same_class_dis) else 0, \
        np.mean(different_class_dis) if len(different_class_dis) else 0, \
        np.mean(inter_client_losses) if len(inter_client_losses) else 0


# def classification_training(dataset, local_model:Model, batch_size, device):
#     local_model = local_model.to(device)
#     local_model.train()
    
#     dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=False)
#     optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-3)
#     loss_fn = torch.nn.CrossEntropyLoss()
    
#     epoch_loss = []
#     for batch, (X, y) in enumerate(dataloader):
#         X, y = X.to(device), y.to(device)
#         rep = local_model.feature_extractor(X)
#         pred = local_model.classifier(rep)
        
#         loss = loss_fn(pred, y)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         epoch_loss.append(loss.detach().item())
    
#     return np.mean(epoch_loss)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--contrastive", type=int, required=False, default=0)
    
    args = parser.parse_args()
    batch_size = args.batch_size
    epochs = args.epochs
    
    training_data = datasets.MNIST(
        root="../data",
        train=True,
        download=False,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
    )
    
    testing_data = datasets.MNIST(
        root="../data",
        train=False,
        download=False,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
    )
    
    client_id_list = [0,1,2,3,4]
    clients_dataset = [CustomDataset(training_data, json.load(open(f"./jsons/client{client_id}.json", 'r'))) for client_id in client_id_list]
    total_sample = np.sum([len(dataset) for dataset in clients_dataset])
    
    
    global_model = Model().to(device)
    local_models = [None for client_id in client_id_list]
    
    local_loss_record = {client_id:[] for client_id in client_id_list}
    local_cfmtx_bfag_record = {client_id:[] for client_id in client_id_list}
    local_cfmtx_afag_record = {client_id:[] for client_id in client_id_list}
    global_cfmtx_record = []
    U_cfmtx_record = []
    local_constrastive_info = {client_id:{"same": [], "diff": []} for client_id in client_id_list}
    
    for cur_round in range(args.round):
        print("============ Round {} ==============".format(cur_round))
        impact_factors = [len(clients_dataset[client_id])/total_sample for client_id in client_id_list]
        
        # Local training
        for client_id in client_id_list:
            print("    Client {} training... ".format(client_id), end="")
            
            """
            Each client has a separate model, personalized for that dataset
            The purpose is to train the feature extractor accross the clients,
            but the classifier is personalized
            """
            mydataset = clients_dataset[client_id]   
            if local_models[client_id] is None: 
                local_models[client_id] = copy.deepcopy(global_model)
            else:
                local_models[client_id].feature_extractor = copy.deepcopy(global_model.feature_extractor)
            
            # Training process
            print("\n\t  Local training...", end="")
            epoch_loss, same_ , diff_, inter_ = [], [], [], []
            for t in range(epochs):
                classification_loss, same, diff, inter = training(mydataset, local_models[client_id], global_model, impact_factors[client_id], 
                                                            t, batch_size, args.contrastive > 0, device)
                same_.append(same)
                diff_.append(diff)
                inter_.append(inter)
                epoch_loss.append(classification_loss)
            print(f"Done! Aver. loss: {np.mean(epoch_loss):>.3f}, same {np.mean(same_):>.3f}, diff {np.mean(diff_):>.3f}, inter {np.mean(inter_):>.3f}")
            local_constrastive_info[client_id]["same"].append(np.mean(same_))
            local_constrastive_info[client_id]["diff"].append(np.mean(diff_))
            local_loss_record[client_id].append(np.mean(epoch_loss))
            
            # Testing the local_model to its own data
            cfmtx = test(local_models[client_id], mydataset)
            local_cfmtx_bfag_record[client_id].append(cfmtx)
            
        print("    # Server aggregating... ", end="")
        # Aggregation
        global_model = fmodule._model_sum([model * pk for model, pk in zip(local_models, impact_factors)])
        print("Done!")
        
        print("    # Server testing... ", end="")
        cfmtx = test(global_model, testing_data)
        global_cfmtx_record.append(cfmtx)
        print("Done!")
        np.set_printoptions(precision=2, suppress=True)
        print_cfmtx(cfmtx)
        
    if not Path("records/fedavgv2").exists():
        os.makedirs("records/fedavgv2")
    
    json.dump(local_loss_record,        open("records/fedavgv2/local_loss_record.json", "w"),         cls=NumpyEncoder)
    json.dump(local_cfmtx_bfag_record,  open("records/fedavgv2/local_cfmtx_bfag_record.json", "w"),   cls=NumpyEncoder)
    json.dump(global_cfmtx_record,      open("records/fedavgv2/global_cfmtx_record.json", "w"),       cls=NumpyEncoder)
    json.dump(local_constrastive_info,  open("records/fedavgv2/local_constrastive_info.json", "w"),   cls=NumpyEncoder)