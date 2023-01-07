from utils.train_smt import test, print_cfmtx, NumpyEncoder
from utils.reader import read_jsons
from utils.parser import read_arguments

from utils import fmodule
from pathlib import Path
from torch.utils.data import DataLoader
from utils.FIM2 import MLP2, flatten_model, compute_grads, compute_natural_grads, get_module_from_model
import torch, json, os, numpy as np, copy, random
import torch.nn.functional as F

def set_seed(seed):
    random.seed(1+seed)
    np.random.seed(21+seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(12+seed)
    torch.cuda.manual_seed_all(123+seed)

device = "cuda" if torch.cuda.is_available() else "cpu"


def FIM_step(centroid, clients_training_dataset, client_id_list, eta=0.1):
    print("Perform one step natural gradient with Fisher Matrix... ", end="")
    # Each client compute grads using their own dataset but the centroid's model
    grad_list = []
    for client_id in client_id_list:
        # Training process
        my_training_dataset = clients_training_dataset[client_id]
        train_dataloader = DataLoader(my_training_dataset, batch_size=len(my_training_dataset), shuffle=True, drop_last=False)
        
        X, y = next(iter(train_dataloader))
        X, y = X.to(device), y.to(device)
        grad_list.append(compute_grads(centroid, X, y))
        
    # Server average the gradients
    vgrad = [None for i in range(len(grad_list[0]))]
    for grad in grad_list:
        # Example: grad = (A0, A1, G1, G2, dw1, dw2)
        for i in range(len(grad)):
            vgrad[i] = grad[i] if vgrad[i] is None else vgrad[i] + grad[i]
    
    vgrad = [vg/len(grad_list) for vg in vgrad]
    
    # Server compute natural gradients
    nat_grads = compute_natural_grads(vgrad)
    
    # Server update the model
    global_model = step(centroid.cpu(), nat_grads, lr=eta)
    print("Done!")   
    return global_model


def train(dataloader, model, loss_fn, optimizer):   
    model = model.cuda()
    model.train()
    losses = []
        
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        # # KL
        pred = F.softmax(pred, dim=1) 
        ground = F.softmax(F.one_hot(y, 10) * 1.0, dim=1) 
        loss = torch.sum(ground * torch.log(ground/pred)) / (ground.shape[0] * ground.shape[1])
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        
    return losses


@torch.no_grad()
def centroid_model(models):
    """
    Return index of the model whose the closest to all other model
    """
    try: # Vectorization
        fms = [flatten_model(m) for m in models]
        fms = torch.cat(fms).unsqueeze(1)
        dis_mtx = torch.norm(fms - fms.transpose(0,1), dim=2)
    except: # Iteration
        dis_mtx = torch.zeros([len(models), len(models)])
        for id1 in range(len(models)):
            for id2 in range(len(models)):
                dis_mtx[id1][id2] = (models[id1] - models[id2]).norm()
    
    best = torch.argmin(torch.sum(dis_mtx, dim=0)).item()
    return best


def step(centroid: MLP2, nat_grads, lr = 1.0):
    res = MLP2()
    res_modules = get_module_from_model(res)
    cen_modules = get_module_from_model(centroid)
    with torch.no_grad():
        for r_module, c_module, nat_grad in zip(res_modules, cen_modules, nat_grads):
            r_module._parameters['weight'].copy_(c_module._parameters['weight'] - lr * nat_grad)
    return res


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
    
    print("============ Start ==============")
    client_models = []
    impact_factors = [len(clients_training_dataset[client_id])/total_sample for client_id in client_id_list]
    
    # Local training
    for client_id in client_id_list:
        print(f"Client {client_id:>2d} training... ", end="")
        # Training process
        my_training_dataset = clients_training_dataset[client_id]
        my_testing_dataset = clients_testing_dataset[client_id]
    
        local_model = copy.deepcopy(global_model)
        
        train_dataloader = DataLoader(my_training_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        # loss_fn = torch.nn.CrossEntropyLoss()
        loss_fn = torch.nn.KLDivLoss(reduction='batchmean') # reduction='batchmean'
        optimizer = torch.optim.SGD(local_model.parameters(), lr=1e-3)
        
        epoch_loss = []
        for t in range(epochs):
            epoch_loss.append(np.mean(train(train_dataloader, local_model, loss_fn, optimizer)))
        
        client_models.append(local_model)
        
        # Testing the local_model
        test_acc, _ = test(local_model, my_testing_dataset)
        train_acc, _ = test(local_model, my_training_dataset)
        
        norm_diff = (local_model - global_model).norm()
        print(f"Done! Aver. round loss: {np.mean(epoch_loss):>.3f}, test acc {test_acc:>.3f}, train acc {train_acc:>.3f}, shift length {norm_diff:>.5f}")

    centroid = fmodule._model_sum([model * pk for model, pk in zip(client_models, impact_factors)])
    
    global_model = FIM_step(centroid, clients_training_dataset, client_id_list, eta=1).to(device)
    
    # Testing
    print("Server testing... ", end="")
    acc, cfmtx = test(global_model, global_testing_dataset, device)
    print(f"Done! Avg. acc {acc:>.3f}")
    
    # global_model = FIM_step(global_model, clients_training_dataset, client_id_list, eta=1)
    
    # # Testing
    # print("Server testing... ", end="")
    # acc, cfmtx = test(global_model, global_testing_dataset, device)
    # print(f"Done! Avg. acc {acc:>.3f}")
    
    print("Centroid testing... ", end="")
    acc, cfmtx = test(centroid, global_testing_dataset, device)
    print(f"Done! Avg. acc {acc:>.3f}")
    # print_cfmtx(cfmtx)
    