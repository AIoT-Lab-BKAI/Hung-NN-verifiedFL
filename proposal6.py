from utils.train_smt import test, print_cfmtx, NumpyEncoder
from pathlib import Path
from torch.utils.data import DataLoader
from utils.dataloader import CustomDataset
from torchvision import datasets, transforms
from utils.proposal_6.model import Model
from utils import fmodule
import torch, argparse, json, os, numpy as np, copy

device = "cuda" if torch.cuda.is_available() else "cpu"


def classifier_training(dataset, model:Model, batch_size=4, device="cuda:1"):
    model = model.to(device)
    model.train()
    losses = []

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        r_x = model.feature_extractor(X)
        l_x = model.classifier(r_x)
        loss = loss_fn(l_x, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    return np.mean(losses)


def representation_training(dataset, local_model:Model, model_p:Model, device="cuda:1"):
    """
    This method trains for a discriminative representation space,
    using constrastive learning
    """
    feature_extractor = local_model.feature_extractor.to(device)
    feature_extractor.train()
    same_class_dis = []
    different_class_dis = []
    inter_losses = []
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(feature_extractor.parameters(), lr=1e-3)
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        representations = feature_extractor(X)
        
        similarity = (representations[0] @ representations[1]) / (torch.norm(representations[0]).detach() * torch.norm(representations[1]).detach())
        if y[0].detach().item() == y[1].detach().item():
            loss = 1 - similarity
            same_class_dis.append(loss.detach().item())
        else:
            loss = similarity
            different_class_dis.append(loss.detach().item())

        inter_client_loss = 0.5 * torch.sum(torch.pow(model_p.classifier(representations), 2))
        inter_losses.append(inter_client_loss.detach().item())
        
        # loss = loss + 0.01 * inter_client_loss 
        # # Backpropagation
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

    return np.mean(same_class_dis) if len(same_class_dis) else 0, np.mean(different_class_dis) if len(different_class_dis) else 0, np.mean(inter_losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--round", type=int, default=1)
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
    local_models = [Model().to(device) for client_id in client_id_list]
    local_loss_record = {client_id:[] for client_id in client_id_list}
    local_cfmtx_bfag_record = {client_id:[] for client_id in client_id_list}
    local_cfmtx_afag_record = {client_id:[] for client_id in client_id_list}
    global_cfmtx_record = []
    U_cfmtx_record = []
    
    for cur_round in range(args.round):
        print("============ Round {} ==============".format(cur_round))
        impact_factors = [len(clients_dataset[client_id])/total_sample for client_id in client_id_list]
        
        # Local training
        for client_id in client_id_list:
            print("\tClient {} training... ".format(client_id))
            # Training process
            mydataset = clients_dataset[client_id]
            
            model_p = fmodule._model_sub(global_model, impact_factors[client_id] * local_models[client_id])
            model_p.freeze_grad()
            local_models[client_id] = copy.deepcopy(global_model)
            
            # Classifier training
            epoch_loss = []
            print("\t  Classifier training ...", end="")
            for t in range(epochs):
                epoch_loss.append(np.mean(classifier_training(mydataset, local_models[client_id], batch_size, device)))
            local_loss_record[client_id].append(np.mean(epoch_loss))
            print(f"Done! Aver. round loss: {np.mean(epoch_loss):>.3f}")
            
            # Representation training
            print("\t  Representation training ...", end="")
            same_class_dis = []
            diff_class_dis = []
            inter_client_loss = []
            for t in range(epochs):
                same, diff, inter = representation_training(mydataset, local_models[client_id], model_p, device)
                same_class_dis.append(same)
                diff_class_dis.append(diff)
                inter_client_loss.append(inter)
                
            print(f"Done! Aver. same {np.mean(same_class_dis):.5f} and diff {np.mean(diff_class_dis):>.5f}, inter client similarity {np.mean(inter_client_loss):.5f}")
            
            # Testing the local_model to its own data
            cfmtx = test(local_models[client_id], mydataset)
            local_cfmtx_bfag_record[client_id].append(cfmtx)
            
        print("\t# Server aggregating... ", end="")
        # Aggregation
        global_model = fmodule._model_sum([model * pk for model, pk in zip(local_models, impact_factors)])
        print("Done!")
        
        print("\t# Server testing... ", end="")
        cfmtx = test(global_model, testing_data)
        global_cfmtx_record.append(cfmtx)
        print("Done!")
        np.set_printoptions(precision=2, suppress=True)
        print_cfmtx(cfmtx)
        
    if not Path("records/proposal6").exists():
        os.makedirs("records/proposal6")
    
    json.dump(local_loss_record,        open("records/proposal6/local_loss_record.json", "w"),         cls=NumpyEncoder)
    json.dump(local_cfmtx_bfag_record,  open("records/proposal6/local_cfmtx_bfag_record.json", "w"),   cls=NumpyEncoder)
    json.dump(global_cfmtx_record,      open("records/proposal6/global_cfmtx_record.json", "w"),       cls=NumpyEncoder)