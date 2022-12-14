from fedavg import train
from utils.train_smt import train_representation, test
from utils.aggregate import aggregate, check_representations
from pathlib import Path
from torch.utils.data import DataLoader
from utils.dataloader import CustomDataset
from torchvision import datasets, transforms
from utils.model import NeuralNetwork, DNN
from torchmetrics import ConfusionMatrix

import torch, argparse, json, os, numpy as np, copy

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_mask(dim, dataset):
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    mask = torch.zeros([dim, dim])
    for X, y in train_dataloader:
        label = y.item()
        mask[label, label] = 1
    return mask

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def print_cfmtx(mtx):
    for i in range(mtx.shape[0]):
        for j in range(mtx.shape[1]):
            if i == j:
                print(f"\033[48;5;225m{mtx[i,j]:>.3f}\033[0;0m", end="  ")
            else:
                print(f"{mtx[i,j]:>.3f}", end="  ")
        print()
    return


def test_ideal(model, testing_data):
    test_loader = DataLoader(testing_data, batch_size=32, shuffle=True, drop_last=False)
    model.cuda()

    loss_fn = torch.nn.CrossEntropyLoss()
    device = "cuda"

    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    model.eval()
    test_loss, correct = 0, 0
    confmat = ConfusionMatrix(num_classes=10).to(device)
    cmtx = 0
    
    label_list = [[1,2],[3,4],[5,6],[7,8],[9,0]]
    def recoord(labels):
        coord = []
        for label in labels:
            for idx in range(len(label_list)):
                if label in label_list[idx]:
                    coord.append(idx)
        true_psi = torch.zeros([len(labels), len(label_list)])
        """
        coord = [0,1,2,1,1,3]
        """
        for i in range(len(coord)):
            true_psi[i][coord[i]] = 1
        return true_psi

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            true_psi = recoord(y).to(device)
            pred = model(X, true_psi)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            cmtx += confmat(pred, y)

    test_loss /= num_batches
    correct /= size

    acc, cfmtx =  correct, cmtx.cpu().numpy()
    down = np.sum(cfmtx, axis=1, keepdims=True)
    down[down == 0] = 1
    cfmtx = cfmtx/down
    return cfmtx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--round", type=int, default=1)
    args = parser.parse_args()
    batch_size = args.batch_size
    epochs = args.epochs

    training_data = datasets.MNIST(
        root="./data",
        train=True,
        download=False,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
    )
    
    testing_data = datasets.MNIST(
        root="data",
        train=False,
        download=False,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
    )
    
    client_id_list = [0,1,2,3,4]
    clients_dataset = [CustomDataset(training_data, json.load(open(f"./jsons/client{client_id}.json", 'r'))) for client_id in client_id_list]
    
    global_model = DNN(bias=False).to(device)
    # global_model = NeuralNetwork(bias=False).to(device)
    local_loss_record = {client_id:[] for client_id in client_id_list}
    local_cfmtx_bfag_record = {client_id:[] for client_id in client_id_list}
    local_cfmtx_afag_record = {client_id:[] for client_id in client_id_list}
    global_cfmtx_record = []
    U_cfmtx_record = []
    
    for cur_round in range(args.round):
        print("============ Round {} ==============".format(cur_round))
        condense_representation_list = []
        client_models = []
        
        # Local training
        for client_id in client_id_list:
            print("    Client {} training... ".format(client_id), end="")
            # Training process
            mydataset = clients_dataset[client_id]
            train_dataloader = DataLoader(mydataset, batch_size=batch_size, shuffle=True, drop_last=False)
            
            U = create_mask(dim=10, dataset=mydataset)
                
            local_model = copy.deepcopy(global_model)
            local_model.update_mask(U)

            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(local_model.parameters(), lr=1e-3)

            epoch_loss = []
            for t in range(epochs):
                epoch_loss.append(np.mean(train(train_dataloader, local_model, loss_fn, optimizer)))
            local_loss_record[client_id].append(np.mean(epoch_loss))
            
            client_models.append(local_model)
            
            # Testing the local_model to its own data
            cfmtx = test(local_model, mydataset)
            local_cfmtx_bfag_record[client_id].append(cfmtx)
            
            # Generating representations
            condense_representation = torch.randn([100], requires_grad=True).unsqueeze(0).cuda()
            for t in range(epochs):
                condense_representation = train_representation(train_dataloader, local_model, condense_representation)
            condense_representation_list.append(condense_representation.detach().cpu())
            local_model.zero_grad()
            print(f"Done! Aver. round loss: {np.mean(epoch_loss):>.3f}")
        
        print("    # Server aggregating... ", end="")
        # Aggregation
        condensed_rep = torch.cat(condense_representation_list, dim=0)
        global_model = aggregate(client_models, condensed_rep, indexes=client_id_list)
        print("Done!")
        
        # Testing
        print("    # Server testing... ", end="")
        cfmtx = test_ideal(global_model, testing_data)
        global_cfmtx_record.append(cfmtx)
        print("Done!")
        np.set_printoptions(precision=2, suppress=True)
        print_cfmtx(cfmtx)
        
        U_cfmtx = check_representations(global_model, condensed_rep, testing_data, "cuda")
        U_cfmtx_record.append(U_cfmtx)
    
    if not Path("records/proposal_ideal").exists():
        os.makedirs("records/proposal_ideal")
    
    json.dump(local_loss_record,        open("records/proposal_ideal/local_loss_record.json", "w"),         cls=NumpyEncoder)
    json.dump(local_cfmtx_bfag_record,  open("records/proposal_ideal/local_cfmtx_bfag_record.json", "w"),   cls=NumpyEncoder)
    json.dump(global_cfmtx_record,      open("records/proposal_ideal/global_cfmtx_record.json", "w"),       cls=NumpyEncoder)
    json.dump(U_cfmtx_record,           open("records/proposal_ideal/U_cfmtx_record.json", "w"),            cls=NumpyEncoder)