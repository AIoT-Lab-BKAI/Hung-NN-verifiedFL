from torch.utils.data import DataLoader
from utils.dataloader import CustomDataset
from torchvision import datasets, transforms
from utils.base_model import NeuralNetwork
import torch, argparse, json


device = "cuda" if torch.cuda.is_available() else "cpu"


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
            
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
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
    json_data = []
    for client_id in client_id_list: 
        json_data += json.load(open(f"./jsons/client{client_id}.json", 'r')) 
    
    print(json_data)
    clients_dataset = CustomDataset(training_data, json_data)
    train_dataloader = DataLoader(clients_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(testing_data, batch_size=32, shuffle=False, drop_last=False)

    model = NeuralNetwork().cuda()
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")