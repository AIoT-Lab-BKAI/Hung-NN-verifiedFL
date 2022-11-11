import json
from utils.dataloader import CustomDataset
from torchvision import datasets, transforms

def read_jsons(folder_path, dataset="mnist"):
    
    if dataset == "mnist":
        training_data = datasets.MNIST(
            root="../data",
            train=True,
            download=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        )
        testing_data = datasets.MNIST(
            root="../data",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        )
        
    elif dataset == "cifar10":
        training_data = datasets.CIFAR10(
            root="../data",
            train=True,
            download=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        )

        testing_data = datasets.CIFAR10(
            root="../data",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        )
        
    elif dataset == "cifar100":
        training_data = datasets.CIFAR100(
            root="../data",
            train=True,
            download=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        )

        testing_data = datasets.CIFAR100(
            root="../data",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        )
        
    else:
        raise Exception("Not support", dataset, "yet")
    
    training_jsons = json.load(open(f"{folder_path}/clients_training_data.json", 'r'))
    testing_jsons = json.load(open(f"{folder_path}/clients_testing_data.json", 'r'))
    
    training_set = []
    testing_set = []
    
    num_client = 0
    for client_id in training_jsons.keys():
        num_client += 1
        training_set.append(CustomDataset(training_data, training_jsons[client_id]))
        testing_set.append(CustomDataset(testing_data, testing_jsons[client_id]))
        
    return num_client, training_set, testing_set, testing_data