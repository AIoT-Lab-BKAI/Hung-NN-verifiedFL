import json
from utils.dataloader import CustomDataset
from torchvision import datasets, transforms

def read_jsons(folder_path, dataset="mnist"):
    
    if dataset == "mnist":
        training_data = datasets.MNIST(
            root="../easyFL/benchmark/mnist/data",
            train=True,
            download=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        )
        testing_data = datasets.MNIST(
            root="../easyFL/benchmark/mnist/data",
            train=False,
            download=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        )
        
    elif dataset == "cifar10":
        training_data = datasets.CIFAR10(
            root="../easyFL/benchmark/cifar10/data",
            train=True,
            download=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        )

        testing_data = datasets.CIFAR10(
            root="../easyFL/benchmark/cifar10/data",
            train=False,
            download=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        )
        
    elif dataset == "cifar100":
        training_data = datasets.CIFAR100(
            root="../easyFL/benchmark/cifar100/data",
            train=True,
            download=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        )

        testing_data = datasets.CIFAR100(
            root="../easyFL/benchmark/cifar100/data",
            train=False,
            download=False,
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
        
    singleset_json = []
    for client_id in training_jsons.keys():
        singleset_json += training_jsons[client_id]
    singleset = CustomDataset(training_data, singleset_json)
        
    return num_client, training_set, testing_set, testing_data, singleset