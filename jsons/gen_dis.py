import torch
from torchvision import datasets
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

training_data = datasets.MNIST(
    root="../data",
    train=True,
    download=False,
)

dis = torch.zeros([5, 10])
list_dir = sorted(os.listdir(), key=lambda x: x)

for i in range(len(list_dir)):
    filename = list_dir[i]
    if ".json" in filename:
        # print(filename)
        indx_list = json.load(open(filename, "r"))
        for idx in indx_list:
            label = training_data.targets[idx]
            dis[i][label] += 1

# print(dis.numpy())
plt.figure(figsize=(10,5))
sns.heatmap(data=dis.numpy(), annot=True, cmap="YlGnBu", cbar=False)
plt.xlabel("Label", fontsize=16)
plt.ylabel("Client", fontsize=16)
plt.savefig("data_dis.png", dpi=128)