import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="fedavg")
args = parser.parse_args()

folder = args.folder

global_cfmtx = json.load(open(f"./records/{folder}/global_cfmtx_record.json", "r"))
local_cfmtx = json.load(open(f"./records/{folder}/local_cfmtx_bfag_record.json", "r"))

if "proposal" in folder:
    U_cfmtx = json.load(open(f"./records/{folder}/U_cfmtx_record.json", "r"))

for r in range(100):
    if not Path(f"figures/{folder}/round_{r}").exists():
        os.makedirs(f"figures/{folder}/round_{r}")

    if "proposal" in folder:
        # Server plot
        f, axes = plt.subplots(1,2, figsize=(8,4.5), facecolor='white')
        s = sns.heatmap(ax=axes[0], data=global_cfmtx[r], annot=True, fmt=".2f", cmap="YlGnBu", cbar=False)
        axes[0].set_title(f"Server prediction, Acc = {np.mean(np.diag(global_cfmtx[r])):>.3f}")

        s = sns.heatmap(ax=axes[1], data=U_cfmtx[r], annot=True, fmt=".2f", cmap="YlGnBu", cbar=False)
        axes[1].set_title(f"Representation classification, Acc = {np.mean(np.diag(U_cfmtx[r])):>.3f}")

        plt.suptitle(f"Server after aggregation at round {r}", fontsize=20)
        f.tight_layout()
    else:
        f = plt.figure(figsize=(4.5,4.5), facecolor='white')
        s = sns.heatmap(data=global_cfmtx[r], annot=True, fmt=".2f", cmap="YlGnBu", cbar=False)
        plt.title(f"Server prediction, Acc = {np.mean(np.diag(global_cfmtx[r])):>.3f}")
        f.tight_layout()
        
    plt.savefig(f"figures/{folder}/round_{r}/server.png", bbox_inches="tight", transparent=False, facecolor=f.get_facecolor())
    plt.close('all')

    # Client plots
    f, axes = plt.subplots(1,5, figsize=(20,4.5), facecolor='white')
    for client_id in local_cfmtx.keys():
        data = local_cfmtx[client_id][r]
        sns.heatmap(ax=axes[int(client_id)], data=data, annot=True, fmt=".2f", cmap="YlGnBu", cbar=False)
        axes[int(client_id)].set_title(f"Client {client_id}, Acc = {np.mean(np.diag(data)):>.3f}")
    plt.suptitle(f"Client's prediction after training at round {r}", fontsize=20)
    f.tight_layout()
    plt.savefig(f"figures/{folder}/round_{r}/clients.png", bbox_inches="tight", transparent=False, facecolor=f.get_facecolor())
    plt.close('all')
    
    print("Done: ", f"figures/{folder}/round_{r}")