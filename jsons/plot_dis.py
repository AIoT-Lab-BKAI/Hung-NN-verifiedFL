import argparse
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

def recursive_plot(folder_path):
    for f in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, f)):
            recursive_plot(os.path.join(folder_path,f))
        else:
            if ".csv" in f:
                # Visualize confusion matrix
                print("Checking", os.path.join(folder_path,f))
                dis = np.loadtxt(os.path.join(folder_path, f), delimiter=",", dtype=np.int16)
                # plt.figure(figsize=(int(dis.shape[1]/2),int(dis.shape[0]/2)))
                plt.figure(figsize=(dis.shape[1] - 2, dis.shape[0]))
                sns.heatmap(data=dis, annot=True, fmt="d", cmap="YlGnBu", cbar=False)
                figname = f.split(".")[0]
                plt.tight_layout()
                plt.savefig(folder_path + "/" + figname + ".png")
                plt.close('all')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="./baseline/simple_7")
    args = parser.parse_args()

    recursive_plot(args.folder)