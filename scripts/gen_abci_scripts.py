import os
from pathlib import Path

dataset = "mnist"
noniid = "dir_1_sparse"
N = 500
total_epochs = 2000
batch_size = 16

model = "cnn"
algos = ["scaffold", "fedavg", "fedprox", "fednova", "feddyn", "singleset"]

if not Path(f"./{dataset}/{noniid}/{N}_clients").exists():
    os.makedirs(f"./{dataset}/{noniid}/{N}_clients")


header_text = "\
#!/bin/bash\n\
#$ -cwd\n\
#$ -l rt_G.small=1\n\
#$ -l h_rt=36:00:00\n\
#$ -o /home/aaa10078nj/Federated_Learning/HungNN-verifiedFL/logs/mnist/$JOB_NAME_$JOB_ID.log\n\
#$ -j y\n\n\
source /etc/profile.d/modules.sh\n\
module load gcc/11.2.0\n\
module load openmpi/4.1.3\n\
module load cuda/11.5/11.5.2\n\
module load cudnn/8.3/8.3.3\n\
module load nccl/2.11/2.11.4-1\n\
module load python/3.10/3.10.4\n\
source ~/venv/pytorch1.11+horovod/bin/activate\n\n\
LOG_DIR=\"/home/aaa10078nj/Federated_Learning/HungNN-verifiedFL/logs/mnist/$JOB_NAME_$JOB_ID\"\n\
rm -r ${LOG_DIR}\n\
mkdir ${LOG_DIR}\n\n\
#Dataset\n\
DATA_DIR=\"$SGE_LOCALDIR/$JOB_ID/\"\n\
cp -r ./easyFL/benchmark/mnist/data ${DATA_DIR}\n\n\
"

body_text = "\
python ${ALG}.py \
--wandb ${WANDB} \
--seed ${SEED} \
--dataset ${DATASET} \
--data_folder ${DATA_DIR} \
--log_folder ${LOG_DIR} \
--idx_folder ${IDX_DIR} \
--round ${ROUND} \
--epochs ${EPOCH_PER_ROUND} \
--batch_size ${BATCH} \
"

formated_command = "\
ALG=\"{}\"\n\
SEED=0\n\
WANDB=1\n\
ROUND={}\n\
EPOCH_PER_ROUND={}\n\
BATCH={}\n\
DATASET={}\n\
IDX_DIR=\"./jsons/dataset_idx/{}/dirichlet/{}/{}client\"\n\n\
cd Hung-NN-verifiedFL\n\n\
"

for E in [1, 5, 10]:
    task_name = f"{dataset}_{noniid}_N{N}_E{E}"

    for algo in algos:
        command = formated_command.format(
            algo, int(total_epochs/E), E, 16, dataset, dataset, noniid, N
        )
        
        file = open(f"./{dataset}/{noniid}/{N}_clients/{task_name}_{algo}.sh", "w")
        file.write(header_text + command + body_text)
        file.close()