# #!/bin/bash
# #$ -cwd
# #$ -l rt_G.small=1
# #$ -l h_rt=36:00:00
# #$ -o /home/aaa10078nj/Federated_Learning/HungNN-verifiedFL/logs/mnist/$JOB_NAME_$JOB_ID.log
# #$ -j y

# source /etc/profile.d/modules.sh
# module load gcc/11.2.0
# module load openmpi/4.1.3
# module load cuda/11.5/11.5.2
# module load cudnn/8.3/8.3.3
# module load nccl/2.11/2.11.4-1
# module load python/3.10/3.10.4
# source ~/venv/pytorch1.11+horovod/bin/activate

# LOG_DIR="/home/aaa10078nj/Federated_Learning/HungNN-verifiedFL/logs/mnist/$JOB_NAME_$JOB_ID"
# rm -r ${LOG_DIR}
# mkdir ${LOG_DIR}

# #Dataset
# DATA_DIR="$SGE_LOCALDIR/$JOB_ID/"
# cp -r ./easyFL/benchmark/mnist/data ${DATA_DIR}

ALG="scaffold"
SEED=0
WANDB=1
ROUND=400
EPOCH_PER_ROUND=5
BATCH=16
DATASET=mnist
IDX_DIR="./jsons/dataset_idx/mnist/dirichlet/dir_1_sparse/200client"

cd Hung-NN-verifiedFL

python ${ALG}.py --wandb ${WANDB} --seed ${SEED} --dataset ${DATASET} --data_folder ${DATA_DIR} --log_folder ${LOG_DIR} --idx_folder ${IDX_DIR} --round ${ROUND} --epochs ${EPOCH_PER_ROUND} --batch_size ${BATCH} 