# CUDA_VISIBLE_DEVICES=1 python fedavg.py --seed 0 --epochs 4 --round 50 --batch_size 16 --dataset "cifar10" --exp_folder "../jsons/baseline/simple_9"
# CUDA_VISIBLE_DEVICES=1 python scaffold.py --seed 0 --epochs 4 --round 50 --batch_size 16 --dataset "cifar10" --exp_folder "../jsons/baseline/simple_9" > scaffold.log &
CUDA_VISIBLE_DEVICES=0 python fim.py --seed 0 --epochs 50 --batch_size 16 --dataset "cifar10" --exp_folder "./jsons/dataset_idx/cifar10/dirichlet/dir_0.1_sparse/100client"


# CUDA_VISIBLE_DEVICES=0,1 python feddyn.py --epochs 8 --batch_size 4 --round 100
# CUDA_VISIBLE_DEVICES=0,1 python fedavg.py --epochs 8 --batch_size 4 --round 100
# CUDA_VISIBLE_DEVICES=0,1 python scaffold.py --epochs 8 --batch_size 4 --round 100
# CUDA_VISIBLE_DEVICES=0 python fedprox.py --epochs 8 --batch_size 4 --round 100
