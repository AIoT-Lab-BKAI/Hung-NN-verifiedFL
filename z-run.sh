# CUDA_VISIBLE_DEVICES=2 python feddyn.py --seed 0 --epochs 8 --round 100 --batch_size 16 --dataset "cifar10" --exp_folder "./jsons/dataset_idx/cifar10/dirichlet/dir_0.1_sparse/500client"
# CUDA_VISIBLE_DEVICES=1 python scaffold.py --seed 0 --epochs 5 --round 100 --batch_size 16 --dataset "cifar10" --exp_folder "./jsons/dataset_idx/cifar10/dirichlet/dir_0.1_sparse/50client"
# CUDA_VISIBLE_DEVICES=1 python fednova.py --seed 0 --epochs 5 --round 100 --batch_size 16 --dataset "cifar10" --exp_folder "./jsons/dataset_idx/cifar10/dirichlet/dir_0.1_sparse/50client"
CUDA_VISIBLE_DEVICES=2 python fedavg.py --seed 0 --epochs 8 --round 100 --batch_size 16 --dataset "cifar10" --exp_folder "./jsons/dataset_idx/cifar10/dirichlet/dir_0.1_sparse/500client"
CUDA_VISIBLE_DEVICES=1 python fim.py --seed 0 --epochs 100 --batch_size 16 --dataset "cifar10" --exp_folder "./jsons/dataset_idx/cifar10/dirichlet/dir_0.1_sparse/500client" > fim-100r.log &
