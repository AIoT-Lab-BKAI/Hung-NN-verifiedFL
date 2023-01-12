# CUDA_VISIBLE_DEVICES=3 python feddyn.py --seed 0 --epochs 1 --round 100 --batch_size 16 --dataset "cifar10" --idx_folder "./jsons/dataset_idx/cifar10/dirichlet/dir_0.1_sparse/500client"
# CUDA_VISIBLE_DEVICES=1,0,3,2 python scaffold.py --seed 0 --epochs 5 --round 5 --batch_size 16 --dataset "cifar10" --idx_folder "./jsons/dataset_idx/cifar10/dirichlet/dir_0.1_sparse/500client"
# CUDA_VISIBLE_DEVICES=2 python fednova.py --seed 0 --epochs 5 --round 300 --batch_size 16 --dataset "cifar10" --idx_folder "./jsons/dataset_idx/cifar10/dirichlet/dir_0.1_sparse/500client" > fednova.log &
# CUDA_VISIBLE_DEVICES=2 python fedprox.py --seed 0 --epochs 5 --round 300 --batch_size 16 --dataset "cifar10" --idx_folder "./jsons/dataset_idx/cifar10/dirichlet/dir_0.1_sparse/500client" > fedprox.log &
CUDA_VISIBLE_DEVICES=2 python fedavg.py --seed 0 --epochs 5 --round 300 --batch_size 16 --dataset "cifar10" --data_folder "../easyFL/benchmark/cifar10/data" --idx_folder "./jsons/dataset_idx/cifar10/dirichlet/dir_0.1_sparse/500client"
# CUDA_VISIBLE_DEVICES=2 python fim.py --seed 0 --epochs 0 --batch_size 16 --dataset "cifar10" --idx_folder "./jsons/dataset_idx/cifar10/dirichlet/dir_0.1_sparse/500client"
