# CUDA_VISIBLE_DEVICES=1 python fim.py --seed 0 --epochs 50 --batch_size 16 --dataset "cifar10" --exp_folder "./jsons/dataset_idx/cifar10/dirichlet/dir_0.1_sparse/50client"
CUDA_VISIBLE_DEVICES=0 python scaffold.py --seed 0 --epochs 5 --round 50 --batch_size 16 --dataset "cifar10" --exp_folder "./jsons/dataset_idx/cifar10/dirichlet/dir_0.1_sparse/50client"
# CUDA_VISIBLE_DEVICES=1 python scaffold.py --seed 0 --epochs 5 --round 10 --batch_size 16 --dataset "cifar10" --exp_folder "./jsons/dataset_idx/cifar10/dirichlet/dir_0.5_sparse/20client"
