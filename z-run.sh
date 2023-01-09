CUDA_VISIBLE_DEVICES=1 python fim.py --seed 0 --epochs 100 --batch_size 16 --dataset "mnist" --exp_folder "./jsons/baseline/simple_7"
# CUDA_VISIBLE_DEVICES=1 python fedavg.py --seed 0 --epochs 2 --round 50 --batch_size 16 --dataset "mnist" --exp_folder "./jsons/baseline/simple_7"
# CUDA_VISIBLE_DEVICES=1 python scaffold.py --seed 0 --epochs 2 --round 50 --batch_size 16 --dataset "mnist" --exp_folder "./jsons/baseline/simple_7"
