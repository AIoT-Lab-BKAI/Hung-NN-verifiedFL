# CUDA_VISIBLE_DEVICES=0 python fedpretrained.py --epochs 8 --batch_size 4 --round 100 --exp_folder "./jsons/baseline/simple_7"
# CUDA_VISIBLE_DEVICES=1 python fedavg.py --epochs 8 --batch_size 4 --round 100 --exp_folder "./jsons/baseline/simple_7"
# CUDA_VISIBLE_DEVICES=1 python test_op.py --epochs 8 --batch_size 4 --round 100 --exp_folder "./jsons/baseline/simple_7"
CUDA_VISIBLE_DEVICES=1 python test_algo.py --epochs 500 --batch_size 4 --exp_folder "./jsons/baseline/simple_7"


# CUDA_VISIBLE_DEVICES=0 python fedavgv2.py --epochs 8 --batch_size 4 --round 100 --contrastive 0 --exp_folder "./jsons/baseline/simple_5"
# CUDA_VISIBLE_DEVICES=0 python fedavg.py --epochs 8 --batch_size 4 --round 100 --contrastive 0 --exp_folder "./jsons/baseline/simple_5"
# CUDA_VISIBLE_DEVICES=0 python scaffold.py --epochs 8 --batch_size 4 --round 100 --contrastive 0 --exp_folder "./jsons/baseline/simple_5"

# CUDA_VISIBLE_DEVICES=0 python fedavgv3.py --epochs 8 --batch_size 4 --round 100 --exp_folder "./jsons/baseline/simple_6"
# CUDA_VISIBLE_DEVICES=0 python fedavgv2.py --epochs 8 --batch_size 4 --round 100 --contrastive 0 --exp_folder "./jsons/baseline/simple_6"
# CUDA_VISIBLE_DEVICES=0 python fedavg.py --epochs 8 --batch_size 4 --round 100 --contrastive 0 --exp_folder "./jsons/baseline/simple_6"
# CUDA_VISIBLE_DEVICES=0 python scaffold.py --epochs 8 --batch_size 4 --round 100 --contrastive 0 --exp_folder "./jsons/baseline/simple_6"

# CUDA_VISIBLE_DEVICES=0 python fedavgv3.py --epochs 8 --batch_size 4 --round 100 --exp_folder "./jsons/baseline/simple_7"
# CUDA_VISIBLE_DEVICES=0 python fedavgv2.py --epochs 8 --batch_size 4 --round 100 --contrastive 0 --exp_folder "./jsons/baseline/simple_7"
# CUDA_VISIBLE_DEVICES=0 python fedavg.py --epochs 8 --batch_size 4 --round 100 --contrastive 0 --exp_folder "./jsons/baseline/simple_7"
# CUDA_VISIBLE_DEVICES=0 python scaffold.py --epochs 8 --batch_size 4 --round 100 --contrastive 0 --exp_folder "./jsons/baseline/simple_7"


# CUDA_VISIBLE_DEVICES=0,1 python feddyn.py --epochs 8 --batch_size 4 --round 100
# CUDA_VISIBLE_DEVICES=0,1 python fedavg.py --epochs 8 --batch_size 4 --round 100
# CUDA_VISIBLE_DEVICES=0,1 python scaffold.py --epochs 8 --batch_size 4 --round 100
# CUDA_VISIBLE_DEVICES=0 python fedprox.py --epochs 8 --batch_size 4 --round 100
