# CUDA_VISIBLE_DEVICES=1,3 python train_smt.py --epochs 8 --batch_size 4 --round 0
# CUDA_VISIBLE_DEVICES=1,3 python aggregate.py --round 0

# CUDA_VISIBLE_DEVICES=1,3 python train_smt.py --epochs 8 --batch_size 4 --round 1
# CUDA_VISIBLE_DEVICES=1,3 python aggregate.py --round 1

# CUDA_VISIBLE_DEVICES=1,3 python train_smt.py --epochs 8 --batch_size 4 --round 2
# CUDA_VISIBLE_DEVICES=1,3 python aggregate.py --round 2

# CUDA_VISIBLE_DEVICES=1,3 python train_smt.py --epochs 8 --batch_size 4 --round 3
# CUDA_VISIBLE_DEVICES=1,3 python aggregate.py --round 3

# CUDA_VISIBLE_DEVICES=1,3 python train_smt.py --epochs 8 --batch_size 4 --round 4
# CUDA_VISIBLE_DEVICES=1,3 python aggregate.py --round 4

# CUDA_VISIBLE_DEVICES=1,3 python train_smt.py --epochs 8 --batch_size 4 --round 5
# CUDA_VISIBLE_DEVICES=1,3 python aggregate.py --round 5

# CUDA_VISIBLE_DEVICES=1,3 python train_smt.py --epochs 8 --batch_size 4 --round 6
# CUDA_VISIBLE_DEVICES=1,3 python aggregate.py --round 6

# CUDA_VISIBLE_DEVICES=1,3 python train_smt.py --epochs 8 --batch_size 4 --round 7
# CUDA_VISIBLE_DEVICES=1,3 python aggregate.py --round 7

# CUDA_VISIBLE_DEVICES=1,3 python train_smt.py --epochs 8 --batch_size 4 --round 8
# CUDA_VISIBLE_DEVICES=1,3 python aggregate.py --round 8

# CUDA_VISIBLE_DEVICES=1,3 python train_smt.py --epochs 8 --batch_size 4 --round 9
# CUDA_VISIBLE_DEVICES=1,3 python aggregate.py --round 9

# CUDA_VISIBLE_DEVICES=1,3 python train_smt.py --epochs 8 --batch_size 4 --round 10
# CUDA_VISIBLE_DEVICES=1,3 python aggregate.py --round 10

# CUDA_VISIBLE_DEVICES=1,3 python train_smt.py --epochs 8 --batch_size 4 --round 11
# CUDA_VISIBLE_DEVICES=1,3 python aggregate.py --round 11

# CUDA_VISIBLE_DEVICES=1,3 python train_smt.py --epochs 8 --batch_size 4 --round 12
# CUDA_VISIBLE_DEVICES=1,3 python aggregate.py --round 12

# CUDA_VISIBLE_DEVICES=1,3 python train_smt.py --epochs 8 --batch_size 4 --round 13
# CUDA_VISIBLE_DEVICES=1,3 python aggregate.py --round 13

# CUDA_VISIBLE_DEVICES=1,3 python train_smt.py --epochs 8 --batch_size 4 --round 14
# CUDA_VISIBLE_DEVICES=1,3 python aggregate.py --round 14

# CUDA_VISIBLE_DEVICES=1,3 python train_smt.py --epochs 8 --batch_size 4 --round 15
# CUDA_VISIBLE_DEVICES=1,3 python aggregate.py --round 15

# CUDA_VISIBLE_DEVICES=1,3 python train_smt.py --epochs 8 --batch_size 4 --round 16
# CUDA_VISIBLE_DEVICES=1,3 python aggregate.py --round 16
# python plot.py --folder models

CUDA_VISIBLE_DEVICES=1,3 python proposal.py --epochs 8 --batch_size 4 --round 100
CUDA_VISIBLE_DEVICES=1,3 python fedavg.py --epochs 8 --batch_size 4 --round 100
CUDA_VISIBLE_DEVICES=1,3 python scaffold.py --epochs 8 --batch_size 4 --round 1
CUDA_VISIBLE_DEVICES=1,3 python singleset.py