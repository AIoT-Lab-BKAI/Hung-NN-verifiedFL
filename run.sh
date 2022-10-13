# CUDA_VISIBLE_DEVICES=1,3 python proposal.py --epochs 8 --batch_size 4 --round 50
CUDA_VISIBLE_DEVICES=1,3 python feddyn.py --epochs 8 --batch_size 4 --round 100
# CUDA_VISIBLE_DEVICES=1,3 python fedavg.py --epochs 8 --batch_size 4 --round 100
# CUDA_VISIBLE_DEVICES=1,3 python scaffold.py --epochs 8 --batch_size 4 --round 1
# CUDA_VISIBLE_DEVICES=1,3 python singleset.py