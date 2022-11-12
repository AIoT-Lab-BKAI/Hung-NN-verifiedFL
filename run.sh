# CUDA_VISIBLE_DEVICES=0 python fedavgv2.py --epochs 8 --batch_size 4 --round 100 --contrastive 0
CUDA_VISIBLE_DEVICES=0 python fedavgv2.py --epochs 8 --batch_size 4 --round 100 --contrastive 1

# CUDA_VISIBLE_DEVICES=0,1 python feddyn.py --epochs 8 --batch_size 4 --round 100
# CUDA_VISIBLE_DEVICES=0,1 python fedavg.py --epochs 8 --batch_size 4 --round 100
# CUDA_VISIBLE_DEVICES=0,1 python scaffold.py --epochs 8 --batch_size 4 --round 100
# CUDA_VISIBLE_DEVICES=0 python fedprox.py --epochs 8 --batch_size 4 --round 100
# CUDA_VISIBLE_DEVICES=0,1 python singleset.py

# python records/aplot.py --folder fedavg
# python records/aplot.py --folder feddyn
# python records/aplot.py --folder proposal_ideal
# python records/aplot.py --folder scaffold
# python records/aplot.py --folder fedprox
