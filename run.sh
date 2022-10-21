CUDA_VISIBLE_DEVICES=1,3 python proposal4.py --epochs 8 --batch_size 4 --round 10 --lr 0.01 > log.txt &
# CUDA_VISIBLE_DEVICES=1,3 python feddyn.py --epochs 8 --batch_size 4 --round 100
# CUDA_VISIBLE_DEVICES=1,3 python fedavg.py --epochs 8 --batch_size 4 --round 100
# CUDA_VISIBLE_DEVICES=1,3 python scaffold.py --epochs 8 --batch_size 4 --round 100
# CUDA_VISIBLE_DEVICES=1,3 python fedprox.py --epochs 8 --batch_size 4 --round 100
# CUDA_VISIBLE_DEVICES=1,3 python singleset.py

# python records/aplot.py --folder fedavg
# python records/aplot.py --folder feddyn
# python records/aplot.py --folder proposal_ideal
# python records/aplot.py --folder scaffold
# python records/aplot.py --folder fedprox
