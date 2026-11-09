#!/bin/bash
# Fine-tuning experiments F1-F6: SegFormer 10 m, each run changes ONE setting compared with the baseline.
cd /home/jovyan/vault/heidelberg_lulc_2025 && source env.sh
mkdir -p logs/finetune
B="python3 scripts/train.py --arch segformer --train-res 10 --gpu 0"
nohup $B --no-pretrain        --run-name segformer_train10_ft-F1-scratch        > logs/finetune/F1.out 2>&1 &
nohup $B --encoder mit_b0     --run-name segformer_train10_ft-F2-mitb0          > logs/finetune/F2.out 2>&1 &
nohup $B --encoder mit_b4     --run-name segformer_train10_ft-F3-mitb4          > logs/finetune/F3.out 2>&1 &
nohup $B --scheduler cosine   --run-name segformer_train10_ft-F4-cosine         > logs/finetune/F4.out 2>&1 &
nohup $B --lr 1e-4            --run-name segformer_train10_ft-F5-lr1e-4         > logs/finetune/F5.out 2>&1 &
nohup $B --class-weight none  --run-name segformer_train10_ft-F6-noclassweight  > logs/finetune/F6.out 2>&1 &
sleep 5; echo "started:"; pgrep -af "ft-F" | sed 's/.*--run-name //'
