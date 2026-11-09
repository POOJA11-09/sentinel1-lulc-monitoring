#!/bin/bash
# Copies the evaluation results used by the app into prototype/evaluation/ so the app also works on a laptop.
P=/home/jovyan/vault/heidelberg_lulc_2025
E=$P/prototype/evaluation
B=segformer_train10_VV-VH-RATIO_s42
mkdir -p $E/train $E/val $E/ablation/val $E/$B $E/diagnostics/weak_classes/$B $E/figures
cp $P/outputs/evaluation/summary_overall.csv $P/outputs/evaluation/summary_per_class.csv $E/
cp $P/outputs/evaluation/train/summary_*.csv $E/train/
cp $P/outputs/evaluation/val/summary_*.csv $E/val/
cp $P/outputs/evaluation/ablation/summary_*.csv $E/ablation/
cp $P/outputs/evaluation/ablation/val/summary_*.csv $E/ablation/val/
cp $P/outputs/evaluation/$B/confusion_test10.csv $E/$B/
cp $P/outputs/diagnostics/weak_classes/$B/*.csv $P/outputs/diagnostics/weak_classes/$B/*.png $E/diagnostics/weak_classes/$B/ 2>/dev/null
cp $P/outputs/figures/*.png $E/figures/ 2>/dev/null
echo "Copied into $E:"; find $E -type f | sed "s|$E/||" | sort
