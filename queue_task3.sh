#!/bin/bash
# pi0.5 default on the 3-task SO101 left-arm set (user request 2026-10-30): norm stats, then train.
cd /mnt/data/sftp/data/hunght23/workspace/projects/openpi
LOG=queue_task3.log
log() { echo "$1 $(date +%F_%T)" >> $LOG; }
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
log "start norm stats"
HF_LEROBOT_HOME=/mnt/data/sftp/data/hunght23/DATA_ZOO/so101_leftarm_task3_v21 \
  .venv/bin/python scripts/compute_norm_stats.py --config-name pi05_so101_leftarm_task3 > norm_task3.log 2>&1 \
  || { log "norm stats failed"; exit 1; }
log "end norm stats"
log "start train"
./run_so101_task3_80gb.sh > train_task3.log 2>&1
log "end train exit=$?"
