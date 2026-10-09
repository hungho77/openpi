#!/usr/bin/env bash
# Single-MIG-slice variant of run_so101_leftarm.sh.
#
# Use this when the other slice is occupied. DDP needs ~35.3GB per rank at global batch
# 32, and the fixed cost of weights + grads + AdamW state is ~29GB of a 39.5GB slice, so
# any co-tenant holding several GB makes the two-slice run impossible rather than slower.
#
# Overrides instead of a second TrainConfig: batch 16 is the single-slice ceiling
# (measured 34.5GB reserved), and 48_000 x 16 = 768_000 samples keeps the same sample
# budget as the committed two-slice plan and as the right-arm run, so the checkpoints stay
# comparable. decay_steps must move with num_train_steps or the cosine never decays.
cd /mnt/data/sftp/data/hunght23/workspace/projects/openpi
export HF_LEROBOT_HOME=/mnt/data/sftp/data/hunght23/DATA_ZOO/so101_leftarm_multitask_v21
export OPENPI_DATA_HOME=/mnt/data/sftp/data/hunght23/MODEL_ZOO/openpi/cache
export HF_TOKEN=$(/mnt/data/sftp/data/hunght23/workspace/projects/groot-n17/.venv/bin/python -c "from huggingface_hub import get_token;print(get_token())")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${MIG_UUID:-MIG-0073bead-8aa1-531c-8ec8-5fa67bf8e567}
exec .venv/bin/python scripts/train_pytorch.py pi05_so101_leftarm_multitask \
  --exp_name "${EXP_NAME:-leftarm_1gpu_bs16}" \
  --batch_size 16 \
  --num_train_steps 48000 \
  --lr-schedule.decay-steps 48000 \
  --overwrite
