#!/usr/bin/env bash
cd /mnt/data/sftp/data/hunght23/workspace/projects/openpi
# pi0.5 default recipe (run_so101_leftarm_80gb.sh) on the 3-task SO101 left-arm set; batch/steps/decay from the config.
export HF_LEROBOT_HOME=/mnt/data/sftp/data/hunght23/DATA_ZOO/so101_leftarm_task3_v21
export OPENPI_DATA_HOME=/mnt/data/sftp/data/hunght23/MODEL_ZOO/openpi/cache
export HF_TOKEN=$(/mnt/data/sftp/data/hunght23/workspace/projects/groot-n17/.venv/bin/python -c "from huggingface_hub import get_token;print(get_token())")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec .venv/bin/python scripts/train_pytorch.py pi05_so101_leftarm_task3 \
  --exp_name "${EXP_NAME:-task3_80gb_bs128}" \
  --overwrite
