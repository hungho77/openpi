#!/usr/bin/env bash
cd /mnt/data/sftp/data/hunght23/workspace/projects/openpi
export HF_LEROBOT_HOME=/mnt/data/sftp/data/hunght23/DATA_ZOO/so101_multitask_v21
export HF_TOKEN=$(/mnt/data/sftp/data/hunght23/workspace/projects/groot-n17/.venv/bin/python -c "from huggingface_hub import get_token;print(get_token())")
export OPENPI_DATA_HOME=/mnt/data/sftp/data/hunght23/MODEL_ZOO/openpi/cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec .venv/bin/python scripts/train_pytorch.py pi05_so101 --exp_name so101_bs128 --overwrite
