#!/usr/bin/env bash
# pi0.5 on the 11-task SO101 left-arm multitask set, DDP across two MIG 3g.40gb slices.
#
# CUDA_VISIBLE_DEVICES must name the MIG instance UUIDs, not 0,1 — with MIG enabled a
# plain ordinal addresses the parent H100, which has no compute instance assigned to it.
#
# TORCH_NCCL_BLOCKING_WAIT=1 is deliberate: a first attempt at this run sat for 44 minutes
# past the data loader without completing step 0 and without any error. It has not
# reproduced since this was set. The cause was never identified, so this stays until it is:
# blocking wait makes a stalled collective fail loudly instead of spinning silently.
cd /mnt/data/sftp/data/hunght23/workspace/projects/openpi
export HF_LEROBOT_HOME=/mnt/data/sftp/data/hunght23/DATA_ZOO/so101_leftarm_multitask_v21
export OPENPI_DATA_HOME=/mnt/data/sftp/data/hunght23/MODEL_ZOO/openpi/cache
export HF_TOKEN=$(/mnt/data/sftp/data/hunght23/workspace/projects/groot-n17/.venv/bin/python -c "from huggingface_hub import get_token;print(get_token())")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_VISIBLE_DEVICES=MIG-0073bead-8aa1-531c-8ec8-5fa67bf8e567,MIG-5a896a39-1002-52e3-a999-a46ba02cecd1
exec .venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_pytorch.py pi05_so101_leftarm_multitask \
  --exp_name "${EXP_NAME:-leftarm_bs32}" --overwrite
