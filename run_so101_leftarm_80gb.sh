#!/usr/bin/env bash
# pi0.5 on the 11-task SO101 left-arm multitask set, single undivided 80GB GPU.
#
# batch 128 is only reachable here. Under DDP the optimizer is replicated per rank
# (openpi's PyTorch trainer has no FSDP) and the fixed cost of weights + grads + AdamW
# state is ~29GB, so two 40GB MIG slices top out at a global batch of 48 no matter how
# free they are. One 80GB device fits 128 at ~71GB, measured on the right-arm run.
#
# 6_000 x 128 = 768_000 samples: the same batch size AND step count as the right-arm
# pi05_so101 run, so the two checkpoints are directly comparable with nothing to explain
# away. decay_steps moves with num_train_steps or the cosine never decays.
#
# Everything this needs is on the shared NFS mount, so no files have to be copied: source,
# .venv, dataset, base weights and norm stats all live under /mnt/data/sftp/data/hunght23.
cd /mnt/data/sftp/data/hunght23/workspace/projects/openpi
export HF_LEROBOT_HOME=/mnt/data/sftp/data/hunght23/DATA_ZOO/so101_leftarm_multitask_v21
export OPENPI_DATA_HOME=/mnt/data/sftp/data/hunght23/MODEL_ZOO/openpi/cache
export HF_TOKEN=$(/mnt/data/sftp/data/hunght23/workspace/projects/groot-n17/.venv/bin/python -c "from huggingface_hub import get_token;print(get_token())")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec .venv/bin/python scripts/train_pytorch.py pi05_so101_leftarm_multitask \
  --exp_name "${EXP_NAME:-leftarm_80gb_bs128}" \
  --batch_size 128 \
  --num_train_steps 6000 \
  --lr-schedule.decay-steps 6000 \
  --overwrite
