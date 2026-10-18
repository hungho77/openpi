#!/usr/bin/env bash
# Runs a command with only the device of the current torchrun rank visible.
#
# A CUDA process can only use a single MIG instance, so for multi-GPU training on MIG instances every rank must see
# only its own instance. Set ALL_DEVICES to the comma separated devices of the job (e.g. the CUDA_VISIBLE_DEVICES set
# by Slurm) and run this script with torchrun:
#
#   ALL_DEVICES=$CUDA_VISIBLE_DEVICES uv run torchrun --standalone --nproc_per_node=2 --no-python \
#       scripts/per_device.sh python scripts/train_pytorch.py <config_name> --exp_name <run_name>
set -euo pipefail
IFS=',' read -ra DEVICES <<< "${ALL_DEVICES:?Set ALL_DEVICES to the comma separated devices of the job.}"
if (( LOCAL_RANK >= ${#DEVICES[@]} )); then
    echo "LOCAL_RANK=${LOCAL_RANK} but only ${#DEVICES[@]} devices in ALL_DEVICES=${ALL_DEVICES}" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="${DEVICES[$LOCAL_RANK]}"
exec "$@"
