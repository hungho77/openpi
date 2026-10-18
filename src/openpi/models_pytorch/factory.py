"""Creates PyTorch models from openpi model configs and loads pretrained weights."""

import logging
import re

import safetensors.torch
import torch

logger = logging.getLogger("openpi")

# Weights that are not in the pi0 / pi05 checkpoints and keep their initial values.
DEFAULT_ALLOWED_MISSING = (r".*history_.*", r"value_head\..*", r".*lm_head\..*")


def create_model(model_config) -> torch.nn.Module:
    """Creates the PyTorch model for the given model config (`openpi.models.model.BaseModelConfig`)."""
    # Imported here since `openpi.models.model` imports this module.
    from openpi.models import model as _model

    match model_config.model_type:
        case _model.ModelType.PI0 | _model.ModelType.PI05:
            from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

            return PI0Pytorch(model_config)
        case _model.ModelType.VALUE:
            from openpi.models_pytorch.value_pytorch import ValuePytorch

            return ValuePytorch(model_config)
        case _model.ModelType.SUBTASK:
            from openpi.models_pytorch.subtask_pytorch import SubtaskPytorch

            return SubtaskPytorch(model_config)
        case _:
            raise ValueError(f"Model type {model_config.model_type} is not supported in PyTorch.")


def load_pretrained_weights(
    model: torch.nn.Module, weight_path: str, *, allowed_missing: tuple[str, ...] = DEFAULT_ALLOWED_MISSING
) -> None:
    """Loads the weights with matching names and shapes, e.g. the backbone of a pi05 checkpoint into a value function.

    Weights of the checkpoint that do not exist in the model are ignored. Weights of the model that are missing from
    the checkpoint must match one of `allowed_missing`.
    """
    state = safetensors.torch.load_file(weight_path, device="cpu")
    model_state = model.state_dict()
    loaded = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
    # Tied weights (e.g. the input embeddings and the language model head) are saved only once.
    loaded_storages = {model_state[k].data_ptr() for k in loaded}
    missing = sorted(k for k in model_state if k not in loaded and model_state[k].data_ptr() not in loaded_storages)
    not_allowed = [k for k in missing if not any(re.fullmatch(p, k) for p in allowed_missing)]
    if not_allowed:
        raise ValueError(f"{len(not_allowed)} weights are missing from {weight_path}, e.g. {not_allowed[:5]}")
    model.load_state_dict(loaded, strict=False)
    ignored = len(state) - len(loaded)
    logger.info(
        f"Loaded {len(loaded)} weights from {weight_path} ({len(missing)} kept at initialization: "
        f"{missing[:5]}{'...' if len(missing) > 5 else ''}; {ignored} unused weights in the checkpoint)."
    )
