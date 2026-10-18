import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# 5 arm joints followed by the gripper (absolute joint positions).
ACTION_DIM = 6


def make_so101_example() -> dict:
    """Creates a random input example for the SO101 policy."""
    return {
        "observation/state": np.random.rand(ACTION_DIM),
        "observation/image": np.random.randint(256, size=(360, 640, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "pick up the red cube and put it into the pot",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    """Converts the inputs of a single SO101 arm with a top and a wrist camera to the model format.

    Used for both training and inference. Expected keys: "observation/image" (top camera),
    "observation/wrist_image", "observation/state" (6 joint positions) and optionally "actions" and "prompt".
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": np.asarray(data["observation/state"]),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # There is no second wrist camera.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Padding images are only masked for pi0 models, not pi0-FAST style models.
                "right_wrist_0_rgb": np.True_
                if self.model_type in (_model.ModelType.PI0_FAST, _model.ModelType.SUBTASK)
                else np.False_,
            },
        }

        # Actions are only available during training, or as the action prefix for real-time chunking.
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    """Returns the 6 SO101 action dimensions (the model actions are padded)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :ACTION_DIM])}
