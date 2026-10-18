"""High-level policy that predicts the next subtask.

The high-level policy has the same architecture as pi0-FAST (an autoregressive PaliGemma model). It reads the camera
images, the task and the previous subtasks, and generates the text of the next subtask, which is then given to the
low-level policy (e.g. pi05 with episode metadata prompting). It is trained on the subtask segments of the annotations
file, for example collected by verbally coaching the low-level policy through a new task.
"""

import dataclasses

from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_fast


@dataclasses.dataclass(frozen=True)
class SubtaskPolicyConfig(pi0_fast.Pi0FASTConfig):
    # The model does not predict actions, a single step keeps the data pipeline cheap.
    action_horizon: int = 1
    max_token_len: int = 128

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.SUBTASK
