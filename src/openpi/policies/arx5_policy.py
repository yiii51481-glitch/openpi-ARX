"""OpenPI transforms for a single left ARX5 arm in absolute EEF-pose space."""

import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import libero_policy


@dataclasses.dataclass(frozen=True)
class ARX5LeftEefInputs(transforms.DataTransformFn):
    """Map the local LeRobot feature names into the model input dictionary."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # The dataset uses the same camera and state keys as the LIBERO LeRobot example. Its input
        # transform also zero-pads the unavailable right-wrist image for pi0.5.
        return libero_policy.LiberoInputs(model_type=self.model_type)(data)


@dataclasses.dataclass(frozen=True)
class ARX5LeftEefOutputs(transforms.DataTransformFn):
    """Return ARX5's 10-D absolute EEF-pose action from the padded model action."""

    def __call__(self, data: dict) -> dict:
        # [x, y, z, rot6d_0..rot6d_5, gripper]
        return {"actions": np.asarray(data["actions"][..., :10])}
