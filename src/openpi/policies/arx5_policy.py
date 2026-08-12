"""OpenPI transforms for ARX5 single-arm policies in absolute EEF-pose space."""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class ARX5SingleArmEefInputs(transforms.DataTransformFn):
    """Map the local single-arm LeRobot feature names into pi0.5 inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class ARX5SingleArmEefOutputs(transforms.DataTransformFn):
    """Return ARX5's 10-D absolute EEF-pose action from the padded model action."""

    def __call__(self, data: dict) -> dict:
        # [x, y, z, rot6d_0..rot6d_5, gripper]
        return {"actions": np.asarray(data["actions"][..., :10])}


ARX5LeftEefInputs = ARX5SingleArmEefInputs
ARX5LeftEefOutputs = ARX5SingleArmEefOutputs
