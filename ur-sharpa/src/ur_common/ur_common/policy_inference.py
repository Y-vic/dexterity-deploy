"""Lossless JSON representation for policy response arrays and request context."""
import math

import numpy as np


def inference_json(value):
    if isinstance(value, np.ndarray):
        return dict(dtype=value.dtype.str, shape=list(value.shape), data=inference_json(value.tolist()))
    if isinstance(value, np.generic):
        return inference_json(value.item())
    if isinstance(value, dict):
        return {str(key): inference_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [inference_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return dict(nonfinite=repr(value))
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f'unsupported inference result type: {type(value).__name__}')
