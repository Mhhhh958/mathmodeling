from __future__ import annotations

import json
import numpy as np

# Q3B produces a validation structure that can contain numpy scalar booleans/numbers.
# Normalize only those scalars at JSON serialization time; do not change any experiment logic.
_original_dumps = json.dumps


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _safe_dumps(*args, **kwargs):
    kwargs.setdefault("default", _json_default)
    return _original_dumps(*args, **kwargs)


json.dumps = _safe_dumps

import q3b_unsupervised_transfer as q3b


if __name__ == "__main__":
    q3b.main()
