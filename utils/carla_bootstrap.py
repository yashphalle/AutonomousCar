

import glob
import os
import sys


def _find_carla_root():
    candidates = []
    env = os.environ.get("CARLA_ROOT")
    if env:
        candidates.append(env)
    for pattern in ("~/CARLA_*", "~/Downloads/CARLA_*", "/opt/CARLA_*"):
        candidates += sorted(glob.glob(os.path.expanduser(pattern)), reverse=True)
    candidates.append("/opt/carla-simulator")

    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, "PythonAPI", "carla", "agents")):
            return candidate
    return None


_carla_root = _find_carla_root()
if _carla_root is None:
    raise RuntimeError(
        "Could not locate CARLA. Set CARLA_ROOT to your install directory:\n"
        "  export CARLA_ROOT=/path/to/CARLA_0.9.x"
    )

_agents_path = os.path.join(_carla_root, "PythonAPI", "carla")
if _agents_path not in sys.path:
    sys.path.insert(0, _agents_path)
