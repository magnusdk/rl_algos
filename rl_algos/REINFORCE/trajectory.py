from collections import namedtuple
from typing import List

import numpy as np

TrajectoryItem = namedtuple("TrajectoryItem", ["s", "a", "r", "next_s"])


class Trajectory:
    """Utility class for keeping track of and working with trajectories.

    Implements __getitem__ and __len__, so works kind of like a list."""

    def __init__(self, traj: List[TrajectoryItem] = []):
        self._traj = traj

    def append(self, s, a, r, next_s):
        self._traj.append(TrajectoryItem(s=s, a=a, r=r, next_s=next_s))

    @property
    def s(self):
        return np.array([s for s, _, _, _ in self._traj])

    @property
    def a(self):
        return np.array([a for _, a, _, _ in self._traj])

    @property
    def r(self):
        return np.array([r for _, _, r, _ in self._traj])

    def __getitem__(self, key):
        res = self._traj.__getitem__(key)
        if isinstance(res, list):
            res = Trajectory(res)
        return res

    def __len__(self):
        return len(self._traj)

    def __repr__(self):
        return f"Trajectory({repr(self._traj)})"
