"""Type stubs and constants."""

from typing import Any, Dict, Literal

import numpy as np
import numpy.typing as npt

ACTION = npt.ArrayLike
OBSERVATION = npt.ArrayLike
REWARD = float
TERMINATED = bool
TRUNCATED = bool
INFO = Dict[str, Any]

LAYOUTS = Literal["three", "four", "five", "six", "seven", "eight"]
WHITE = np.array([255, 255, 255])
GRAY = np.array([192, 192, 192])
BLACK = np.array([0, 0, 0])
RED = np.array([255, 0, 0])
GREEN = np.array([0, 255, 0])
BLUE = np.array([0, 0, 255])
YELLOW = np.array([255, 255, 0])
MAGENTA = np.array([255, 0, 255])
CYAN = np.array([0, 255, 255])
PURPLE = np.array([128, 0, 128])
TEAL = np.array([0, 128, 128])
COLORS = {
    "1": RED,
    "2": GREEN,
    "3": BLUE,
    "4": YELLOW,
    "5": MAGENTA,
    "6": CYAN,
    "7": PURPLE,
    "8": TEAL,
}
