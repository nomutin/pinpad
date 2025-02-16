# Copyright 2022 Danijar Hafner.
"""
Visual Pin Pad benchmark gymnasium re-implementation.

References
----------
- Deep Hierarchical Planning from Pixels [Hafner+ NeurIPS2022]
- [Original Implementation](https://github.com/danijar/director)

"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Literal

import numpy as np
from gymnasium import Env, spaces
from numpy.random import Generator, RandomState, default_rng
from numpy.typing import NDArray
from typing_extensions import Self, TypeAlias

ACTION: TypeAlias = NDArray[np.int_]
OBSERVATION: TypeAlias = NDArray[np.int_]
REWARD: TypeAlias = float
TERMINATED: TypeAlias = bool
TRUNCATED: TypeAlias = bool
INFO: TypeAlias = Dict[str, Any]
LAYOUTS: TypeAlias = Literal["three", "four", "five", "six", "seven", "eight"]


@dataclass
class ColorData:
    """Color Data."""

    index: str
    rgb: NDArray[np.int_]


class Color(ColorData, Enum):
    """Color."""

    RED = ("1", np.array([255, 0, 0]))
    GREEN = ("2", np.array([0, 255, 0]))
    BLUE = ("3", np.array([0, 0, 255]))
    YELLOW = ("4", np.array([255, 255, 0]))
    MAGENTA = ("5", np.array([255, 0, 255]))
    CYAN = ("6", np.array([0, 255, 255]))
    PURPLE = ("7", np.array([128, 0, 128]))
    TEAL = ("8", np.array([0, 128, 128]))
    WHITE = ("x", np.array([255, 255, 255]))
    GRAY = ("xx", np.array([192, 192, 192]))
    BLACK = ("xxx", np.array([0, 0, 0]))

    def __new__(cls, index: str, *_: NDArray[np.int_]) -> Self:
        """Create a new instance of Color."""
        obj = ColorData.__new__(cls)
        obj._value_ = index
        return obj


class PinPad(Env[OBSERVATION, ACTION]):
    """
    Visual Pin Pad benchmark environment.

    Methods
    -------
    step
        Updates an environment with actions returning the next observation.
    reset
        Resets the environment to an initial state.
        Returns the first agent observation for an episode and information.
    render
        Renders the environments to help visualize what the agent see.

    Parameters
    ----------
    layout : str
        The layout to be used for the PinPad.
    targets : list[list[str]]
        The targets to be used for the PinPad.
    """

    action_space: spaces.Space = spaces.Discrete(5)  # type: ignore[type-arg]
    observation_space = spaces.Box(0, 255, shape=(3,), dtype=np.uint8)
    render_mode = "rgb_array"
    _np_random: Generator = default_rng()

    def __init__(self, layout: str, targets: list[list[str]]) -> None:
        super().__init__()
        self.layout = np.array([list(line) for line in layout.split("\n")]).T
        self.pads = set(self.layout.flatten().tolist()) - set("* #\n")
        self.targets = targets

        self.random = RandomState()
        self.spawns = []
        for (x, y), char in np.ndenumerate(self.layout):
            if char != "#":
                self.spawns.append((x, y))
        self.sequence: list[str] = []
        self.reset()

    def step(self, action: ACTION) -> tuple[OBSERVATION, REWARD, TERMINATED, TRUNCATED, INFO]:
        """
        Run one time step of the environment's dynamics.

        When end of episode is reached, you are responsible for calling
        `reset()` to reset this environment's state.

        Parameters
        ----------
        action : ACTION
            an action provided by the agent

        Returns
        -------
        observation : OBSERVATION
            agent's observation of the current environment
        reward : REWARD
            amount of reward returned after previous action
        terminated : TERMINATED
            Whether the agent reaches the terminal state.
            NOT used in this environment.
        truncated : TRUNCATED
            Whether the truncation condition outside the scope of
            the MDP is satisfied.
            NOT used in this environment.
        info : INFO
            contains player position and sequence
        """
        reward = 0.0
        move = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)][action]
        x = np.clip(self.player[0] + move[0], 0, 15)
        y = np.clip(self.player[1] + move[1], 0, 13)
        tile = self.layout[x][y]
        if tile != "#":
            self.player = (x, y)

        is_new_tile = not self.sequence or self.sequence[-1] != tile
        if tile in self.pads and is_new_tile:
            self.sequence.append(tile[0])
        if self.sequence in self.targets:
            reward = 1.0

        observation = self.render()
        terminated = False
        truncated = False
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[OBSERVATION, INFO]:
        """
        Reset the environment.

        Parameters
        ----------
        seed : int | None
            seed for random number generator
        options : dict[str, Any] | None
            Explicitly specify the next spawn location with the "player" key.

        Returns
        -------
        tuple[OBSERVATION, INFO]
            OBSERVATION
                the initial observation of the space.
            INFO
                contains NEW player position and PREVIOUS sequence
        """
        if seed is not None:
            self._np_random = default_rng(seed)
        self.player = tuple(self._np_random.choice(self.spawns))
        if options is not None and "player" in options:
            self.player = options["player"]
        info = self._get_info()
        self.sequence.clear()
        return self.render(), info

    def render(self) -> OBSERVATION:  # type: ignore[override]
        """
        Render the environment.

        Returns
        -------
        OBSERVATION
            the current observation of the space.
        """
        grid = np.zeros((16, 16, 3), np.uint8) + 255

        current = self.layout[self.player[0]][self.player[1]]
        for (x, y), char in np.ndenumerate(self.layout):
            if char == "#":
                grid[x, y] = Color.GRAY.rgb
            elif char in self.pads:
                color = Color(char).rgb
                if char != current:
                    color = (0.1 * color + 0.9 * Color.WHITE.rgb).astype(np.uint8)
                grid[x, y] = color
        grid[self.player] = Color.BLACK.rgb
        grid[:, -2:] = Color.GRAY.rgb
        for i, char in enumerate(self.sequence[-7:]):
            grid[2 * i + 1, -2] = Color(char).rgb
        image = np.repeat(np.repeat(grid, 4, 0), 4, 1)
        return image.transpose((1, 0, 2))

    @property
    def player(self) -> tuple[int, int]:
        """Get the current player position."""
        return self._player

    @player.setter
    def player(self, value: tuple[int, int]) -> None:
        """Set the current player position."""
        self._player = value

    def _get_info(self) -> INFO:
        """
        Get current object/environment information.

        Returns
        -------
        INFO
            A dictionary containing the player position and sequence.
        """
        return {
            "player": self.player,
            "sequence": "".join(self.sequence),
        }

    @classmethod
    def make(cls, layout: LAYOUTS) -> PinPad:
        """
        Class method to create a new instance of PinPad.

        Parameters
        ----------
        layout : LAYOUTS
            The layout to be used for the PinPad.

        Returns
        -------
        PinPad
            A new instance of PinPad with the specified layout.
        """
        path = Path(__file__).parent / "layouts" / f"{layout}.txt"
        return cls(layout=path.read_text().strip("\n"), targets=[])
