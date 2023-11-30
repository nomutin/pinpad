"""
Visual Pin Pad benchmark gymnasium re-implementation.

References
----------
- Deep Hierarchical Planning from Pixels [Hafner+ NeurIPS2022]
- [Original Implementation](https://github.com/danijar/director)

"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import Env, spaces

from .types import (
    ACTION,
    BLACK,
    COLORS,
    GRAY,
    INFO,
    LAYOUTS,
    OBSERVATION,
    REWARD,
    TERMINATED,
    TRUNCATED,
    WHITE,
)

__all__ = ["PinPad"]


class PinPad(Env):
    """
    Visual Pin Pad benchmark environment.

    - :meth:`step`
        - Updates an environment with actions returning the next observation.
    - :meth:`reset`
        - Resets the environment to an initial state.
        - Returns the first agent observation for an episode and information.
    - :meth:`render`
        - Renders the environments to help visualize what the agent see.
    """

    action_space = spaces.Discrete(5)
    observation_space = spaces.Box(0, 255, shape=(3,), dtype=np.uint8)
    render_mode = "rgb_array"
    _np_random: np.random.Generator = np.random.default_rng()

    def __init__(self, layout: str, target: list[str]) -> None:
        """Initialize the environment."""
        super().__init__()
        self.layout = np.array([list(line) for line in layout.split("\n")]).T
        self.pads = set(self.layout.flatten().tolist()) - set("* #\n")
        self.target = target

        self.random = np.random.RandomState()
        self.spawns = []
        for (x, y), char in np.ndenumerate(self.layout):
            if char != "#":
                self.spawns.append((x, y))
        self.sequence: list[str] = []
        self.reset()

    def step(
        self,
        action: ACTION,
    ) -> tuple[OBSERVATION, REWARD, TERMINATED, TRUNCATED, INFO]:
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
        observation : Observation
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
        if self.sequence[-7:] == self.target:
            reward += 10.0

        observation = self.render()
        terminated = False
        truncated = False
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[OBSERVATION, INFO]:
        """
        Reset the environment.

        Parameters
        ----------
        seed : int, optional
            seed for random number generator
        options : dict[str, Any], optional
            Explicitly specify the next spawn location with the "player" key
        """
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        self.player = self._np_random.choice(self.spawns)
        if options is not None and "player" in options:
            self.player = options["player"]
        self.sequence.clear()
        info = self._get_info()
        return self.render(), info

    def render(self) -> OBSERVATION:  # type: ignore[override]
        """Render the environment."""
        grid = np.zeros((16, 16, 3), np.uint8) + 255

        current = self.layout[self.player[0]][self.player[1]]
        for (x, y), char in np.ndenumerate(self.layout):
            if char == "#":
                grid[x, y] = GRAY
            elif char in self.pads:
                color = np.array(COLORS[char])
                if char != current:
                    color = (10 * color + 90 * WHITE) / 100
                grid[x, y] = color
        grid[self.player] = BLACK
        grid[:, -2:] = GRAY
        for i, char in enumerate(self.sequence[-7:]):
            grid[2 * i + 1, -2] = COLORS[char]
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
        with path.open(mode="r") as f:
            return cls(layout=f.read().strip("\n"), target=[])
