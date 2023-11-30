# Visual Pin Pad

![python](https://img.shields.io/badge/python-3.8-blue)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ci](https://github.com/nomutin/pinpad/actions/workflows/ci.yaml/badge.svg)](https://github.com/nomutin/pinpad/actions/workflows/ci.yaml)

![pinpad](https://ar5iv.labs.arxiv.org/html/2206.04114/assets/x7.png)

[Gymnasium](https://github.com/Farama-Foundation/Gymnasium) re-implementation of visual pin pad benchmark.

## Usage

```python
from pinpad import PinPad

env = PinPad.make(layout="eight")
env.reset()

for _ in range(100):
    action = env.action_space.sample()
    observation, reward, *_ = env.step(action=action)
```

## Credits

This project uses code from [danijar/director](https://github.com/danijar/director).
