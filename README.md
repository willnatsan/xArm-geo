# xArm Geometric Modelling & Control

## Development Setup

This repository uses [pre-commit](https://pre-commit.com) to autoformat C/C++
(via `clang-format`) and Python (via `ruff`) on every commit.

After cloning, install the lint environment and the git hook once:

```bash
pixi install -e lint
pixi run -e lint install-hooks
```

From then on, `git commit` will automatically format staged files. If a hook
modifies a file, the commit is aborted; re-stage the changes and commit again.

To run all hooks against the entire repository on demand:

```bash
pixi run -e lint lint
```

## Documentation

Authoring guides for using the library:

- [`docs/authoring_controllers.md`](docs/authoring_controllers.md) — writing custom controllers
- [`docs/authoring_trajectories.md`](docs/authoring_trajectories.md) — writing custom trajectories
