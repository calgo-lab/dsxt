Data Science Toolkit
====================

A collection of tools/extension for data science.

Installation of Python environment
----------------------------------

In order to set up the necessary environment:

0. install [Poetry](https://python-poetry.org/)
1. install dependencies (https://python-poetry.org/docs/basic-usage/#installing-dependencies)
    ```
    poetry install
    ```
2. activate the new environment (https://python-poetry.org/docs/basic-usage/#activating-the-virtual-environment)
    ```
    source `poetry env info --path`/bin/activate
    ```

Optional and needed only once after `git clone`:

4. install several [pre-commit] git hooks with:
    ```
    pre-commit install
    ```
    and checkout the configuration under `.pre-commit-config.yaml`.
    The `-n, --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks temporarily.
