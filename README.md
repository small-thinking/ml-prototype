# ml-prototype
The prototype of ML models.

The project is implemented based on Torch Lightning.

Install the dependencies with [uv](https://github.com/astral-sh/uv):
```bash
uv sync
```

### Typical `uv` commands

`uv` manages dependencies declared in `pyproject.toml` and keeps the
`uv.lock` file in sync.

- **Add a runtime package**
  ```bash
  uv add <package>
  ```

- **Add a development package**
  ```bash
  uv add --dev <package>
  ```

- **Remove a package**
  ```bash
  uv remove <package>
  ```

- **Install dev dependencies and run tests**
  ```bash
  uv sync
  pytest
  ```

The config file is in ml_prototype/config/transformer_lm.yaml.

Run the training job with the following command:

```bash
python ml_prototype/cli.py fit --config ml_prototype/config/transformer_lm.yaml
```
