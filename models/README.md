# Model weights

Model checkpoints are not committed because the original PyTorch files are about 80 MB each and are runtime artifacts.

Place trusted state dictionaries here using these default names:

```text
models/
├── classification_1km.pth
├── classification_2km.pth
├── classification_3km.pth
└── regression.pth
```

The desktop app uses `classification_3km.pth` by default. Set `SOIL_MODEL_PATH` in `.env` to choose a different trusted checkpoint.

Only load model weights from a source you trust. PyTorch checkpoint formats can execute unsafe pickle payloads in older loading modes; the application and evaluation notebooks use `weights_only=True`.
