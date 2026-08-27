# Security

Never commit `.env`, Sentinel Hub credentials, private coordinates, source datasets, or model checkpoints.

If a credential is accidentally committed, revoke or rotate it immediately in Sentinel Hub, remove it from the current source, and clean it from Git history before making the repository public.

Only load PyTorch model weights from a trusted source. The application uses `torch.load(..., weights_only=True)` to reduce checkpoint deserialization risk.
